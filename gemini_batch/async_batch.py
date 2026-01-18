"""
Async batch processing for OpenAI-compatible APIs.

Provides async_process() function with the same interface as batch_process(),
but uses OpenAI-compatible APIs (DeepSeek, Together, Groq, etc.) with true
asyncio concurrency.

Install with: pip install gemini-batch[async]
"""

import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from PIL import Image
from pydantic import BaseModel

from . import config
from .utils import extract_json_from_text

logger = logging.getLogger(__name__)


class AsyncOpenAIClient:
    """
    Async client wrapper for OpenAI-compatible APIs.

    Manages connection, rate limiting via semaphore, and lazy initialization.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_concurrent: int = 10,
        timeout: float = 60.0,
    ):
        """
        Initialize the async client.

        Args:
            api_key: API key. Falls back to OPENAI_API_KEY env var.
            base_url: API base URL (e.g., "https://api.deepseek.com").
            max_concurrent: Maximum concurrent requests (rate limiting).
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self._client = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout = timeout

    @property
    def client(self):
        """Lazy initialization of AsyncOpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self._timeout,
            )
        return self._client

    async def create_completion(
        self,
        messages: List[Dict],
        model: str,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Create a chat completion with rate limiting.

        Uses semaphore to limit concurrent requests.
        """
        async with self._semaphore:
            return await self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                **kwargs,
            )


def detect_mime_type(data: bytes) -> str:
    """Detect MIME type from magic bytes."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    elif data[:2] == b'\xff\xd8':
        return "image/jpeg"
    elif data[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    else:
        return "image/png"  # Default fallback


def encode_media_for_openai(part: Union[Path, Image.Image, bytes]) -> Dict[str, Any]:
    """
    Encode media content for OpenAI API format.

    OpenAI uses base64-encoded images inline as data URLs, not file URIs.

    Args:
        part: Path, PIL.Image, or bytes

    Returns:
        Dict with 'type': 'image_url' and base64-encoded data URL

    Raises:
        ValueError: If file not found or unsupported type
    """
    if isinstance(part, Path):
        if not part.exists():
            raise ValueError(f"File not found: {part}")
        mime_type, _ = mimetypes.guess_type(str(part))
        mime_type = mime_type or "image/png"
        with open(part, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    elif isinstance(part, Image.Image):
        buffer = io.BytesIO()
        # Convert RGBA to RGB if needed (JPEG doesn't support alpha)
        if part.mode == 'RGBA':
            part = part.convert('RGB')
        part.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime_type = "image/png"

    elif isinstance(part, bytes):
        mime_type = detect_mime_type(part)
        image_data = base64.b64encode(part).decode("utf-8")

    else:
        raise ValueError(f"Unsupported media type: {type(part)}")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{image_data}"
        }
    }


def build_openai_messages(
    prompt_parts: List[Union[str, Path, Image.Image, bytes]],
    schema: Optional[Type[BaseModel]] = None,
) -> List[Dict]:
    """
    Convert gemini-batch prompt format to OpenAI messages format.

    Args:
        prompt_parts: List of parts (text strings, paths, images, bytes)
        schema: Optional Pydantic model for structured output

    Returns:
        List of message dicts in OpenAI format
    """
    content_parts = []

    for part in prompt_parts:
        if isinstance(part, str):
            content_parts.append({"type": "text", "text": part})
        elif isinstance(part, (Path, Image.Image, bytes)):
            content_parts.append(encode_media_for_openai(part))
        else:
            raise ValueError(f"Unsupported part type: {type(part)}")

    messages = [{"role": "user", "content": content_parts}]

    # Add system message for structured output if schema provided
    if schema is not None:
        schema_json = schema.model_json_schema()
        system_msg = {
            "role": "system",
            "content": (
                f"You must respond with valid JSON matching this schema:\n"
                f"{json.dumps(schema_json, indent=2)}\n"
                f"Only output the JSON, no other text or explanation."
            )
        }
        messages.insert(0, system_msg)

    return messages


async def process_single_request(
    client: AsyncOpenAIClient,
    messages: List[Dict],
    model: str,
    schema: Optional[Type[BaseModel]],
    **kwargs
) -> Tuple[Optional[Union[BaseModel, str]], Optional[Dict]]:
    """
    Process a single request with structured output handling.

    Uses JSON mode + Pydantic validation for providers without json_schema support.
    """
    response_format = None
    if schema is not None:
        response_format = {"type": "json_object"}

    response = await client.create_completion(
        messages=messages,
        model=model,
        response_format=response_format,
        **kwargs
    )

    # Extract response text
    text = response.choices[0].message.content

    # Build metadata
    metadata = None
    if hasattr(response, 'usage') and response.usage:
        metadata = {
            'usageMetadata': {
                'promptTokenCount': response.usage.prompt_tokens,
                'candidatesTokenCount': response.usage.completion_tokens,
                'totalTokenCount': response.usage.total_tokens,
            },
            'modelVersion': response.model if hasattr(response, 'model') else model,
        }

    # Return raw text if no schema
    if schema is None:
        return text, metadata

    # Parse structured output
    extracted = extract_json_from_text(text)
    if extracted is None:
        extracted = text

    try:
        parsed = schema.model_validate_json(extracted)
        return parsed, metadata
    except Exception as e:
        logger.warning(f"Failed to parse response as {schema.__name__}: {e}")
        try:
            # Try parsing as dict first
            parsed = schema.model_validate(json.loads(extracted))
            return parsed, metadata
        except Exception:
            logger.error(f"All parsing attempts failed for schema {schema.__name__}")
            return None, metadata


async def process_single_request_with_retry(
    client: AsyncOpenAIClient,
    messages: List[Dict],
    model: str,
    schema: Optional[Type[BaseModel]],
    retry_count: int = 3,
    retry_delay: float = 1.0,
    **kwargs
) -> Tuple[Optional[Any], Optional[Dict]]:
    """Process with exponential backoff retry."""
    last_error = None

    for attempt in range(retry_count + 1):
        try:
            return await process_single_request(
                client, messages, model, schema, **kwargs
            )
        except Exception as e:
            last_error = e
            if attempt < retry_count:
                delay = retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

    logger.error(f"All {retry_count + 1} attempts failed: {last_error}")
    return None, None


async def async_process(
    prompts: List[List[Union[str, Path, Image.Image, bytes]]],
    schema: Optional[Type[BaseModel]] = None,
    model: str = config.ASYNC_CONFIG["default_model"],
    max_concurrent: int = config.ASYNC_CONFIG["default_max_concurrent"],
    return_metadata: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    timeout: float = config.ASYNC_CONFIG["default_timeout"],
    retry_count: int = config.ASYNC_CONFIG["default_retry_count"],
    retry_delay: float = config.ASYNC_CONFIG["default_retry_delay"],
) -> Union[List[BaseModel], List[str], Tuple[List, List]]:
    """
    Process prompts through OpenAI-compatible API with async concurrency.

    Same interface as batch_process() but for providers without batch APIs.
    Uses asyncio.Semaphore for rate limiting.

    Args:
        prompts: List of prompts (same format as batch_process). Each prompt is a
            list of parts that can be strings (text), Path (images), PIL.Image, or bytes.
        schema: Pydantic model for structured output. Uses JSON mode + parsing.
        model: Model name (provider-specific, e.g., "deepseek-chat")
        max_concurrent: Maximum concurrent requests (rate limiting)
        return_metadata: If True, returns tuple of (results, metadata_list)
        base_url: API base URL (e.g., "https://api.deepseek.com")
        api_key: API key (falls back to OPENAI_API_KEY env var)
        temperature: Sampling temperature
        max_tokens: Max output tokens
        top_p: Top-p sampling
        timeout: Request timeout in seconds
        retry_count: Number of retries on failure
        retry_delay: Initial delay between retries (exponential backoff)

    Returns:
        Same as batch_process():
        - List[BaseModel] if schema provided
        - List[str] if schema is None
        - Tuple of (results, metadata) if return_metadata=True

    Examples:
        >>> from pydantic import BaseModel
        >>> class Answer(BaseModel):
        ...     result: str
        >>>
        >>> # Async usage
        >>> results = await async_process(
        ...     prompts=[["What is 2+2?"], ["What is 3+3?"]],
        ...     schema=Answer,
        ...     model="deepseek-chat",
        ...     base_url="https://api.deepseek.com",
        ...     max_concurrent=10,
        ... )
        >>>
        >>> # With metadata
        >>> results, metadata = await async_process(
        ...     prompts=[["Question"]],
        ...     schema=Answer,
        ...     return_metadata=True,
        ... )
    """
    # Validate inputs
    if not prompts:
        if return_metadata:
            return [], []
        return []

    # Initialize client
    client = AsyncOpenAIClient(
        api_key=api_key,
        base_url=base_url,
        max_concurrent=max_concurrent,
        timeout=timeout,
    )

    # Build generation kwargs
    gen_kwargs = {}
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if max_tokens is not None:
        gen_kwargs["max_tokens"] = max_tokens
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    # Build all request tasks
    tasks = []
    for i, prompt_parts in enumerate(prompts):
        messages = build_openai_messages(prompt_parts, schema)

        task = process_single_request_with_retry(
            client=client,
            messages=messages,
            model=model,
            schema=schema,
            retry_count=retry_count,
            retry_delay=retry_delay,
            **gen_kwargs,
        )
        tasks.append(task)

    # Execute all tasks concurrently
    results_with_metadata = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    results = []
    metadata_list = []

    for item in results_with_metadata:
        if isinstance(item, Exception):
            logger.error(f"Task failed with exception: {item}")
            results.append(None)
            metadata_list.append(None)
        else:
            result, metadata = item
            results.append(result)
            metadata_list.append(metadata)

    if return_metadata:
        return results, metadata_list
    return results


def process(
    prompts: List[List[Union[str, Path, Image.Image, bytes]]],
    schema: Optional[Type[BaseModel]] = None,
    **kwargs
) -> Union[List[BaseModel], List[str], Tuple[List, List]]:
    """
    Synchronous wrapper for async_process().

    Uses asyncio.run() to execute the async function.
    Useful for scripts that don't need async context.

    See async_process() for full documentation.

    Examples:
        >>> from gemini_batch import process
        >>> results = process(
        ...     prompts=[["What is 2+2?"]],
        ...     model="deepseek-chat",
        ...     base_url="https://api.deepseek.com",
        ... )
    """
    return asyncio.run(async_process(prompts, schema, **kwargs))
