"""
Direct Gemini API processing for gemini-batch library.

Provides async_process() and process() functions that use the Gemini API directly
(client.aio.models.generate_content) instead of the batch API. Same interface as
batch_process() but for immediate results at full cost.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_origin, get_args
from pathlib import Path

from PIL import Image
from pydantic import BaseModel
from google.genai import types as genai_types

from . import config
from .utils import (
    GeminiClient,
    build_generation_config,
    collect_and_upload_files,
    extract_json_from_text,
    is_list_schema,
    create_list_wrapper,
    logger,
    _compute_content_hash,
)
from .batch import _extract_text_from_parts


def _build_content_parts(
    prompt_parts: List[Any],
    path_to_uri: Dict[str, Dict[str, str]],
    content_hashes: Dict[str, Dict[str, str]],
    part_media_resolution: Optional[str] = None,
) -> List[genai_types.Part]:
    """
    Build genai Part objects from prompt parts using uploaded file URIs.

    Args:
        prompt_parts: List of parts (str, Path, Image, bytes, Part)
        path_to_uri: Mapping of resolved path strings to URI info
        content_hashes: Mapping of content hashes to URI info
        part_media_resolution: Optional per-part media resolution

    Returns:
        List of genai_types.Part objects
    """
    parts = []
    for part in prompt_parts:
        if isinstance(part, str):
            parts.append(genai_types.Part(text=part))
        elif isinstance(part, genai_types.Part):
            # Pre-built Part object - pass through with validation
            if part.inline_data is not None:
                raise ValueError(
                    "Part objects with inline_data are not supported. "
                    "Use bytes type directly for content that needs uploading."
                )
            if part.file_data is not None and not part.file_data.file_uri:
                raise ValueError(
                    "Part objects with file_data must have file_uri set."
                )
            # Apply global part_media_resolution only if Part doesn't have its own
            if (
                part_media_resolution is not None
                and part.media_resolution is None
                and part.file_data is not None
            ):
                part = genai_types.Part(
                    file_data=part.file_data,
                    media_resolution=genai_types.PartMediaResolution(
                        level=part_media_resolution
                    ),
                )
            parts.append(part)
        elif isinstance(part, Path):
            path_key = str(part.resolve())
            file_info = path_to_uri[path_key]
            file_part = genai_types.Part(
                file_data=genai_types.FileData(
                    file_uri=file_info["uri"],
                    mime_type=file_info["mime_type"],
                ),
            )
            if part_media_resolution is not None:
                file_part = genai_types.Part(
                    file_data=file_part.file_data,
                    media_resolution=genai_types.PartMediaResolution(
                        level=part_media_resolution
                    ),
                )
            parts.append(file_part)
        elif isinstance(part, (Image.Image, bytes)):
            content_hash = _compute_content_hash(part)
            file_info = content_hashes[content_hash]
            file_part = genai_types.Part(
                file_data=genai_types.FileData(
                    file_uri=file_info["uri"],
                    mime_type=file_info["mime_type"],
                ),
            )
            if part_media_resolution is not None:
                file_part = genai_types.Part(
                    file_data=file_part.file_data,
                    media_resolution=genai_types.PartMediaResolution(
                        level=part_media_resolution
                    ),
                )
            parts.append(file_part)
        else:
            raise ValueError(
                f"Unsupported part type: {type(part)}. "
                f"Supported types: str, pathlib.Path, PIL.Image.Image, bytes, types.Part"
            )
    return parts


def _extract_metadata(response: Any) -> Optional[Dict[str, Any]]:
    """Extract metadata dict from a generate_content response."""
    metadata: Dict[str, Any] = {}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        metadata['usageMetadata'] = {
            'totalTokenCount': getattr(response.usage_metadata, 'total_token_count', None),
            'promptTokenCount': getattr(response.usage_metadata, 'prompt_token_count', None),
            'candidatesTokenCount': getattr(response.usage_metadata, 'candidates_token_count', None),
            'cachedContentTokenCount': getattr(response.usage_metadata, 'cached_content_token_count', None),
            'thoughtsTokenCount': getattr(response.usage_metadata, 'thoughts_token_count', None),
        }
    if hasattr(response, 'model_version') and response.model_version:
        metadata['modelVersion'] = response.model_version
    return metadata if metadata else None


async def async_process(
    prompts: List[List[Union[str, Path, Image.Image, bytes, genai_types.Part]]],
    schema: Optional[Type[BaseModel]] = None,
    model: str = config.MODEL_CONFIG["default_model"],
    return_metadata: bool = False,
    media_resolution: Optional[str] = None,
    part_media_resolution: Optional[str] = None,
    max_upload_workers: int = 10,
    max_concurrent: int = config.DIRECT_CONFIG["default_max_concurrent"],
    rpm: Optional[int] = None,
    retry_count: int = config.DIRECT_CONFIG["default_retry_count"],
    retry_delay: float = config.DIRECT_CONFIG["default_retry_delay"],
    show_progress: bool = True,
    vertexai: Optional[bool] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    **generation_kwargs,
) -> Union[List[BaseModel], List[str], Tuple[List, List]]:
    """
    Process prompts through direct Gemini API with structured or raw text output.

    Same interface as batch_process() but uses direct API calls for immediate results
    (no batch job queuing). Useful for smaller workloads or when you need results quickly.

    Args:
        prompts: List of prompts, where each prompt is a list of parts that can be:
            - str: Text content
            - pathlib.Path: Image file paths (auto-uploaded)
            - Image.Image: PIL images (auto-uploaded)
            - bytes: Raw image data (auto-uploaded)
            - genai_types.Part: Pre-built Gemini Part objects
        schema: Optional Pydantic BaseModel class for structured output. If None, returns raw text.
        model: Gemini model to use (default: gemini-3-flash-preview)
        return_metadata: If True, returns tuple of (results, metadata_list) with usage stats
        media_resolution: Optional media resolution for generation config level
        part_media_resolution: Optional media resolution set on each file/image part
        max_upload_workers: Maximum concurrent file uploads (default: 10)
        max_concurrent: Maximum concurrent API requests (default: 10)
        rpm: Maximum requests per minute. If None, no rate limiting beyond max_concurrent.
        retry_count: Number of retries on failure (default: 3)
        retry_delay: Initial delay between retries in seconds (exponential backoff)
        show_progress: Whether to show progress bars (default: True)
        vertexai: If True, use Vertex AI backend. If None, auto-detect.
        project: GCP project ID (Vertex AI only)
        location: GCP region (Vertex AI only)
        gcs_bucket: GCS bucket name (Vertex AI only)
        **generation_kwargs: Additional generation config (temperature, thinking_budget, etc.)

    Returns:
        If return_metadata=False:
            - If schema provided: List of parsed Pydantic models
            - If schema is None: List of raw text strings
        If return_metadata=True:
            - Tuple of (results, metadata_list)
    """
    if not prompts:
        if return_metadata:
            return [], []
        return []

    # Initialize client
    gemini_client = GeminiClient(
        vertexai=vertexai,
        project=project,
        location=location,
        gcs_bucket=gcs_bucket,
    )

    # Handle List[T] schema wrapping
    _list_item_type = None
    _effective_schema = schema
    if schema is not None and is_list_schema(schema):
        args = get_args(schema)
        _list_item_type = args[0]
        _effective_schema = create_list_wrapper(_list_item_type)

    # Upload files
    path_to_uri, content_hashes = await collect_and_upload_files(
        prompts, gemini_client,
        max_upload_workers=max_upload_workers,
        show_progress=show_progress,
    )

    # Build generation config
    gen_config = build_generation_config(
        response_schema=_effective_schema,
        media_resolution=media_resolution,
        model=model,
        **generation_kwargs,
    )

    # Build content parts for all prompts
    all_content_parts = []
    for prompt_parts in prompts:
        parts = _build_content_parts(
            prompt_parts, path_to_uri, content_hashes, part_media_resolution
        )
        all_content_parts.append(parts)

    # Rate limiter state
    last_request_time: List[float] = [0.0]
    rate_lock = asyncio.Lock()

    async def wait_for_rate_limit() -> None:
        if rpm is None:
            return
        min_interval = 60.0 / rpm
        wait_time = 0.0
        async with rate_lock:
            now = time.time()
            elapsed = now - last_request_time[0]
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
            last_request_time[0] = now + wait_time  # Reserve the slot
        if wait_time > 0:
            await asyncio.sleep(wait_time)

    # Semaphore for concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single(
        idx: int, parts: List[genai_types.Part]
    ) -> Tuple[int, Optional[Any], Optional[Dict]]:
        """Process a single prompt with retry logic."""
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                async with semaphore:
                    await wait_for_rate_limit()
                    response = await gemini_client.client.aio.models.generate_content(
                        model=model,
                        contents=parts,
                        config=gen_config,
                    )

                # Extract text
                text = _extract_text_from_parts(response.candidates[0].content.parts)

                # Extract metadata
                metadata = _extract_metadata(response)

                # Parse if schema provided
                if _effective_schema is None:
                    return (idx, text, metadata)

                extracted = extract_json_from_text(text)
                if extracted is None:
                    extracted = text

                try:
                    parsed = _effective_schema.model_validate_json(extracted)
                    return (idx, parsed, metadata)
                except Exception:
                    try:
                        import json
                        parsed = _effective_schema.model_validate(json.loads(extracted))
                        return (idx, parsed, metadata)
                    except Exception as parse_err:
                        logger.error(f"Failed to parse response for prompt {idx}: {parse_err}")
                        return (idx, None, metadata)

            except Exception as e:
                last_error = e
                if attempt < retry_count:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Prompt {idx} attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)

        logger.error(f"Prompt {idx} failed after {retry_count + 1} attempts: {last_error}")
        return (idx, None, None)

    # Process all prompts concurrently
    tasks = [process_single(i, parts) for i, parts in enumerate(all_content_parts)]

    if show_progress:
        from tqdm.asyncio import tqdm_asyncio
        raw_results = await tqdm_asyncio.gather(
            *tasks, desc="Processing prompts", unit="prompt"
        )
    else:
        raw_results = await asyncio.gather(*tasks)

    # Sort results by index to preserve prompt order
    raw_results = sorted(raw_results, key=lambda x: x[0])

    # Extract results and metadata
    results = []
    metadata_list = []
    for idx, result, metadata in raw_results:
        results.append(result)
        if return_metadata:
            metadata_list.append(metadata)

    # Unwrap List[T] results if applicable
    if _list_item_type is not None:
        if return_metadata:
            results = [r.items if r is not None else None for r in results]
            return results, metadata_list
        else:
            results = [r.items if r is not None else None for r in results]
            return results

    if return_metadata:
        return results, metadata_list
    return results


def process(
    prompts: List[List[Union[str, Path, Image.Image, bytes, genai_types.Part]]],
    schema: Optional[Type[BaseModel]] = None,
    **kwargs,
) -> Union[List[BaseModel], List[str], Tuple[List, List]]:
    """
    Synchronous wrapper for async_process().

    Process prompts through direct Gemini API. Same interface as batch_process()
    but uses direct API calls for immediate results.

    See async_process() for full documentation.

    Examples:
        >>> from gemini_batch import process
        >>> from pydantic import BaseModel
        >>>
        >>> class Answer(BaseModel):
        ...     result: str
        >>>
        >>> results = process([["What is 2+2?"]], Answer)
        >>> print(results)  # [Answer(result='4')]
    """
    return asyncio.run(async_process(prompts, schema, **kwargs))
