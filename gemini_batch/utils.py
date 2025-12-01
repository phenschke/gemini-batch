"""
Core utilities for gemini-batch: client management, image processing, and utilities.
"""

import os
import io
import re
import mimetypes
from typing import Union, Optional, List, Type, Dict, Any
from pathlib import Path
from pydantic import BaseModel
from PIL import Image
from google import genai
from google.genai import types
from google.genai import _transformers
import pymupdf as fitz
import logging

from . import config
from .types import TokenStatistics

# Logging configuration
def setup_logging(level: int = logging.INFO, format_string: str = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s", filename: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    handlers = []
    if filename:
        handlers.append(logging.FileHandler(filename))
    handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )
    return logging.getLogger("gemini_batch")

logger = setup_logging()

# API key configuration
def get_api_key() -> str:
    """Get Gemini API key from environment."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return api_key


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON from text that may contain explanatory content or markdown formatting.

    This function handles common patterns where LLMs wrap JSON in additional text:
    - Markdown code blocks: ```json {...} ```
    - Explanatory prefixes: "Here's the result: {...}"
    - Trailing text after JSON

    Args:
        text: Raw text that may contain JSON

    Returns:
        Cleaned JSON string, or None if no valid JSON structure found

    Examples:
        >>> extract_json_from_text('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> extract_json_from_text('Here is the data: {"key": "value"}')
        '{"key": "value"}'
        >>> extract_json_from_text('{"key": "value"}')
        '{"key": "value"}'
    """
    if not text or not isinstance(text, str):
        return None

    # Strip whitespace
    text = text.strip()

    # Pattern 1: Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    markdown_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    markdown_match = re.search(markdown_pattern, text, re.DOTALL)
    if markdown_match:
        extracted = markdown_match.group(1).strip()
        if extracted:
            logger.debug("Extracted JSON from markdown code block")
            return extracted

    # Pattern 2: Find JSON object {...} or array [...]
    # Look for the first { or [ and find its matching closing bracket
    json_start_obj = text.find('{')
    json_start_arr = text.find('[')

    # Determine which comes first (if any)
    json_start = -1
    is_object = True

    if json_start_obj != -1 and json_start_arr != -1:
        json_start = min(json_start_obj, json_start_arr)
        is_object = (json_start == json_start_obj)
    elif json_start_obj != -1:
        json_start = json_start_obj
        is_object = True
    elif json_start_arr != -1:
        json_start = json_start_arr
        is_object = False

    if json_start == -1:
        # No JSON structure found
        return None

    # Find matching closing bracket
    open_bracket = '{' if is_object else '['
    close_bracket = '}' if is_object else ']'

    depth = 0
    in_string = False
    escape_next = False
    json_end = -1

    for i in range(json_start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == open_bracket:
                depth += 1
            elif char == close_bracket:
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

    if json_end != -1:
        extracted = text[json_start:json_end]
        if json_start > 0:
            logger.debug(f"Extracted JSON from text (prefix removed: '{text[:json_start][:50]}...')")
        return extracted

    # If we couldn't find a complete JSON structure, return None
    return None


class GeminiClient:
    """Wrapper for Gemini client with reusable connection."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key()
        self._client = None
    
    @property
    def client(self) -> genai.Client:
        """Get or create Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client


def upload_file_to_gemini(
    file: Union[Path, Image.Image, bytes],
    client: genai.Client
) -> Dict[str, str]:
    """
    Upload a file to Gemini File API and return URI and MIME type.

    Args:
        file: File to upload - pathlib.Path (file path), PIL.Image.Image, or bytes
        client: Gemini client instance

    Returns:
        Dictionary with 'uri' and 'mime_type' keys

    Raises:
        ValueError: If file type is unsupported or file not found
    """
    # Handle different input types
    if isinstance(file, Path):
        # File path - upload directly
        file_path = str(file)
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        logger.info(f"Uploading file: {file_path}")
        uploaded_file = client.files.upload(file=file_path)
        logger.info(f"Uploaded file: {uploaded_file.name} with MIME type: {uploaded_file.mime_type}")

        return {
            "uri": uploaded_file.uri,
            "mime_type": uploaded_file.mime_type
        }

    elif isinstance(file, Image.Image):
        # PIL Image - save to temp file and upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            file.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            logger.info(f"Uploading PIL Image as PNG")
            uploaded_file = client.files.upload(file=tmp_path)
            logger.info(f"Uploaded image: {uploaded_file.name} with MIME type: {uploaded_file.mime_type}")

            return {
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type
            }
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    elif isinstance(file, bytes):
        # Raw bytes - save to temp file and upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file)
            tmp_path = tmp.name

        try:
            logger.info(f"Uploading bytes as PNG")
            uploaded_file = client.files.upload(file=tmp_path)
            logger.info(f"Uploaded bytes: {uploaded_file.name} with MIME type: {uploaded_file.mime_type}")

            return {
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type
            }
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    else:
        raise ValueError(
            f"Unsupported file type: {type(file)}. "
            f"Supported types: pathlib.Path (file path), PIL.Image.Image, bytes"
        )


def _convert_page_to_image(page: fitz.Page, doc: fitz.Document, dpi: int) -> Image.Image:
    """
    Converts a single PDF page to a PIL Image.
    Tries to extract an embedded image first, otherwise renders the page.
    """
    img = None
    img_list = page.get_images(full=True)
    
    if img_list:
        largest_img_xref = -1
        max_area = 0
        for img_info in img_list:
            width, height = img_info[2], img_info[3]
            area = width * height
            if area > max_area:
                max_area = area
                largest_img_xref = img_info[0]

        if largest_img_xref != -1:
            base_image = doc.extract_image(largest_img_xref)
            if base_image and "image" in base_image:
                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes))
                logger.info(f"Extracting largest embedded image from page {page.number + 1}. Image size: {img.width}x{img.height}")

    if img is None:
        logger.info(f"No embedded image found on page {page.number + 1}. Rendering page with DPI={dpi}.")
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        
    return img


def pdf_pages_to_images(
    pdf_path: str,
    dpi: int = config.IMAGE_PROCESSING_CONFIG["default_dpi"],
    max_pages: Optional[int] = None,
    start_page: int = 1,
    grayscale: bool = False,
    display: bool = False,
    crop_sides: int = 0
) -> List[Image.Image]:
    """
    Convert PDF pages to a list of PIL Images.

    Args:
        pdf_path: Path to PDF (scanned pages).
        dpi: Render resolution for pages that are not image-based.
        max_pages: Optional cap on number of pages.
        start_page: Page index to start from (1-based).
        grayscale: Convert to grayscale to reduce tokens.
        display: Whether to display the images using matplotlib.
        crop_sides: Number of pixels to crop from left and right sides.
    Returns:
        List of PIL Image objects in page order.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    images: List[Image.Image] = []
    start_page_idx = max(0, start_page - 1)
    end_page_idx = doc.page_count if max_pages is None else min(doc.page_count, start_page_idx + max_pages)
    
    logger.info(f"Converting pages {start_page_idx + 1} to {end_page_idx} to images (dpi for rendering fallback={dpi})")
    
    for page_index in range(start_page_idx, end_page_idx):
        logger.info(f"Processing page {page_index + 1} of {doc.page_count}")
        page = doc.load_page(page_index)
        img = _convert_page_to_image(page, doc, dpi)

        if grayscale:
            img = img.convert("L")
        if crop_sides > 0:
            left = crop_sides
            upper = 0
            right = img.width - crop_sides
            lower = img.height
            if right > left:
                img = img.crop((left, upper, right, lower))
        
        images.append(img)
        
        if display:
            _show_image_popup(img, page_index + 1)
            
    logger.info(f"Converted {len(images)} pages to images.")
    doc.close()
    
    return images



def _show_image_popup(img: Image.Image, page_number: int) -> None:
    """
    Show a single image as a popup window and block until closed, at original resolution.
    Args:
        img: PIL Image to display.
        page_number: Page number for the window title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for displaying images. "
            "Install it with: pip install matplotlib"
        )

    dpi = 100  # Use a standard DPI for display
    width, height = img.size
    figsize = (width / dpi, height / dpi)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
    plt.title(f'Page {page_number}')
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)


def build_generation_config(
    response_schema: Optional[Union[types.Schema, Type[BaseModel]]] = None,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    media_resolution: Optional[str] = None,
) -> types.GenerateContentConfig:
    """
    Builds a generation configuration for the Gemini API.

    Args:
        response_schema: The schema for the response (either types.Schema or Pydantic BaseModel class).
        thinking_budget: The thinking budget (int, for Flash models). Cannot be used with thinking_level.
        thinking_level: The thinking level (str: "low" or "high", for Gemini 3.0 Pro). Cannot be used with thinking_budget.
        temperature: The temperature.
        top_p: The top-p value.
        top_k: The top-k value.
        max_output_tokens: The maximum number of output tokens.
        media_resolution: The media resolution for image/video inputs. Valid values:
            - "MEDIA_RESOLUTION_LOW": Lower token usage, faster/cheaper, less detail (70-280 tokens/frame)
            - "MEDIA_RESOLUTION_MEDIUM": Balanced detail, cost, and speed (560 tokens for images)
            - "MEDIA_RESOLUTION_HIGH": Higher token usage, more detail, increased latency/cost (1120 tokens for images)
            Controls the maximum number of tokens allocated for media inputs.

    Returns:
        A types.GenerateContentConfig object representing the generation configuration.

    Raises:
        ValueError: If both thinking_budget and thinking_level are provided.
    """
    gen_config_dict = {
        "temperature": temperature if temperature is not None else config.MODEL_CONFIG["generation_config"]["temperature"],
        "max_output_tokens": max_output_tokens if max_output_tokens is not None else config.MODEL_CONFIG["generation_config"]["max_output_tokens"],
    }

    if response_schema is not None:
        gen_config_dict["response_mime_type"] = "application/json"
        gen_config_dict["response_json_schema"] = _transformers.t_schema(client=None, origin=response_schema)

    # Validate that only one thinking parameter is provided
    if thinking_budget is not None and thinking_level is not None:
        raise ValueError("Cannot specify both thinking_budget and thinking_level. Use thinking_budget for Flash models or thinking_level for Gemini 3.0 Pro.")

    if thinking_budget is not None:
        include_thoughts = False if thinking_budget == 0 else None
        gen_config_dict["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget, include_thoughts=include_thoughts)
    elif thinking_level is not None:
        gen_config_dict["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

    if top_p is not None:
        gen_config_dict["top_p"] = top_p
    if top_k is not None:
        gen_config_dict["top_k"] = top_k

    if media_resolution is not None:
        gen_config_dict["media_resolution"] = media_resolution

    return types.GenerateContentConfig(**gen_config_dict)


def calculate_token_statistics(
    metadata_list: List[Union[Dict[str, Any], None]]
) -> TokenStatistics:
    """
    Calculate token usage statistics from batch processing metadata.

    This function aggregates token counts from the metadata returned by
    batch_process(return_metadata=True), providing both total and average
    statistics across all requests.

    Args:
        metadata_list: List of metadata dictionaries from batch_process(), where
            each entry is either a dict with 'usageMetadata' or None (for failed requests).
            Structure: {'usageMetadata': {'totalTokenCount': int, ...}, 'modelVersion': str}

    Returns:
        TokenStatistics object containing aggregated token usage statistics including:
        - Request counts (total, successful, failed)
        - Total token counts across all successful requests
        - Average tokens per successful request

    Examples:
        >>> from gemini_batch import batch_process, calculate_token_statistics
        >>> results, metadata = batch_process(prompts, schema, return_metadata=True)
        >>> stats = calculate_token_statistics(metadata)
        >>> print(f"Total tokens: {stats.total_tokens}")
        >>> print(f"Average per request: {stats.avg_total_tokens}")
        >>> print(f"Success rate: {stats.successful_requests}/{stats.total_requests}")
    """
    # Initialize counters
    total_requests = len(metadata_list)
    successful_requests = 0
    failed_requests = 0

    # Initialize token totals
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_tokens = 0
    total_cached_tokens = 0
    total_thoughts_tokens = 0

    # Single-pass iteration through metadata
    for metadata in metadata_list:
        # Check if entry is None or missing usageMetadata
        if metadata is None:
            failed_requests += 1
            continue

        if 'usageMetadata' not in metadata:
            failed_requests += 1
            logger.warning("Metadata entry missing 'usageMetadata' field, treating as failed request")
            continue

        # Entry is successful, extract usage data
        successful_requests += 1
        usage = metadata['usageMetadata']

        # Accumulate token counts (handle None values by treating as 0)
        total_prompt_tokens += (usage.get('promptTokenCount') or 0)
        total_candidates_tokens += (usage.get('candidatesTokenCount') or 0)
        total_tokens += (usage.get('totalTokenCount') or 0)
        total_cached_tokens += (usage.get('cachedContentTokenCount') or 0)
        total_thoughts_tokens += (usage.get('thoughtsTokenCount') or 0)

    # Calculate averages (None if no successful requests)
    if successful_requests > 0:
        avg_prompt_tokens = total_prompt_tokens / successful_requests
        avg_candidates_tokens = total_candidates_tokens / successful_requests
        avg_total_tokens = total_tokens / successful_requests
        avg_cached_tokens = total_cached_tokens / successful_requests
        avg_thoughts_tokens = total_thoughts_tokens / successful_requests
    else:
        avg_prompt_tokens = None
        avg_candidates_tokens = None
        avg_total_tokens = None
        avg_cached_tokens = None
        avg_thoughts_tokens = None

    return TokenStatistics(
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        total_prompt_tokens=total_prompt_tokens,
        total_candidates_tokens=total_candidates_tokens,
        total_tokens=total_tokens,
        total_cached_tokens=total_cached_tokens,
        total_thoughts_tokens=total_thoughts_tokens,
        avg_prompt_tokens=avg_prompt_tokens,
        avg_candidates_tokens=avg_candidates_tokens,
        avg_total_tokens=avg_total_tokens,
        avg_cached_tokens=avg_cached_tokens,
        avg_thoughts_tokens=avg_thoughts_tokens,
    )
