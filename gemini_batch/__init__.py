"""
gemini-batch: Simple batch processing library for Google Gemini API.

Provides a single-function interface for processing large volumes of requests
with structured output at 50% cost savings.
"""

from typing import Union, Optional, List, Type, Dict, Any
from pathlib import Path
from PIL import Image
from pydantic import BaseModel

from . import config
from . import batch
from . import utils
from . import aggregation
from .batch import (
    create_batch_job,
    monitor_batch_job,
    download_batch_results,
    get_inline_results,
    parse_batch_results,
)
from .aggregation import aggregate_records, ListVoteConfig, MajorityVoteResult

__version__ = "0.4.0"
__all__ = [
    "batch_process",
    "create_batch_job",
    "monitor_batch_job",
    "download_batch_results",
    "get_inline_results",
    "parse_batch_results",
    "aggregate_records",
    "ListVoteConfig",
    "MajorityVoteResult",
]


def batch_process(
    prompts: List[List[Union[str, Path, Image.Image, bytes]]],
    schema: Optional[Type[BaseModel]] = None,
    model: str = config.MODEL_CONFIG["default_model"],
    wait: bool = True,
    n_samples: int = 1,
    job_display_name: Optional[str] = None,
    poll_interval: int = config.BATCH_CONFIG["poll_interval"],
    output_dir: Optional[str] = None,
    jsonl_dir: Optional[str] = None,
    return_metadata: bool = False,
    media_resolution: Optional[str] = None,
    **generation_kwargs
) -> Union[List[BaseModel], List[str], str, tuple]:
    """
    Process prompts through Gemini Batch API with structured or raw text output.

    This is the main entry point for the library. It handles mixed text and image
    content, automatically uploads files to Gemini File API, and manages batch
    job creation, monitoring, and result parsing.

    Args:
        prompts: List of prompts, where each prompt is a list of parts that can be:
            - str: Text content
            - pathlib.Path: Image file paths (auto-uploaded to File API)
            - Image.Image: PIL images (auto-uploaded)
            - bytes: Raw image data (auto-uploaded)
            Each inner list represents one batch request with mixed text/image content.
        schema: Optional Pydantic BaseModel class for structured output. If None, returns raw text.
        model: Gemini model to use (default: gemini-2.5-flash)
        wait: If True, wait for completion and return results. If False, return job name.
        n_samples: Number of times to process each prompt (for majority voting)
        job_display_name: Optional display name for the batch job
        poll_interval: Seconds between job status checks (if wait=True)
        output_dir: Directory to save results (defaults to .gemini_batch/)
        jsonl_dir: Directory to save JSONL request files (defaults to .gemini_batch/)
        return_metadata: If True, returns tuple of (results, metadata_list) with usage stats
        media_resolution: Optional media resolution for image/video inputs. Valid values:
            - "MEDIA_RESOLUTION_LOW": Lower token usage, faster/cheaper, less detail
            - "MEDIA_RESOLUTION_MEDIUM": Balanced detail, cost, and speed
            - "MEDIA_RESOLUTION_HIGH": Higher token usage, more detail, increased latency/cost
            Controls quality vs cost tradeoff for media processing.
        **generation_kwargs: Additional generation config (temperature, max_output_tokens, etc.)

    Returns:
        If wait=False: Job name string for later retrieval
        If wait=True and return_metadata=False:
            - If schema provided: List of parsed Pydantic models (one per prompt)
            - If schema is None: List of raw text strings (one per prompt)
        If wait=True and return_metadata=True:
            - Tuple of (results, metadata_list) where metadata_list contains usage metadata
              dicts with token counts (including thoughtsTokenCount), model versions, etc.

    Examples:
        >>> from pydantic import BaseModel
        >>> from pathlib import Path
        >>>
        >>> # Raw text mode (no schema)
        >>> prompts = [
        ...     ["What is 2+2?"],
        ...     ["What is the capital of France?"]
        ... ]
        >>> results = batch_process(prompts, wait=True)  # Returns ["4", "Paris"]
        >>>
        >>> # Structured output mode
        >>> class Recipe(BaseModel):
        ...     name: str
        ...     ingredients: List[str]
        >>>
        >>> # Simple text-only prompts
        >>> prompts = [
        ...     ["Give me a recipe for pancakes"],
        ...     ["Give me a recipe for cookies"]
        ... ]
        >>> results = batch_process(prompts, Recipe, wait=True)
        >>>
        >>> # Mixed text and images
        >>> prompts = [
        ...     ["Extract recipe from this image:", "scan1.jpg"],
        ...     ["What recipe is shown here?", Path("scan2.jpg"), " Focus on ingredients."]
        ... ]
        >>> results = batch_process(prompts, Recipe, n_samples=3)
        >>>
        >>> # Complex multimodal prompts
        >>> from PIL import Image
        >>> img = Image.open("recipe.jpg")
        >>> prompts = [
        ...     ["You are a recipe extraction expert. ", img, " Extract the recipe name and ingredients."]
        ... ]
        >>> results = batch_process(prompts, Recipe)
        >>>
        >>>
        >>> # Async mode
        >>> job_name = batch_process(prompts, Recipe, wait=False)
        >>> # ... check status later ...
        >>> from gemini_batch import monitor_batch_job, parse_batch_results
        >>> state = monitor_batch_job(job_name)
        >>> results = parse_batch_results("results.jsonl", Recipe)
    """
    # Initialize Gemini client for file uploads
    gemini_client = utils.GeminiClient()
    client = gemini_client.client

    # Build requests from prompts
    requests = []
    for i, prompt_parts in enumerate(prompts):
        for sample_idx in range(n_samples):
            request_key = f"prompt_{i}_sample_{sample_idx}"

            # Process each part in the prompt
            content_parts = []
            for part in prompt_parts:
                if isinstance(part, str):
                    # Text content
                    content_parts.append({"text": part})
                elif isinstance(part, (Path, Image.Image, bytes)):
                    # File content - upload to File API
                    file_info = utils.upload_file_to_gemini(part, client)
                    content_parts.append({
                        "file_data": {
                            "file_uri": file_info["uri"],
                            "mime_type": file_info["mime_type"]
                        }
                    })
                else:
                    raise ValueError(
                        f"Unsupported part type: {type(part)}. "
                        f"Supported types: str (text), pathlib.Path (file path), PIL.Image.Image, bytes"
                    )

            # Build request
            request = {
                "key": request_key,
                "request": {
                    "contents": [{"parts": content_parts}]
                }
            }
            requests.append(request)

    # Build generation config
    gen_config = utils.build_generation_config(
        response_schema=schema,
        media_resolution=media_resolution,
        **generation_kwargs
    )

    # Create batch job
    job_name = create_batch_job(
        requests=requests,
        model_name=model,
        job_display_name=job_display_name,
        generation_config=gen_config,
        jsonl_dir=jsonl_dir,
    )

    if not wait:
        return job_name

    # Wait for completion
    final_state = monitor_batch_job(job_name, poll_interval)

    if final_state != 'JOB_STATE_SUCCEEDED':
        raise RuntimeError(f"Batch job failed with state: {final_state}")

    # Get results (always file-based)
    results_file = download_batch_results(job_name, output_dir)
    parsed = parse_batch_results(results_file, schema, return_metadata=return_metadata)

    # Handle n_samples > 1: group results by prompt and aggregate
    if n_samples > 1:
        # If return_metadata=True, parsed is a tuple (results, metadata)
        if return_metadata:
            results, metadata = parsed
        else:
            results = parsed
            metadata = None

        # Group results by prompt index
        grouped_results = []
        grouped_metadata = [] if return_metadata else None

        for i in range(len(prompts)):
            # Get samples for this prompt
            start_idx = i * n_samples
            end_idx = (i + 1) * n_samples
            samples = results[start_idx:end_idx]

            # Filter out None values before aggregation
            valid_samples = [s for s in samples if s is not None]

            if not valid_samples:
                # All samples failed - append None
                grouped_results.append(None)
                if return_metadata and metadata is not None:
                    grouped_metadata.append(None)
            elif len(valid_samples) == 1:
                # Only one valid sample - use it directly (no aggregation needed)
                grouped_results.append(valid_samples[0])
                if return_metadata and metadata is not None:
                    # Get metadata for the one valid sample
                    sample_metadata = [metadata[start_idx + j] for j, s in enumerate(samples) if s is not None]
                    grouped_metadata.append(sample_metadata[0])
            else:
                # Multiple valid samples - aggregate via majority voting
                aggregated = aggregate_records(valid_samples, as_model=schema)
                grouped_results.append(aggregated.aggregated)
                if return_metadata and metadata is not None:
                    # Aggregate metadata (take first valid metadata)
                    sample_metadata = [metadata[start_idx + j] for j, s in enumerate(samples) if s is not None]
                    grouped_metadata.append(sample_metadata[0] if sample_metadata else None)

        if return_metadata:
            return grouped_results, grouped_metadata
        return grouped_results

    return parsed
