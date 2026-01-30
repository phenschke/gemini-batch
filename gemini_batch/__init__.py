"""
gemini-batch: Simple batch processing library for Google Gemini API.

Provides a single-function interface for processing large volumes of requests
with structured output at 50% cost savings.

Supports both:
- Gemini Developer API (default): Uses File API for storage
- Vertex AI: Uses Google Cloud Storage (GCS) for storage

For Vertex AI, install with: pip install gemini-batch[vertexai]
"""

from typing import Union, Optional, List, Type, Dict, Any, get_origin, get_args
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
from google.genai import types as genai_types
import hashlib
import io
import time
import json
import logging

logger = logging.getLogger(__name__)

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
    get_batch_job_output_uri,
    resume_batch_job,
)
from .aggregation import aggregate_records
from .utils import (
    calculate_token_statistics,
    GeminiClient,
    upload_file_for_batch,
    upload_to_gcs,
    download_from_gcs,
    list_gcs_blobs,
    is_list_schema,
    create_list_wrapper,
)
from .types import ListVoteConfig, MajorityVoteResult, TokenStatistics
from .embedding import (
    batch_embed,
    embed,
    async_embed,
    create_embedding_batch_job,
    download_embedding_results,
    parse_embedding_results,
)

# Optional async processing (requires openai package)
try:
    from .async_batch import async_process, process
except ImportError:
    async_process = None
    process = None

__version__ = "0.9.5"
__all__ = [
    "batch_process",
    # Batch embeddings
    "batch_embed",
    # Direct API embeddings
    "embed",
    "async_embed",
    # Low-level embedding API
    "create_embedding_batch_job",
    "download_embedding_results",
    "parse_embedding_results",
    # Async processing (OpenAI-compatible APIs)
    "async_process",
    "process",
    # Low-level batch API
    "create_batch_job",
    "monitor_batch_job",
    "download_batch_results",
    "get_inline_results",
    "parse_batch_results",
    "get_batch_job_output_uri",
    "resume_batch_job",
    "aggregate_records",
    "ListVoteConfig",
    "MajorityVoteResult",
    "calculate_token_statistics",
    "TokenStatistics",
    # Vertex AI / GCS utilities
    "GeminiClient",
    "upload_file_for_batch",
    "upload_to_gcs",
    "download_from_gcs",
    "list_gcs_blobs",
]


def _compute_content_hash(file: Union[Path, Image.Image, bytes]) -> str:
    """Compute hash for deduplication (first 64KB + size for speed)."""
    CHUNK_SIZE = 65536  # 64KB

    if isinstance(file, Path):
        size = file.stat().st_size
        with open(file, 'rb') as f:
            chunk = f.read(CHUNK_SIZE)
        return hashlib.sha256(chunk + str(size).encode()).hexdigest()

    elif isinstance(file, Image.Image):
        # Use raw pixel bytes - much faster than PNG encoding
        data = file.tobytes()
        mode_size = f"{file.mode}_{file.size}".encode()
        return hashlib.sha256(data[:CHUNK_SIZE] + mode_size).hexdigest()

    elif isinstance(file, bytes):
        return hashlib.sha256(file[:CHUNK_SIZE] + str(len(file)).encode()).hexdigest()

    raise ValueError(f"Unsupported file type: {type(file)}")


def batch_process(
    prompts: List[List[Union[str, Path, Image.Image, bytes, genai_types.Part]]],
    schema: Optional[Type[BaseModel]] = None,
    model: str = config.MODEL_CONFIG["default_model"],
    wait: bool = True,
    keys: Optional[List[str]] = None,
    job_display_name: Optional[str] = None,
    poll_interval: int = config.BATCH_CONFIG["poll_interval"],
    output_dir: Optional[str] = None,
    jsonl_dir: Optional[str] = None,
    return_metadata: bool = False,
    media_resolution: Optional[str] = None,
    part_media_resolution: Optional[str] = None,
    max_upload_workers: int = 10,
    show_progress: bool = True,
    # Resume from failed job
    resume_from: Optional[str] = None,
    # Vertex AI parameters
    vertexai: Optional[bool] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    **generation_kwargs
) -> Union[List[BaseModel], List[str], str, tuple]:
    """
    Process prompts through Gemini Batch API with structured or raw text output.

    This is the main entry point for the library. It handles mixed text and image
    content, automatically uploads files, and manages batch job creation,
    monitoring, and result parsing.

    Supports both backends:
    - Gemini Developer API (default): Uses File API for storage
    - Vertex AI: Uses Google Cloud Storage (GCS) for storage

    Args:
        prompts: List of prompts, where each prompt is a list of parts that can be:
            - str: Text content
            - pathlib.Path: Image file paths (auto-uploaded)
            - Image.Image: PIL images (auto-uploaded)
            - bytes: Raw image data (auto-uploaded)
            - genai_types.Part: Pre-built Gemini Part objects (with file_data or text).
              Use this for per-part media_resolution control. Part objects with
              inline_data are not supported; use bytes type instead.
            Each inner list represents one batch request with mixed text/image content.
        schema: Optional Pydantic BaseModel class for structured output. If None, returns raw text.
        model: Gemini model to use (default: gemini-2.5-flash)
        wait: If True, wait for completion and return results. If False, return job name.
        keys: Optional list of unique string identifiers for each prompt. When provided:
            - Must have same length as prompts
            - Each key must be unique
            - Request key format: {i}_{key} (e.g., "0_doc_001", "1_doc_002")
            - Useful for mapping results back to source data when recovering batch jobs
            If None (default), request key format is just the index {i} (e.g., "0", "1").
        job_display_name: Optional display name for the batch job
        poll_interval: Seconds between job status checks (if wait=True)
        output_dir: Directory to save results (defaults to .gemini_batch/)
        jsonl_dir: Directory to save JSONL request files (defaults to .gemini_batch/)
        return_metadata: If True, returns tuple of (results, metadata_list) with usage stats
        media_resolution: Optional media resolution for image/video inputs (generation config level). Valid values:
            - "MEDIA_RESOLUTION_LOW": Lower token usage, faster/cheaper, less detail
            - "MEDIA_RESOLUTION_MEDIUM": Balanced detail, cost, and speed
            - "MEDIA_RESOLUTION_HIGH": Higher token usage, more detail, increased latency/cost
            Controls quality vs cost tradeoff for media processing.
        part_media_resolution: Optional media resolution set on each file/image part (experimental,
            requires v1alpha API). Valid values:
            - "MEDIA_RESOLUTION_LOW": Lower token usage, faster/cheaper, less detail
            - "MEDIA_RESOLUTION_MEDIUM": Balanced detail, cost, and speed
            - "MEDIA_RESOLUTION_HIGH": Higher token usage, more detail, increased latency/cost
            - "MEDIA_RESOLUTION_ULTRA_HIGH": Maximum detail for high-resolution media
            This sets resolution per-part rather than globally in generation config.
        max_upload_workers: Maximum number of concurrent file uploads (default: 10).
            Files are uploaded in parallel to improve performance with many images.
        show_progress: Whether to show a progress bar during file uploads (default: True).
        resume_from: Job name or GCS URI to resume from. When provided, creates a new job
            using the previous job's output as input - the system skips already-completed
            requests and only processes incomplete/failed ones. Requires vertexai=True.
            Example: "projects/.../batchJobs/123" or "gs://bucket/path/predictions.jsonl"
        vertexai: If True, use Vertex AI backend with GCS. If None, auto-detect from
                  GOOGLE_GENAI_USE_VERTEXAI env var. (Default: Gemini Developer API)
        project: GCP project ID (Vertex AI only). Falls back to GOOGLE_CLOUD_PROJECT env var.
        location: GCP region (Vertex AI only). Falls back to GOOGLE_CLOUD_LOCATION env var,
                  then defaults to "us-central1".
        gcs_bucket: GCS bucket name for file storage (Vertex AI only). Auto-created if needed.
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
        >>> # Raw text mode (no schema) - Gemini Developer API (default)
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
        >>> # Using Vertex AI backend
        >>> results = batch_process(
        ...     prompts,
        ...     Recipe,
        ...     vertexai=True,
        ...     project="my-gcp-project",
        ...     gcs_bucket="my-batch-bucket"
        ... )
        >>>
        >>> # Or via environment variables:
        >>> # export GOOGLE_GENAI_USE_VERTEXAI=true
        >>> # export GOOGLE_CLOUD_PROJECT=my-gcp-project
        >>> results = batch_process(prompts, Recipe)
        >>>
        >>> # Mixed text and images
        >>> prompts = [
        ...     ["Extract recipe from this image:", "scan1.jpg"],
        ...     ["What recipe is shown here?", Path("scan2.jpg"), " Focus on ingredients."]
        ... ]
        >>> results = batch_process(prompts, Recipe)
        >>>
        >>> # Async mode
        >>> job_name = batch_process(prompts, Recipe, wait=False)
        >>> # ... check status later ...
        >>> from gemini_batch import monitor_batch_job, parse_batch_results
        >>> state = monitor_batch_job(job_name)
        >>> results = parse_batch_results("results.jsonl", Recipe)
        >>>
        >>> # Resume a failed Vertex AI job
        >>> results = batch_process(
        ...     prompts=[],  # Empty when resuming
        ...     schema=Recipe,
        ...     resume_from="projects/.../batchJobs/123",  # or GCS URI
        ...     vertexai=True,
        ... )
    """
    import asyncio

    # Initialize Gemini client (auto-detects backend from params or env vars)
    gemini_client = utils.GeminiClient(
        vertexai=vertexai,
        project=project,
        location=location,
        gcs_bucket=gcs_bucket,
    )

    # Validate keys parameter
    if keys is not None:
        if len(keys) != len(prompts):
            raise ValueError(
                f"Length of 'keys' ({len(keys)}) must match length of 'prompts' ({len(prompts)})"
            )
        if len(keys) != len(set(keys)):
            raise ValueError("Custom keys must be unique")

    # Validate keys not used with resume_from
    if resume_from is not None and keys is not None:
        raise ValueError("'keys' parameter cannot be used with 'resume_from'")

    # Handle resume mode - skip normal prompt processing and use resume_batch_job
    if resume_from is not None:
        if not gemini_client.vertexai:
            raise ValueError("resume_from is only supported with Vertex AI backend (set vertexai=True)")

        # Handle List[T] schema types by wrapping in a model
        _list_item_type = None
        _effective_schema = schema
        if schema is not None and is_list_schema(schema):
            args = get_args(schema)
            _list_item_type = args[0]
            _effective_schema = create_list_wrapper(_list_item_type)

        # Build generation config for the resumed job
        gen_config = utils.build_generation_config(
            response_schema=_effective_schema,
            media_resolution=media_resolution,
            model=model,
            **generation_kwargs
        )

        # Use resume_batch_job to create a new job from the previous output
        job_name = batch.resume_batch_job(
            job_name_or_gcs_uri=resume_from,
            model_name=model,
            job_display_name=job_display_name,
            generation_config=gen_config,
            client=gemini_client,
        )

        if not wait:
            return job_name

        # Wait for completion
        final_state = batch.monitor_batch_job(job_name, poll_interval, client=gemini_client)

        if final_state != 'JOB_STATE_SUCCEEDED':
            raise RuntimeError(f"Resumed batch job failed with state: {final_state}")

        # Get results
        results_file = batch.download_batch_results(job_name, output_dir, client=gemini_client)
        parsed = batch.parse_batch_results(results_file, _effective_schema, return_metadata=return_metadata)

        # Unwrap List[T] results if we wrapped the schema
        if _list_item_type is not None:
            if return_metadata:
                results, metadata = parsed
                results = [r.items if r is not None else None for r in results]
                parsed = (results, metadata)
            else:
                parsed = [r.items if r is not None else None for r in parsed]

        return parsed

    # Phase 1: Collect files and compute content hashes for deduplication
    files_to_upload = []       # Only unique files: [(prompt_idx, part_idx, file)]
    position_to_hash = {}      # All positions: {(i, j): content_hash}
    hash_to_position = {}      # First occurrence: {content_hash: (prompt_idx, part_idx)}

    logger.info("Hashing images for deduplication...")

    for i, prompt_parts in enumerate(prompts):
        for j, part in enumerate(prompt_parts):
            if isinstance(part, (Path, Image.Image, bytes)):
                content_hash = _compute_content_hash(part)
                position_to_hash[(i, j)] = content_hash

                if content_hash not in hash_to_position:
                    # First occurrence - mark for upload
                    hash_to_position[content_hash] = (i, j)
                    files_to_upload.append((i, j, part))

    # Phase 2: Upload new files in chunks to prevent memory spikes
    # For many images (e.g. 3000), holding all upload tasks in memory causes OOM
    # even if concurrency is limited. We must chunk the task creation itself.
    uploaded_files: Dict[tuple, Dict[str, str]] = {}
    
    CHUNK_SIZE = 100
    if files_to_upload:
        total_files = len(files_to_upload)
        total_chunks = (total_files + CHUNK_SIZE - 1) // CHUNK_SIZE
        logger.info(f"Uploading {total_files} files in {total_chunks} chunks...")
        for i in range(0, total_files, CHUNK_SIZE):
            chunk_num = (i // CHUNK_SIZE) + 1
            logger.info(f"Uploading chunk {chunk_num}/{total_chunks}...")
            chunk = files_to_upload[i : i + CHUNK_SIZE]
            
            # Upload this chunk
            chunk_results = asyncio.run(
                utils.upload_files_parallel(
                    chunk,
                    gemini_client,
                    max_concurrent=max_upload_workers,
                    show_progress=show_progress,
                )
            )
            
            # Accumulate results
            uploaded_files.update(chunk_results)
            
            # Explicitly clear chunk memory
            del chunk
            del chunk_results

        # Clear original list
        del files_to_upload

    # Phase 3: Build requests from prompts using uploaded file URIs
    # Build hash -> URI mapping for deduplication lookup
    hash_to_uri = {
        position_to_hash[pos]: uri_info
        for pos, uri_info in uploaded_files.items()
    }

    requests = []
    for i, prompt_parts in enumerate(prompts):
        if keys is not None:
            request_key = f"{i}_{keys[i]}"
        else:
            request_key = str(i)

        # Process each part in the prompt
        content_parts = []
        for j, part in enumerate(prompt_parts):
            if isinstance(part, str):
                # Text content
                content_parts.append({"text": part})
            elif isinstance(part, genai_types.Part):
                # Pre-built Part object - validate and convert to dict
                if part.inline_data is not None:
                    raise ValueError(
                        "Part objects with inline_data are not supported. "
                        "Use bytes type directly for content that needs uploading."
                    )
                if part.file_data is not None and not part.file_data.file_uri:
                    raise ValueError(
                        "Part objects with file_data must have file_uri set."
                    )
                # Convert Part to dict, preserving all fields including media_resolution
                part_dict = part.model_dump(exclude_none=True)
                # Apply global part_media_resolution only if Part doesn't have its own
                if part_media_resolution is not None and "media_resolution" not in part_dict:
                    if part.file_data is not None:
                        part_dict["media_resolution"] = {"level": part_media_resolution}
                content_parts.append(part_dict)
            elif isinstance(part, (Path, Image.Image, bytes)):
                # File content - lookup via content hash (enables deduplication)
                content_hash = position_to_hash[(i, j)]
                file_info = hash_to_uri[content_hash]
                file_part = {
                    "file_data": {
                        "file_uri": file_info["uri"],
                        "mime_type": file_info["mime_type"]
                    }
                }
                # Add part-level media resolution if specified (experimental, v1alpha API)
                if part_media_resolution is not None:
                    file_part["media_resolution"] = {"level": part_media_resolution}
                content_parts.append(file_part)
            else:
                raise ValueError(
                    f"Unsupported part type: {type(part)}. "
                    f"Supported types: str, pathlib.Path, PIL.Image.Image, bytes, types.Part"
                )

        # Build request
        request = {
            "key": request_key,
            "request": {
                "contents": [{"role": "user", "parts": content_parts}]
            }
        }
        requests.append(request)

    # Handle List[T] schema types by wrapping in a model
    # (Google GenAI SDK doesn't support List[T] as top-level schema)
    _list_item_type = None
    _effective_schema = schema
    if schema is not None and is_list_schema(schema):
        args = get_args(schema)
        _list_item_type = args[0]
        _effective_schema = create_list_wrapper(_list_item_type)

    # Build generation config
    gen_config = utils.build_generation_config(
        response_schema=_effective_schema,
        media_resolution=media_resolution,
        model=model,
        **generation_kwargs
    )

    # Create batch job
    job_name = create_batch_job(
        requests=requests,
        model_name=model,
        job_display_name=job_display_name,
        generation_config=gen_config,
        jsonl_dir=jsonl_dir,
        client=gemini_client,
    )

    if not wait:
        return job_name

    # Wait for completion
    final_state = monitor_batch_job(job_name, poll_interval, client=gemini_client)

    if final_state != 'JOB_STATE_SUCCEEDED':
        raise RuntimeError(f"Batch job failed with state: {final_state}")

    # Get results (always file-based)
    results_file = download_batch_results(job_name, output_dir, client=gemini_client)
    parsed = parse_batch_results(results_file, _effective_schema, return_metadata=return_metadata)

    # Unwrap List[T] results if we wrapped the schema
    if _list_item_type is not None:
        if return_metadata:
            results, metadata = parsed
            results = [r.items if r is not None else None for r in results]
            parsed = (results, metadata)
        else:
            parsed = [r.items if r is not None else None for r in parsed]

    return parsed
