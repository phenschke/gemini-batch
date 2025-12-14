"""
Batch embedding functionality for gemini-batch library.

Provides batch embedding generation using the Gemini Batch API for 50% cost savings.
"""

import json
import time
from typing import Union, Optional, List, Dict, Any, Tuple
from pathlib import Path
from google.genai import types

from . import config as app_config
from .utils import (
    GeminiClient,
    logger,
)
from .batch import (
    extract_timestamp_from_display_name,
    monitor_batch_job,
)


def create_embedding_batch_job(
    texts: List[str],
    model_name: str = app_config.EMBEDDING_CONFIG["default_model"],
    task_type: str = app_config.EMBEDDING_CONFIG["default_task_type"],
    job_display_name: Optional[str] = None,
    jsonl_dir: Optional[str] = None,
    client: Optional[GeminiClient] = None,
) -> str:
    """
    Create a batch embedding job for processing texts.

    Note: Only supports Gemini Developer API. Vertex AI is not supported for
    batch embeddings because the embedding batch API requires the File API.

    Args:
        texts: List of text strings to embed
        model_name: Gemini embedding model to use
        task_type: Embedding task type (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
        job_display_name: Display name for the batch job
        jsonl_dir: Directory to save JSONL file
        client: GeminiClient instance. If None, creates a new one.

    Returns:
        Batch job name for monitoring

    Raises:
        ValueError: If Vertex AI backend is detected (not supported for embeddings)
    """
    if client is None:
        client = GeminiClient()

    # Embedding batch API only supports Developer API (not Vertex AI)
    # - files.upload() is not available in Vertex AI
    # - create_embeddings() doesn't accept GCS URIs
    if client.vertexai:
        raise ValueError(
            "Batch embeddings are not supported with Vertex AI. "
            "The embedding batch API requires the File API which is only available "
            "with the Gemini Developer API. Use vertexai=False or unset "
            "GOOGLE_GENAI_USE_VERTEXAI environment variable."
        )

    # Validate task_type
    valid_task_types = app_config.EMBEDDING_CONFIG["valid_task_types"]
    if task_type not in valid_task_types:
        raise ValueError(
            f"Invalid task_type '{task_type}'. Must be one of: {valid_task_types}"
        )

    # Build requests with embedding format
    requests = []
    for i, text in enumerate(texts):
        request = {
            "key": str(i),
            "request": {
                "content": {"parts": [{"text": text}]},
                "task_type": task_type
            }
        }
        requests.append(request)

    # Generate timestamp for consistent naming
    timestamp = int(time.time())

    if job_display_name is None:
        job_display_name = f"embed-{timestamp}"

    logger.info(f"Creating embedding batch job '{job_display_name}' with {len(texts)} texts...")

    # Create JSONL file locally
    if jsonl_dir is None:
        jsonl_dir = app_config.BATCH_CONFIG["default_jsonl_dir"]

    Path(jsonl_dir).mkdir(parents=True, exist_ok=True)
    jsonl_filename = str(Path(jsonl_dir) / f"embed_{timestamp}_requests.jsonl")

    with open(jsonl_filename, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Note: Embedding batch API uses File API for both Developer API and Vertex AI
    # (unlike text generation which supports GCS URIs for Vertex AI)
    uploaded_file = client.client.files.upload(
        file=jsonl_filename,
        config=types.UploadFileConfig(
            display_name=f"embed-requests-{timestamp}",
            mime_type="application/json"
        )
    )

    if not uploaded_file.name:
        raise ValueError("Failed to get name for uploaded embedding requests file.")

    logger.info(f"Uploaded embedding requests file: {uploaded_file.name}")

    # Create batch embedding job with file
    batch_job = client.client.batches.create_embeddings(
        model=model_name,
        src=types.EmbeddingsBatchJobSource(file_name=uploaded_file.name),
    )

    if not batch_job.name:
        raise ValueError("Failed to create embedding batch job or get job name.")

    logger.info(f"Created embedding batch job: {batch_job.name}")
    return batch_job.name


def download_embedding_results(
    batch_job_name: str,
    output_dir: Optional[str] = None,
    client: Optional[GeminiClient] = None,
) -> str:
    """
    Download embedding batch results file from completed job.

    Note: Only supports Gemini Developer API (File API).

    Args:
        batch_job_name: Name of the completed batch job
        output_dir: Directory to save the downloaded file (defaults to .gemini_batch/)
        client: GeminiClient instance. If None, creates a new one.

    Returns:
        Path to the downloaded file
    """
    if client is None:
        client = GeminiClient()

    batch_job = client.client.batches.get(name=batch_job_name)

    if batch_job.state is None or batch_job.state.name is None:
        raise ValueError("Batch job state is missing or invalid.")
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        raise ValueError(f"Job not succeeded. Current state: {batch_job.state.name}")

    # Use default output directory if not specified
    if output_dir is None:
        output_dir = app_config.BATCH_CONFIG["default_results_dir"]

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract timestamp from display_name and create result filename
    if batch_job.display_name:
        timestamp = extract_timestamp_from_display_name(batch_job.display_name)
        if timestamp:
            new_file_name = f"embed_{timestamp}_results.jsonl"
        else:
            new_file_name = batch_job.display_name + ".jsonl"
    else:
        new_file_name = "embed_results.jsonl"

    output_path = Path(output_dir) / new_file_name

    # Download from File API
    if batch_job.dest:
        file_name = getattr(batch_job.dest, 'file_name', None)
        if file_name:
            logger.info(f"Downloading embedding results file {file_name} to {output_path}...")
            file_content = client.client.files.download(file=file_name)
            if not file_content:
                raise ValueError("Failed to download embedding results file or file is empty.")
            with open(output_path, "wb") as f:
                f.write(file_content)
            logger.info(f"Downloaded embedding results file to {output_path}")
            return str(output_path)

    raise ValueError("Batch job does not have an associated results file.")


def parse_embedding_results(
    results: Union[str, List[Dict]],
    return_metadata: bool = False
) -> Union[List[List[float]], Tuple[List[List[float]], List[Dict[str, Any]]]]:
    """
    Parse batch embedding results into embedding vectors.

    Args:
        results: Either path to JSONL file or list of result dicts
        return_metadata: If True, returns tuple of (embeddings, metadata_list)

    Returns:
        If return_metadata=False:
            List of embedding vectors (List[List[float]]) with None for failures
        If return_metadata=True:
            Tuple of (embeddings, metadata_list)

        Note: Output list length always matches input length. Failed results are
        represented as None to preserve alignment with input requests.
    """
    # Load results if file path provided
    if isinstance(results, str):
        json_lines = []
        with open(results, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f.readlines(), start=1):
                line = line.strip()
                if not line:
                    json_lines.append(None)
                    continue
                try:
                    json_lines.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Malformed JSON on line {line_num} in {results}: {e}")
                    json_lines.append(None)
    else:
        json_lines = results

    if not json_lines:
        logger.warning("No embedding results content found to process.")
        if return_metadata:
            return [], []
        return []

    # Sort results by key to ensure correct order (batch API doesn't guarantee order)
    def extract_sort_key(line):
        if line is None:
            return float('inf')
        key = line.get('key', '')
        try:
            return int(key)
        except (ValueError, TypeError):
            return float('inf')

    json_lines = sorted(json_lines, key=extract_sort_key)

    embeddings: List[Optional[List[float]]] = []
    metadata_list: List[Optional[Dict[str, Any]]] = []
    success_count = 0
    failure_count = 0

    for line in json_lines:
        if line is None:
            embeddings.append(None)
            if return_metadata:
                metadata_list.append(None)
            failure_count += 1
            continue

        identifier = line.get('key', 'unknown')

        if 'error' in line:
            logger.error(f"Embedding request {identifier} failed with error: {line['error']}")
            embeddings.append(None)
            if return_metadata:
                metadata_list.append(None)
            failure_count += 1
            continue

        if 'response' not in line:
            logger.warning(f"Skipping embedding result {identifier} without 'response' field.")
            embeddings.append(None)
            if return_metadata:
                metadata_list.append(None)
            failure_count += 1
            continue

        try:
            response = line['response']

            # Extract embedding values
            if hasattr(response, 'embedding'):
                # genai response object
                embedding_values = list(response.embedding.values)
                if return_metadata:
                    metadata = {}
                    if hasattr(response, 'usage_metadata'):
                        metadata['usageMetadata'] = {
                            'totalTokenCount': getattr(response.usage_metadata, 'total_token_count', None),
                            'promptTokenCount': getattr(response.usage_metadata, 'prompt_token_count', None),
                        }
                    metadata_list.append(metadata)
            elif isinstance(response, dict):
                # dict format
                embedding_values = response['embedding']['values']
                if return_metadata:
                    metadata = {}
                    if 'usageMetadata' in response:
                        metadata['usageMetadata'] = response['usageMetadata']
                    metadata_list.append(metadata)
            else:
                logger.error(f"Unknown embedding response format for {identifier}")
                embeddings.append(None)
                if return_metadata:
                    metadata_list.append(None)
                failure_count += 1
                continue

            embeddings.append(embedding_values)
            success_count += 1
            logger.debug(f"Successfully parsed embedding for {identifier}")

        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Failed to parse embedding result for {identifier}: {e}")
            embeddings.append(None)
            if return_metadata:
                metadata_list.append(None)
            failure_count += 1

    logger.info(f"Parsed {success_count} successful embeddings, {failure_count} failures")

    if return_metadata:
        return embeddings, metadata_list
    return embeddings


def batch_embed(
    texts: List[str],
    model: str = app_config.EMBEDDING_CONFIG["default_model"],
    task_type: str = app_config.EMBEDDING_CONFIG["default_task_type"],
    wait: bool = True,
    job_display_name: Optional[str] = None,
    poll_interval: int = app_config.BATCH_CONFIG["poll_interval"],
    output_dir: Optional[str] = None,
    jsonl_dir: Optional[str] = None,
    return_metadata: bool = False,
    # Vertex AI parameters
    vertexai: Optional[bool] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
) -> Union[List[List[float]], str, Tuple[List[List[float]], List[Dict]]]:
    """
    Generate embeddings for texts through Gemini Batch API at 50% cost.

    This is the main entry point for batch embeddings. It handles job creation,
    monitoring, result download, and parsing.

    Supports both backends:
    - Gemini Developer API (default): Uses File API for storage
    - Vertex AI: Uses Google Cloud Storage (GCS) for storage

    Args:
        texts: List of text strings to embed
        model: Gemini embedding model to use (default: gemini-embedding-001)
        task_type: Embedding task type. Valid values:
            - "RETRIEVAL_DOCUMENT": For documents to be retrieved
            - "RETRIEVAL_QUERY": For search queries
            - "SEMANTIC_SIMILARITY": For similarity comparisons
            - "CLASSIFICATION": For text classification
            - "CLUSTERING": For clustering tasks
        wait: If True, wait for completion and return results. If False, return job name.
        job_display_name: Optional display name for the batch job
        poll_interval: Seconds between job status checks (if wait=True)
        output_dir: Directory to save results (defaults to .gemini_batch/)
        jsonl_dir: Directory to save JSONL request files (defaults to .gemini_batch/)
        return_metadata: If True, returns tuple of (embeddings, metadata_list)
        vertexai: If True, use Vertex AI backend with GCS. If None, auto-detect from
                  GOOGLE_GENAI_USE_VERTEXAI env var. (Default: Gemini Developer API)
        project: GCP project ID (Vertex AI only). Falls back to GOOGLE_CLOUD_PROJECT env var.
        location: GCP region (Vertex AI only). Falls back to GOOGLE_CLOUD_LOCATION env var.
        gcs_bucket: GCS bucket name for file storage (Vertex AI only).

    Returns:
        If wait=False: Job name string for later retrieval
        If wait=True and return_metadata=False:
            List of embedding vectors (List[List[float]])
        If wait=True and return_metadata=True:
            Tuple of (embeddings, metadata_list)

    Examples:
        >>> # Basic usage
        >>> texts = ["Document about cats", "Document about dogs"]
        >>> embeddings = batch_embed(texts)
        >>> print(len(embeddings[0]))  # Embedding dimension
        768
        >>>
        >>> # With task type for search
        >>> query_embeddings = batch_embed(
        ...     ["search query"],
        ...     task_type="RETRIEVAL_QUERY"
        ... )
        >>>
        >>> # Async mode
        >>> job_name = batch_embed(texts, wait=False)
        >>> # ... check status later ...
    """
    # Initialize Gemini client (auto-detects backend from params or env vars)
    gemini_client = GeminiClient(
        vertexai=vertexai,
        project=project,
        location=location,
        gcs_bucket=gcs_bucket,
    )

    # Create batch embedding job
    job_name = create_embedding_batch_job(
        texts=texts,
        model_name=model,
        task_type=task_type,
        job_display_name=job_display_name,
        jsonl_dir=jsonl_dir,
        client=gemini_client,
    )

    if not wait:
        return job_name

    # Wait for completion
    final_state = monitor_batch_job(job_name, poll_interval, client=gemini_client)

    if final_state != 'JOB_STATE_SUCCEEDED':
        raise RuntimeError(f"Embedding batch job failed with state: {final_state}")

    # Download results
    results_file = download_embedding_results(job_name, output_dir, client=gemini_client)

    # Parse and return embeddings
    return parse_embedding_results(results_file, return_metadata=return_metadata)
