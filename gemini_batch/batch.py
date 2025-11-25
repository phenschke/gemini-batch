"""
Batch processing functionality for gemini-batch library.
"""

import json
import time
from typing import Union, Optional, List, Type, Any, Dict
from pathlib import Path
from google.genai import types
from pydantic import BaseModel, ValidationError

from . import config as app_config
from .utils import GeminiClient, pdf_pages_to_images, build_generation_config, logger, extract_json_from_text


def create_batch_job(
    requests: List[Dict],
    model_name: str = app_config.MODEL_CONFIG["default_model"],
    job_display_name: Optional[str] = None,
    generation_config: Optional[types.GenerateContentConfig] = None,
    jsonl_dir: Optional[str] = None,
) -> str:
    """
    Create a batch job for processing requests using file-based mode.

    Args:
        requests: List of request dictionaries with 'key' and 'request' fields
        model_name: Gemini model to use
        job_display_name: Display name for the batch job
        generation_config: Configuration for generation
        jsonl_dir: Directory to save JSONL file

    Returns:
        Batch job name for monitoring
    """
    client = GeminiClient()

    # Apply generation config to all requests if provided
    if generation_config:
        gen_config_dict = generation_config
        if isinstance(generation_config, types.GenerateContentConfig):
            gen_config_dict = generation_config.model_dump(exclude_unset=True, exclude_none=True, mode='json')

        for req in requests:
            if 'request' in req:
                req['request']['generation_config'] = gen_config_dict

    if job_display_name is None:
        job_display_name = f"batch-{len(requests)}-requests"

    logger.info(f"Creating batch job '{job_display_name}' with {len(requests)} requests...")

    # Create JSONL and upload
    if jsonl_dir is None:
        jsonl_dir = app_config.BATCH_CONFIG["default_jsonl_dir"]

    Path(jsonl_dir).mkdir(parents=True, exist_ok=True)
    jsonl_filename = str(Path(jsonl_dir) / f"batch_requests_{int(time.time())}.jsonl")

    with open(jsonl_filename, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Upload batch requests file
    uploaded_file = client.client.files.upload(
        file=jsonl_filename,
        config=types.UploadFileConfig(
            display_name=f"batch-requests-{int(time.time())}",
            mime_type="application/json"
        )
    )

    if not uploaded_file.name:
        raise ValueError("Failed to get name for uploaded batch requests file.")

    logger.info(f"Uploaded batch requests file: {uploaded_file.name}")

    # Create batch job with file
    batch_job = client.client.batches.create(
        model=model_name,
        src=uploaded_file.name,
        config=types.CreateBatchJobConfig(display_name=job_display_name)
    )

    if not batch_job.name:
        raise ValueError("Failed to create batch job or get job name.")

    logger.info(f"Created batch job: {batch_job.name}")
    return batch_job.name


def monitor_batch_job(job_name: str, poll_interval: int = app_config.BATCH_CONFIG["poll_interval"]) -> str:
    """
    Monitor a batch job until completion.

    Args:
        job_name: Name of the batch job to monitor
        poll_interval: Seconds to wait between status checks

    Returns:
        Final job state
    """
    client = GeminiClient()

    logger.info(f"Monitoring batch job: {job_name}")

    while True:
        batch_job = client.client.batches.get(name=job_name)
        if batch_job.state:
            state = batch_job.state.name
            logger.info(f"Job state: {state}")
            if state in app_config.BATCH_CONFIG["completed_states"]:
                if state == 'JOB_STATE_FAILED':
                    logger.error(f"Job failed with error: {getattr(batch_job, 'error', None)}")
                elif state == 'JOB_STATE_SUCCEEDED':
                    # Log batch stats
                    if hasattr(batch_job, 'batch_stats'):
                        stats = batch_job.batch_stats
                        logger.info(f"Batch stats: {stats}")
                        if hasattr(stats, 'failed_request_count') and stats.failed_request_count > 0:
                            logger.warning(f"{stats.failed_request_count} requests failed")
                return state
        else:
            logger.info("Job state is not available yet, continuing to poll.")
        time.sleep(poll_interval)


def download_batch_results(batch_job_name: str, output_dir: str = ".", overwrite: bool = True) -> str:
    """
    Download batch results file from completed job.

    Args:
        batch_job_name: Name of the completed batch job
        output_dir: Directory to save the downloaded file
        overwrite: If True, overwrites existing file. If False, auto-increments filename.

    Returns:
        Path to the downloaded file
    """
    client = GeminiClient()
    batch_job = client.client.batches.get(name=batch_job_name)

    if batch_job.state is None or batch_job.state.name is None:
        raise ValueError("Batch job state is missing or invalid.")
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        raise ValueError(f"Job not succeeded. Current state: {batch_job.state.name}")
    if not batch_job.dest or not batch_job.dest.file_name:
        raise ValueError("Batch job does not have an associated results file.")

    if batch_job.display_name:
        new_file_name = batch_job.display_name + ".jsonl"
    else:
        new_file_name = batch_job.dest.file_name

    output_path = Path(output_dir) / Path(new_file_name).name

    # Auto-increment filename if it exists and overwrite is False
    if not overwrite and output_path.exists():
        base_name = output_path.stem
        extension = output_path.suffix
        parent_dir = output_path.parent

        counter = 1
        while True:
            new_name = f"{base_name}_{counter}{extension}"
            output_path = parent_dir / new_name
            if not output_path.exists():
                logger.info(f"File exists. Saving to {output_path} instead.")
                break
            counter += 1

    logger.info(f"Downloading batch results file {batch_job.dest.file_name} to {output_path}...")
    file_content = client.client.files.download(file=batch_job.dest.file_name)
    if not file_content:
        raise ValueError("Failed to download batch results file or file is empty.")
    with open(output_path, "wb") as f:
        f.write(file_content)
    logger.info(f"Downloaded batch results file to {output_path}")
    return str(output_path)


def get_inline_results(batch_job_name: str) -> List[Dict]:
    """
    Get inline results from a completed batch job.

    Args:
        batch_job_name: Name of the completed batch job

    Returns:
        List of result dictionaries
    """
    client = GeminiClient()
    batch_job = client.client.batches.get(name=batch_job_name)

    if batch_job.state is None or batch_job.state.name is None:
        raise ValueError("Batch job state is missing or invalid.")
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        raise ValueError(f"Job not succeeded. Current state: {batch_job.state.name}")
    if not batch_job.dest or not batch_job.dest.inlined_responses:
        raise ValueError("Batch job does not have inline responses.")

    results = []
    for i, response_obj in enumerate(batch_job.dest.inlined_responses):
        if response_obj.response:
            results.append({
                'index': i,
                'response': response_obj.response,
            })
        elif response_obj.error:
            results.append({
                'index': i,
                'error': response_obj.error,
            })

    return results


def parse_batch_results(
    results: Union[str, List[Dict]],
    schema: Optional[Type[BaseModel]] = None,
    validate: bool = True,
    return_metadata: bool = False
) -> Union[List[BaseModel], List[str], tuple]:
    """
    Parse batch results into Pydantic models or raw text strings.

    Args:
        results: Either path to JSONL file or list of result dicts (from inline mode)
        schema: Optional Pydantic model class to parse results into. If None, returns raw text.
        validate: If True, validates each result against the schema (only when schema is provided)
        return_metadata: If True, returns tuple of (parsed_results, metadata_list)

    Returns:
        If return_metadata=False:
            - If schema provided: List of parsed Pydantic model instances (with None for failures)
            - If schema is None: List of raw text strings (with None for failures)
        If return_metadata=True:
            - Tuple of (parsed_results, metadata_list) where metadata_list contains
              usage metadata dicts with token counts, model versions, etc.

        Note: Output list length always matches input length. Failed/malformed results are
        represented as None to preserve alignment with input requests.
    """
    # Load results if file path provided
    if isinstance(results, str):
        json_lines = []
        total_lines = 0
        with open(results, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f.readlines(), start=1):
                total_lines += 1
                line = line.strip()
                if not line:
                    # Empty line - append None placeholder
                    json_lines.append(None)
                    continue
                try:
                    json_lines.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Malformed JSON on line {line_num} in {results}: {e}")
                    logger.debug(f"Skipping malformed line: {line[:100]}...")
                    # Append None to preserve alignment
                    json_lines.append(None)
    else:
        json_lines = results

    if not json_lines:
        logger.warning("No results content found to process.")
        if return_metadata:
            return [], []
        return []

    parsed_results: List[Union[BaseModel, str, None]] = []
    metadata_list: List[Union[Dict[str, Any], None]] = []
    success_count = 0
    failure_count = 0

    for line in json_lines:
        # Handle malformed JSONL lines (None placeholder)
        if line is None:
            parsed_results.append(None)
            if return_metadata:
                metadata_list.append(None)
            failure_count += 1
            continue

        # Handle both file-based (has 'key') and inline (has 'index') formats
        identifier = line.get('key', line.get('index', 'unknown'))

        if 'error' in line:
            logger.error(f"Request {identifier} failed with error: {line['error']}")
            parsed_results.append(None)
            if return_metadata:
                metadata_list.append(None)
            failure_count += 1
            continue

        if 'response' not in line:
            logger.warning(f"Skipping result {identifier} without 'response' field.")
            parsed_results.append(None)
            if return_metadata:
                metadata_list.append(None)
            failure_count += 1
            continue

        try:
            # Extract text response
            response = line['response']
            if hasattr(response, 'candidates'):
                # genai response object
                llm_output = response.candidates[0].content.parts[0].text
                # Extract metadata if requested
                if return_metadata:
                    metadata = {}
                    if hasattr(response, 'usage_metadata'):
                        metadata['usageMetadata'] = {
                            'totalTokenCount': getattr(response.usage_metadata, 'total_token_count', None),
                            'promptTokenCount': getattr(response.usage_metadata, 'prompt_token_count', None),
                            'candidatesTokenCount': getattr(response.usage_metadata, 'candidates_token_count', None),
                            'thoughtsTokenCount': getattr(response.usage_metadata, 'thoughts_token_count', None),
                        }
                    if hasattr(response, 'model_version'):
                        metadata['modelVersion'] = response.model_version
                    metadata_list.append(metadata)
            elif isinstance(response, dict):
                # dict format
                llm_output = response['candidates'][0]['content']['parts'][0]['text']
                # Extract metadata if requested
                if return_metadata:
                    metadata = {}
                    if 'usageMetadata' in response:
                        metadata['usageMetadata'] = response['usageMetadata']
                    if 'modelVersion' in response:
                        metadata['modelVersion'] = response['modelVersion']
                    metadata_list.append(metadata)
            else:
                logger.error(f"Unknown response format for {identifier}")
                parsed_results.append(None)
                if return_metadata:
                    metadata_list.append(None)
                failure_count += 1
                continue

            # If no schema, return raw text
            if schema is None:
                parsed_results.append(llm_output)
                success_count += 1
                logger.info(f"Successfully extracted raw text for {identifier}")
            else:
                # Extract JSON from text (handles markdown, explanatory text, etc.)
                extracted_json = extract_json_from_text(llm_output)
                if extracted_json is None:
                    # Fall back to original output if extraction fails
                    logger.debug(f"Could not extract JSON from text for {identifier}, using original output")
                    extracted_json = llm_output

                # Parse and validate with Pydantic
                if validate:
                    parsed = schema.model_validate_json(extracted_json)
                    parsed_results.append(parsed)
                    success_count += 1
                    logger.info(f"Successfully parsed and validated result for {identifier}")
                else:
                    data = json.loads(extracted_json)
                    parsed = schema.model_validate(data)
                    parsed_results.append(parsed)
                    success_count += 1
                    logger.info(f"Successfully parsed result for {identifier}")

        except (KeyError, IndexError, json.JSONDecodeError, AttributeError, ValidationError) as e:
            logger.error(f"Failed to parse result for {identifier}: {e}")
            parsed_results.append(None)
            if return_metadata:
                metadata_list.append(None)
            failure_count += 1

    logger.info(f"Parsed {success_count} successful results, {failure_count} failures")

    if return_metadata:
        return parsed_results, metadata_list
    return parsed_results
