# CLAUDE.md

## Project Overview

**gemini-batch**: Simple Python library for batch processing with Google Gemini API. Single-function interface for structured Pydantic output at 50% cost savings.

**Design Principles:** Simplicity, type-safety, fail-gracefully error handling, comprehensive testing (mocked + live API)

## Architecture

### Core API

```python
# Gemini Batch API (50% cost savings, slower)
results = batch_process(prompts, schema, wait=True)

# Batch Embeddings (50% cost savings)
embeddings = batch_embed(texts, task_type="RETRIEVAL_DOCUMENT")

# OpenAI-compatible APIs (DeepSeek, Together, Groq, etc.)
results = await async_process(prompts, schema, model="deepseek-chat", base_url="https://api.deepseek.com")
# Or sync wrapper:
results = process(prompts, schema, model="deepseek-chat", base_url="https://api.deepseek.com")

# Low-level batch control
job_name = create_batch_job(requests)
monitor_batch_job(job_name)
results_file = download_batch_results(job_name)
parsed = parse_batch_results(results_file, schema)

# Resume failed Vertex AI batch jobs
results = batch_process(prompts=[], schema=MySchema, resume_from="projects/.../batchJobs/123", vertexai=True)
# Or low-level:
new_job = resume_batch_job("projects/.../batchJobs/123")  # or GCS URI
```

### Modules

- **`batch.py`**: Job creation, monitoring, result download/parsing
- **`embedding.py`**: Batch embedding generation with task type support
- **`async_batch.py`**: Async processing for OpenAI-compatible APIs (install with `pip install gemini-batch[async]`)
- **`utils.py`**: `GeminiClient`, file upload, PDF conversion, config building, JSON extraction
- **`aggregation.py`**: Majority voting for `n_samples > 1`
- **`config.py`**: Model, batch, image processing, media resolution, embedding, async defaults

## Implementation Details

### Prompts (v0.2.0+)

`prompts` is a list of lists. Each inner list contains parts:
- **Strings**: Text parts
- **`pathlib.Path`**: Image files (uploaded to File API)
- **`PIL.Image`**: Images (saved to temp file, uploaded)
- **Bytes**: Raw bytes (saved to temp file, uploaded)
- **`google.genai.types.Part`**: Pre-built Part objects for per-part media resolution control

**Important:** Strings are ALWAYS text. Use `Path()` for file paths.

```python
from pathlib import Path
prompts = [
    ["Analyze this invoice:", Path("invoice.jpg"), " Extract total"],
    ["What is 2+2?"]
]
```

**Per-part media resolution:** Use `types.Part` for fine-grained control:
```python
from google.genai import types

prompts = [[
    "Compare:",
    types.Part(
        file_data=types.FileData(file_uri="...", mime_type="image/png"),
        media_resolution=types.PartMediaResolution(level="MEDIA_RESOLUTION_ULTRA_HIGH")
    ),
    types.Part(
        file_data=types.FileData(file_uri="...", mime_type="image/png"),
        media_resolution=types.PartMediaResolution(level="MEDIA_RESOLUTION_LOW")
    ),
]]
```
- Part objects must have `file_data` with `file_uri` set, or `text`
- `inline_data` not supported (use bytes type directly for uploads)
- Per-part `media_resolution` overrides global `part_media_resolution`

**PDFs**: Not directly supported. Use `pdf_pages_to_images()` utility first for explicit page selection.

### Processing Mode

File-based batch processing for all requests:
- Requests → JSONL file → upload to Gemini
- Images → File API → URI references in JSONL
- Results → JSONL file download
- Files: `batch_{timestamp}_requests.jsonl` / `_results.jsonl` in `.gemini_batch/` (gitignored, not auto-cleaned)

### Resuming Failed Jobs (Vertex AI Only)

Resume failed/interrupted batch jobs by using previous output as input. The system automatically skips completed requests and only processes incomplete ones.

```python
from gemini_batch import batch_process, resume_batch_job, monitor_batch_job

# High-level API: resume and get parsed results
results = batch_process(
    prompts=[],  # Empty when resuming
    schema=MySchema,
    resume_from="projects/.../batchJobs/123",  # or GCS URI
    vertexai=True,
)

# Low-level API: more control over the process
new_job_name = resume_batch_job("projects/.../batchJobs/123")
# Or resume from GCS URI directly:
new_job_name = resume_batch_job("gs://my-bucket/batch-results/batch-123/predictions.jsonl")
state = monitor_batch_job(new_job_name)
```

**Key Implementation Details:**
- Uses `get_batch_job_output_uri()` to find the predictions file from a job name
- Creates a new batch job with previous output as `src` (input source)
- Vertex AI recognizes completed requests by `key` and skips them
- New results merged with previous output in the new job's output location

**Functions:**
- `get_batch_job_output_uri(job_name)`: Get GCS URI for a job's output file
- `resume_batch_job(job_name_or_gcs_uri, ...)`: Create new job from previous output
- `batch_process(..., resume_from="...")`: High-level resume with parsing

### Batch Embeddings

Generate embeddings at 50% cost via Gemini Batch API:

```python
from gemini_batch import batch_embed

# Basic usage
embeddings = batch_embed(
    texts=["Document 1 content...", "Document 2 content..."],
    task_type="RETRIEVAL_DOCUMENT",  # or RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, etc.
)
# Returns: [[0.123, -0.456, ...], [0.789, -0.012, ...]]  (3072-dim vectors)
```

**Task Types:**
- `RETRIEVAL_DOCUMENT`: For documents to be retrieved/indexed
- `RETRIEVAL_QUERY`: For search queries
- `SEMANTIC_SIMILARITY`: For similarity comparisons
- `CLASSIFICATION`: For text classification
- `CLUSTERING`: For clustering tasks

**Key Implementation Details:**
- Uses `client.batches.create_embeddings()` (different from text generation's `create()`)
- JSONL format: `{"key": "0", "request": {"content": {"parts": [{"text": "..."}]}, "task_type": "..."}}`
- Result parsing extracts `response["embedding"]["values"]`
- Model: `gemini-embedding-001` (default), produces 3072-dimensional vectors

### Async Processing (OpenAI-compatible APIs)

For providers without batch APIs (DeepSeek, Together, Groq, OpenRouter, etc.):

```python
# Async
results = await async_process(
    prompts=[["What is 2+2?"]],
    schema=Answer,
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="...",  # or OPENAI_API_KEY env var
    max_concurrent=10,  # rate limiting
)

# Sync wrapper
results = process(prompts, schema, model="deepseek-chat", ...)
```

**Key differences from batch_process:**
- Uses `openai` SDK's `AsyncOpenAI` (true asyncio)
- Images encoded as base64 data URLs (not file URIs)
- Rate limiting via `asyncio.Semaphore(max_concurrent)`
- Structured output via JSON mode + Pydantic parsing (not all providers support json_schema)
- Retry with exponential backoff (`retry_count`, `retry_delay`)

**Install:** `pip install gemini-batch[async]` or `pip install gemini-batch[openai]`

### Structured Output & Parsing

- Pydantic schemas passed to `build_generation_config()` via `response_schema`
- Robust JSON extraction (`extract_json_from_text()` in `utils.py`):
  - Strips markdown code blocks
  - Extracts JSON from explanatory text
  - Falls back to original text on failure
- All parsing errors are logged but non-fatal → returns partial successful results

### Media Resolution (v0.3.0+)

Controls quality vs cost: `"MEDIA_RESOLUTION_LOW"` | `"MEDIUM"` | `"HIGH"` (70-1120 tokens)

### Error Handling (Fail-Gracefully)

- **Job-level**: Missing API key raises `ValueError`; job failures logged
- **Request-level**: Per-request failures, malformed JSONL, validation errors → logged, skipped, batch continues
- **Tracking**: `batch_stats.failed_request_count`

## Testing

### Unit Tests (Mocked, Fast)

Run: `.venv/bin/python -m pytest -m "not integration"`

Coverage:
- **`test_utils.py`**: File upload, PDF conversion, config, API client, JSON extraction
- **`test_batch.py`**: Job creation, monitoring, result parsing (including robust parsing: markdown, malformed JSONL, validation errors)
- **`test_embedding.py`**: Embedding batch job creation, result parsing, task type validation
- **`test_async_batch.py`**: Async processing, media encoding, OpenAI message building, retry logic
- **`test_aggregation.py`**: Majority voting

### Integration Tests (Live API, Essential)

Run: `.venv/bin/python -m pytest -m integration -v -s` (requires `GEMINI_API_KEY`)

**Core workflows** (2 inputs each, < $0.01/run):
1. Text → Simple structured output
2. Text → Rich structured output
3. Text + Image → Structured output (multimodal)
4. Text → Embeddings (with task types)

**Why essential**: Verifies library works with real Gemini Batch API quirks.

## Development Workflow

**Note**: Uses `uv` for package management. Run tests with `.venv/bin/python -m pytest`.

```bash
pip install -e ".[dev]"                                        # Install
.venv/bin/python -m pytest -m "not integration"                # Unit tests
.venv/bin/python -m pytest -m integration -v -s                # Integration (needs GEMINI_API_KEY)
.venv/bin/python -m pytest -m "not integration" --cov=gemini_batch  # Coverage
```

### Adding Features

1. Write unit tests (mocked API)
2. Implement until tests pass
3. Add integration test if affects core workflows (text→structured, multimodal→structured)
4. Update docs (README.md, docstrings, CLAUDE.md)

### API Changes

1. Integration tests will fail
2. Update `batch.py` or `utils.py`
3. Update mocks in unit tests
4. Re-run integration tests

## API Quirks & Notes

**Response structures** (handled in `parse_batch_results()`):
- Inline: `batch_job.dest.inlined_responses` vs File: `batch_job.dest.file_name`
- genai objects: `.candidates[0].content.parts[0].text` vs dict: `['candidates'][0]['content']['parts'][0]['text']`

**Job states**: Check all completed states: `JOB_STATE_SUCCEEDED`, `FAILED`, `CANCELLED`, `EXPIRED`

**Failed requests**: Don't fail whole job; check `batch_stats.failed_request_count`

**Rate limiting**: `GeminiClient.generate_content()` has 15 calls/60s limit (not used by batch API)

**File cleanup**: `.gemini_batch/batch_{timestamp}_requests.jsonl` and `_results.jsonl` (gitignored, not auto-cleaned for debugging)

**PDF utility**: `pdf_pages_to_images()` in `utils.py` - extracts embedded images or renders at DPI

## Aggregation (`n_samples > 1`)

```python
from gemini_batch import aggregate_records, ListVoteConfig

result = aggregate_records(
    records=[sample1, sample2, sample3],
    list_configs={("items",): ListVoteConfig(match_on=("id", "name"))},
    as_model=MySchema
)
disagreements = result.disagreements()  # Track consensus quality
```

- **Scalars**: Most common value
- **Lists**: Index or field-based alignment via `ListVoteConfig`
- **Objects**: Recursive aggregation
