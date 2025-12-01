# CLAUDE.md

## Project Overview

**gemini-batch**: Simple Python library for batch processing with Google Gemini API. Single-function interface for structured Pydantic output at 50% cost savings.

**Design Principles:** Simplicity, type-safety, fail-gracefully error handling, comprehensive testing (mocked + live API)

## Architecture

### Core API

```python
# High-level (recommended)
results = batch_process(prompts, schema, wait=True)

# Low-level (advanced control)
job_name = create_batch_job(requests)
monitor_batch_job(job_name)
results_file = download_batch_results(job_name)
parsed = parse_batch_results(results_file, schema)
```

### Modules

- **`batch.py`**: Job creation, monitoring, result download/parsing
- **`utils.py`**: `GeminiClient`, file upload, PDF conversion, config building, JSON extraction
- **`aggregation.py`**: Majority voting for `n_samples > 1`
- **`config.py`**: Model, batch, image processing, media resolution defaults

## Implementation Details

### Prompts (v0.2.0+)

`prompts` is a list of lists. Each inner list contains parts:
- **Strings**: Text parts
- **`pathlib.Path`**: Image files (uploaded to File API)
- **`PIL.Image`**: Images (saved to temp file, uploaded)
- **Bytes**: Raw bytes (saved to temp file, uploaded)

**Important:** Strings are ALWAYS text. Use `Path()` for file paths.

```python
from pathlib import Path
prompts = [
    ["Analyze this invoice:", Path("invoice.jpg"), " Extract total"],
    ["What is 2+2?"]
]
```

**PDFs**: Not directly supported. Use `pdf_pages_to_images()` utility first for explicit page selection.

### Processing Mode

File-based batch processing for all requests:
- Requests → JSONL file → upload to Gemini
- Images → File API → URI references in JSONL
- Results → JSONL file download
- Files: `batch_{timestamp}_requests.jsonl` / `_results.jsonl` in `.gemini_batch/` (gitignored, not auto-cleaned)

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
- **`test_aggregation.py`**: Majority voting

### Integration Tests (Live API, Essential)

Run: `.venv/bin/python -m pytest -m integration -v -s` (requires `GEMINI_API_KEY`)

**Three core workflows** (2 inputs each, < $0.01/run):
1. Text → Simple structured output
2. Text → Rich structured output
3. Text + Image → Structured output (multimodal)

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
