# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**gemini-batch** is a simple, generic Python library for batch processing with Google Gemini API. It provides a single-function interface for processing large volumes of requests with structured Pydantic output at 50% cost savings.

**Key Design Principles:**
- **Simplicity**: Single `batch_process()` function handles everything
- **Generic**: No domain-specific code - works for any use case
- **Type-safe**: Full Pydantic validation and structured output
- **Cost-effective**: Uses Gemini Batch API for 50% savings
- **Well-tested**: Comprehensive unit tests (mocked) and integration tests (live API)

## Architecture

### Core API Design

The library exposes a simple high-level API with low-level functions available for advanced use:

**High-level (recommended):**
```python
results = batch_process(prompts, schema, wait=True)
```

where `prompts` is a list of lists, with each inner list containing text strings and/or image objects.

**Low-level (for advanced control):**
```python
job_name = create_batch_job(requests)
state = monitor_batch_job(job_name)
results_file = download_batch_results(job_name)
parsed = parse_batch_results(results_file, schema)
```

### Module Structure

- **`__init__.py`**: Main entry point
  - `batch_process()`: Single-function API that handles prompt parsing, file uploads, job creation, monitoring, and parsing
  - Exports all public functions and classes

- **`batch.py`**: Core batch workflow
  - `create_batch_job()`: Creates batch jobs (always file-based mode)
  - `monitor_batch_job()`: Polls job status until completion
  - `download_batch_results()`: Downloads file-based results
  - `get_inline_results()`: Retrieves inline results
  - `parse_batch_results()`: Parses JSONL or inline results into Pydantic models

- **`utils.py`**: Shared utilities
  - `GeminiClient`: API client wrapper
  - `upload_file_to_gemini()`: Uploads files to Gemini File API (returns URI and MIME type)
  - `pdf_pages_to_images()`: Extracts or renders PDF pages as images
  - `build_generation_config()`: Builds generation config from Pydantic schema
  - `get_api_key()`: Retrieves API key from environment

- **`aggregation.py`**: Majority voting system
  - `MajorityVoteAggregator`: Aggregates multiple samples via voting
  - `aggregate_records()`: Convenience function for aggregation
  - `ListVoteConfig`: Configures list alignment strategies (index vs field-based)
  - Used when `n_samples > 1` to build consensus from multiple results

- **`config.py`**: Configuration constants
  - `MODEL_CONFIG`: Default model and generation settings
  - `BATCH_CONFIG`: Polling interval, completed states, default directories (`.gemini_batch/`)
  - `IMAGE_PROCESSING_CONFIG`: DPI, grayscale, cropping defaults
  - `MEDIA_RESOLUTION_OPTIONS`: Valid media resolution values for quality control

## Key Implementation Details

### Prompt Structure (v0.2.0+)

`batch_process()` accepts `prompts` as a list of lists, where each inner list contains parts:
- **Text parts**: Strings are added as `{"text": "..."}` parts
- **Image files**: `pathlib.Path` objects are uploaded to File API, added as `{"file_data": {...}}` parts
- **PIL Images**: `Image.Image` objects are saved to temp files, uploaded, then added as file_data parts
- **Raw bytes**: Bytes are saved to temp files, uploaded, then added as file_data parts

**Important:** Raw strings are ALWAYS treated as text. Use `pathlib.Path()` objects for file paths.

**Example:**
```python
from pathlib import Path

prompts = [
    ["Analyze this invoice:", Path("invoice.jpg"), " Extract total amount"],  # Text + image + text
    ["What is 2+2?"],  # Text only
]
```

**Note on PDFs**: Direct PDF input is not supported in `batch_process()`. Users should preprocess PDFs using `pdf_pages_to_images()` utility function to convert pages to images first. This makes the behavior explicit and allows users to control which pages to process.

### Batch Processing Mode

The library uses file-based batch processing for all requests:
- All requests are written to a JSONL file and uploaded to Gemini
- Images are uploaded to Gemini File API and referenced by URI in the JSONL
- Results are downloaded as a JSONL file after processing
- Simpler and more reliable than inline mode (avoids size limits and API quirks)

### Structured Output

All responses use Pydantic schemas for validation:
- Schema passed via `response_schema` parameter to `build_generation_config()`
- Gemini API returns JSON matching the schema
- Results parsed and validated with `schema.model_validate_json()`
- Robust JSON extraction handles LLM responses with explanatory text or markdown wrappers

### Media Resolution Control (v0.3.0+)

The `media_resolution` parameter controls media quality vs cost tradeoff:
- Valid values: `"MEDIA_RESOLUTION_LOW"`, `"MEDIA_RESOLUTION_MEDIUM"`, `"MEDIA_RESOLUTION_HIGH"`
- Controls token allocation for images/videos (70-1120 tokens depending on setting)
- Passed to `batch_process()` and `build_generation_config()`

### Robust JSON Parsing

The library includes robust JSON parsing to handle common LLM output variations:

**Handled patterns:**
- **Markdown code blocks**: ````json\n{...}\n```` → Automatically stripped
- **Explanatory text**: `"Here is the data: {...}"` → JSON extracted
- **Mixed content**: `"Result: {...} as requested"` → JSON extracted
- **Malformed JSONL lines**: Individual corrupted lines skipped, batch continues
- **Schema validation errors**: Individual invalid results skipped, batch continues

**Implementation:**
- `extract_json_from_text()` utility in `utils.py` handles extraction
- Uses regex for markdown blocks and bracket matching for embedded JSON
- Falls back to original text if extraction fails
- All errors are logged but non-fatal - library returns partial successful results

### Error Handling

- **Missing API key**: Raises `ValueError` with clear message
- **Job failures**: Logged with error details from `batch_job.error`
- **Per-request failures**: Logged but don't fail entire batch
- **Malformed JSONL lines**: Skipped with error log, processing continues
- **JSON extraction failures**: Attempts extraction, falls back to original text
- **Validation errors**: Logged and skipped (non-fatal), batch continues
- **Failed requests**: Counted in `batch_stats.failed_request_count`

**Fail-gracefully pattern**: The library prioritizes returning partial successful results over failing entirely. All parsing errors are logged but don't stop batch processing.

## Testing Strategy

### Unit Tests (Fast, No API)

All unit tests use mocked API responses and focus on individual components:
- **`test_utils.py`**: File upload to Gemini File API, PDF conversion, config building, API client, JSON extraction
- **`test_batch.py`**: Batch job creation, monitoring, result parsing (including robust parsing scenarios)
- **`test_aggregation.py`**: Majority voting logic

**Robust parsing test coverage:**
- JSON extraction from markdown code blocks (with/without language specifier)
- JSON extraction from text with explanatory prefixes/suffixes
- Malformed JSONL lines don't crash batch parsing
- Schema validation errors are non-fatal
- Complex mixed issues (markdown + malformed lines + validation errors)

Run with: `.venv/bin/python -m pytest -m "not integration"`

### Integration Tests (Live API)

Integration tests verify the **three core batch workflows** with real API calls:

1. **Text → Simple structured output** (`test_text_to_simple_structured`)
   - Basic batch processing with text prompts
   - Single-field Pydantic schema response

2. **Text → Rich structured output** (`test_text_to_rich_structured`)
   - Multi-field Pydantic schemas
   - Validates complex structured extraction

3. **Text + Image → Structured output** (`test_multimodal_to_structured`)
   - Multimodal prompts with text and image bytes
   - Tests file upload and end-to-end multimodal processing

**Key characteristics:**
- Uses `gemini-flash-lite-latest` (cheapest model)
- Minimal inputs (2 per test) keep cost < $0.01 per run
- Marked with `@pytest.mark.integration`
- Auto-skips if `GEMINI_API_KEY` not set
- Simple, illustrative examples of core functionality

Run with: `.venv/bin/python -m pytest -m integration -v -s`

**Important**: These integration tests are essential for verifying the library works with the real Gemini Batch API, which can have quirks in response formats and behavior.

## Common Development Tasks

### Running Tests

**Note**: This project uses `uv` for package management and virtual environment management. Tests should be run using `.venv/bin/python -m pytest` or through `uv` commands to ensure correct environment isolation.

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Unit tests only (fast)
.venv/bin/python -m pytest -m "not integration"

# Integration tests (requires API key)
export GEMINI_API_KEY="your-key"
.venv/bin/python -m pytest -m integration -v -s

# All tests
.venv/bin/python -m pytest

# With coverage
.venv/bin/python -m pytest -m "not integration" --cov=gemini_batch
```

### Adding New Features

When adding features, follow this workflow:

1. **Write unit tests first** with mocked API responses
2. **Implement the feature** until unit tests pass
3. **Add integration test if core functionality changes**:
   - Only add integration tests for changes that affect the three core workflows
   - Keep integration tests simple and illustrative
   - Focus on happy path, not edge cases
4. **Update documentation** (README.md, docstrings, CLAUDE.md)

### Handling API Changes

If Gemini API behavior changes:

1. **Check integration tests**: They will fail if API breaks
2. **Update affected functions**: Usually in `batch.py` or `utils.py`
3. **Update mocks in unit tests**: Match new API response format
4. **Re-run integration tests**: Verify fix works with live API

## Important Notes

### API Quirks to Watch For

The Gemini Batch API has some quirks that the library handles:

- **Inline vs file-based results**: Different response structures
  - Inline: `batch_job.dest.inlined_responses` (list of response objects)
  - File: `batch_job.dest.file_name` (JSONL file to download)

- **Response format variations**:
  - genai objects have `.candidates[0].content.parts[0].text`
  - Dict format uses `['candidates'][0]['content']['parts'][0]['text']`
  - Both are handled in `parse_batch_results()`

- **Job states**: Always check for all completed states:
  - `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`, `JOB_STATE_CANCELLED`, `JOB_STATE_EXPIRED`

- **Failed requests**: Individual requests can fail without failing the whole job
  - Check `batch_stats.failed_request_count`
  - Parse results handle missing 'response' fields gracefully

### Rate Limiting

`GeminiClient.generate_content()` has built-in rate limiting (15 calls/60s). This only affects non-batch direct API calls, which we don't use in this library. Batch API has separate rate limits.

### File Cleanup

The library creates JSONL files for batch processing:
- **Request files**: Named `batch_{timestamp}_requests.jsonl`
- **Result files**: Named `batch_{timestamp}_results.jsonl`
- **Default location**: `.gemini_batch/` directory for both requests and results
- The `.gemini_batch/` directory is gitignored to avoid cluttering repositories
- **Not automatically cleaned up** - files remain for debugging/auditing purposes
- Both request and result files use the same timestamp for easy pairing

**Timestamp coordination**: The timestamp is embedded in the job's `display_name` (default: `batch-{timestamp}`) during creation, then extracted during result download to ensure matching filenames.

### PDF Processing Utility

The `pdf_pages_to_images()` function in `utils.py` is available as a preprocessing utility:
- Prefers extracting embedded images over rendering
- Falls back to rendering at specified DPI if no embedded image found
- Uses PyMuPDF (fitz) for PDF handling
- Users should call this manually to convert PDFs to images before passing to `batch_process()`
- This design makes page selection explicit (which pages to process) rather than implicit

## Aggregation System

When `n_samples > 1`, multiple results can be aggregated via majority voting:

- **Scalar fields**: Most common value wins
- **Nested objects**: Recursive aggregation
- **Lists**: Can align by index or by field matching
  - Index alignment: Match items by position
  - Field matching: Match items across samples by key fields (e.g., name + type)
- **ListVoteConfig**: Configure alignment strategy per list path
- **Disagreements**: Track agreement percentage for every field

Example:
```python
from gemini_batch import aggregate_records, ListVoteConfig

result = aggregate_records(
    records=[sample1, sample2, sample3],
    list_configs={
        ("items",): ListVoteConfig(match_on=("id", "name"))
    },
    as_model=MySchema
)

# Check consensus quality
disagreements = result.disagreements()
```

## Versioning and Releases

- **Version**: Defined in `pyproject.toml` and `__init__.py`
- **Current**: 0.3.0
- Update both locations when bumping version

**Version History:**
- **0.3.0**: Added media_resolution parameter for quality vs cost control
- **0.2.0**: Improved prompt structure with multi-part support
- **0.1.0**: Initial release (alpha)

## Dependencies

**Core runtime:**
- `google-genai`: Gemini API client
- `pydantic`: Schema validation
- `pillow`: Image processing
- `pymupdf`: PDF handling
- `ratelimit`: Rate limiting decorator
- `polars`: DataFrame operations (only used in aggregation)

**Dev tools:**
- `pytest`: Testing framework
- `pytest-mock`: Mocking utilities
- `black`: Code formatting
- `ruff`: Linting
- `mypy`: Type checking
