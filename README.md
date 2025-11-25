# gemini-batch

Batch processing for Google Gemini API with structured Pydantic output. 50% cheaper than standard API.

## Installation

```bash
pip install gemini-batch
export GEMINI_API_KEY="your-api-key"
```

## Usage

### Text Processing

```python
from pydantic import BaseModel
from gemini_batch import batch_process

class Sentiment(BaseModel):
    sentiment: str
    confidence: float

prompts = [
    ["Analyze: I love this product!"],
    ["Analyze: This is terrible."],
]

results = batch_process(prompts=prompts, schema=Sentiment)
```

### Image Processing

```python
from pathlib import Path

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float

prompts = [
    ["Extract invoice data:", Path("invoice1.jpg")],
    ["Extract invoice data:", Path("invoice2.jpg")],
]

results = batch_process(prompts=prompts, schema=Invoice)
```

### Async Mode

```python
from gemini_batch import batch_process, monitor_batch_job, download_batch_results, parse_batch_results

# Submit without waiting
job_name = batch_process(prompts=prompts, schema=MySchema, wait=False)

# Check later
state = monitor_batch_job(job_name)
if state == "JOB_STATE_SUCCEEDED":
    results_file = download_batch_results(job_name)
    results = parse_batch_results(results_file, MySchema)
```

### Multiple Samples

```python
# Process each prompt 3 times for higher accuracy
results = batch_process(prompts=prompts, schema=MySchema, n_samples=3)
```

## API

### `batch_process(prompts, schema, **kwargs)`

**Key Parameters:**
- `prompts`: List of prompts (each prompt is a list of str/Path/PIL.Image/bytes)
- `schema`: Pydantic BaseModel class
- `model`: Model name (default: "gemini-2.5-flash")
- `wait`: Wait for completion (default: True)
- `n_samples`: Process each prompt N times
- `media_resolution`: Media quality control ("MEDIA_RESOLUTION_LOW"/"MEDIUM"/"HIGH")
- `temperature`, `max_output_tokens`: Generation config

**Returns:** List of Pydantic models (or job name if `wait=False`)

### Lower-level Functions

- `create_batch_job()`, `monitor_batch_job()`, `download_batch_results()`
- `parse_batch_results()`, `aggregate_records()`

## Error Handling

Starting with v0.4.0, `parse_batch_results()` preserves alignment between inputs and outputs:

```python
results = batch_process(prompts, schema)

# Results list has same length as prompts
assert len(results) == len(prompts)

# Failed requests return None
for i, result in enumerate(results):
    if result is None:
        print(f"Prompt {i} failed")
    else:
        print(f"Prompt {i} succeeded: {result}")

# Filter out failures if needed
successful_results = [r for r in results if r is not None]
```

### Breaking Changes in v0.4.0

**Previous behavior (v0.3.0):**
- Failed/invalid results were skipped
- Output list could be shorter than input list
- No way to determine which prompts failed

**New behavior (v0.4.0):**
- Failed/invalid results return `None`
- Output list always matches input length
- Clear indication of which prompts failed

**Migration:**
```python
# Old code (v0.3.0) - assumes all results succeeded
results = parse_batch_results("results.jsonl", MySchema)
for result in results:
    process(result)  # Works but can't tell which prompts failed

# New code (v0.4.0) - handle failures explicitly
results = parse_batch_results("results.jsonl", MySchema)
for i, result in enumerate(results):
    if result is None:
        print(f"Prompt {i} failed")
    else:
        process(result)

# Or filter out None values for old behavior
successful_results = [r for r in results if r is not None]
```

## Testing

```bash
pip install -e ".[dev]"

# Unit tests (mocked, no API key)
pytest -m "not integration"

# Integration tests (live API, < $0.01)
export GEMINI_API_KEY="your-key"
pytest -m integration -v -s
```

## Links

- [Gemini Batch API Docs](https://ai.google.dev/gemini-api/docs/batch-api)
- [Issue Tracker](https://github.com/phenschke/gemini-batch/issues)
