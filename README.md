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
