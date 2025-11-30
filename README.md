# gemini-batch

Batch processing for Google Gemini API with structured Pydantic output. 50% cheaper than standard API.

## Installation

```bash
pip install gemini-batch
export GEMINI_API_KEY="your-api-key"
```

## Quick Start

```python
from pydantic import BaseModel
from gemini_batch import batch_process
from pathlib import Path

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float

prompts = [
    ["Extract invoice data:", Path("invoice1.jpg")],
    ["Extract invoice data:", Path("invoice2.jpg")],
]

results = batch_process(prompts=prompts, schema=Invoice)
# Returns: [Invoice(...), Invoice(...)]
```

## Features

**Simple API:** Single function handles everything
```python
results = batch_process(prompts, schema)
```

**Multimodal:** Mix text, images (Path/PIL.Image/bytes) in any prompt
```python
prompts = [["Analyze:", Path("chart.png"), "What's the trend?"]]
```

**Async:** Submit jobs without waiting
```python
job_name = batch_process(prompts, schema, wait=False)
# ... check later with monitor_batch_job(job_name)
```

**Multiple samples:** Run each prompt N times for consensus
```python
results = batch_process(prompts, schema, n_samples=3)
```

**Media quality control:** Adjust token usage vs quality
```python
results = batch_process(prompts, schema, media_resolution="MEDIA_RESOLUTION_HIGH")
```

## Error Handling

Failed requests return `None` while preserving input-output alignment:

```python
results = batch_process(prompts, schema)  # Same length as prompts
successful = [r for r in results if r is not None]
```

## Development

```bash
pip install -e ".[dev]"
pytest -m "not integration"  # Unit tests (mocked)
pytest -m integration        # Integration tests (live API, < $0.01)
```

[API Docs](https://ai.google.dev/gemini-api/docs/batch-api) • [Issues](https://github.com/phenschke/gemini-batch/issues)
