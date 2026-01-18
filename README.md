# gemini-batch

Batch processing for Google Gemini API with structured Pydantic output. 50% cheaper than standard API.

Supports both **Gemini Developer API** and **Vertex AI** backends.

## Installation

```bash
# Standard (Gemini Developer API)
pip install gemini-batch
export GEMINI_API_KEY="your-api-key"

# With Vertex AI support (Google Cloud)
pip install gemini-batch[vertexai]
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

**Media quality control:** Adjust token usage vs quality
```python
results = batch_process(prompts, schema, media_resolution="MEDIA_RESOLUTION_HIGH")
```

**Per-part media resolution:** Fine-grained control using `google.genai.types.Part`
```python
from google.genai import types

prompts = [[
    "Compare these images:",
    types.Part(
        file_data=types.FileData(file_uri="...", mime_type="image/png"),
        media_resolution=types.PartMediaResolution(level="MEDIA_RESOLUTION_ULTRA_HIGH")
    ),
    types.Part(
        file_data=types.FileData(file_uri="...", mime_type="image/png"),
        media_resolution=types.PartMediaResolution(level="MEDIA_RESOLUTION_LOW")
    ),
]]
results = batch_process(prompts, schema)
```

**Batch embeddings:** Generate embeddings at 50% cost (Gemini Developer API only)
```python
from gemini_batch import batch_embed
embeddings = batch_embed(["doc 1", "doc 2"], task_type="RETRIEVAL_DOCUMENT")
# Returns: [[0.12, -0.45, ...], [0.78, ...]]  (3072-dim)
```

**Direct API embeddings:** Immediate results at full cost (when speed matters)
```python
from gemini_batch import embed
embeddings = embed(["text 1", "text 2"], task_type="RETRIEVAL_QUERY")
# Returns immediately (no batch job), 3072-dim vectors
```
Task types: `RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`

**Token statistics:** Analyze token usage across batches
```python
from gemini_batch import calculate_token_statistics
results, metadata = batch_process(prompts, schema, return_metadata=True)
stats = calculate_token_statistics(metadata)
print(f"Total tokens: {stats.total_tokens}")
print(f"Average per request: {stats.avg_total_tokens:.2f}")
```

## Vertex AI Support

Use Google Cloud Vertex AI instead of the Gemini Developer API. Vertex AI uses Google Cloud Storage (GCS) for file handling instead of the File API.

### Setup

```bash
# Install with Vertex AI support
pip install gemini-batch[vertexai]

# Authenticate with Google Cloud
gcloud auth application-default login

# Set required environment variables
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project-id
```

### Usage

```python
# Option 1: Via environment variables (recommended)
# With GOOGLE_GENAI_USE_VERTEXAI=true set:
results = batch_process(prompts, schema)

# Option 2: Explicit parameters
results = batch_process(
    prompts,
    schema,
    vertexai=True,
    project="your-gcp-project",
    location="us-central1",  # Optional, defaults to us-central1
    gcs_bucket="my-batch-bucket",  # Optional, auto-created if not specified
)
```

### How It Works

When using Vertex AI:
- Files (images, JSONL) are uploaded to GCS instead of File API
- Batch results are stored in GCS and downloaded automatically
- GCS buckets are auto-created if they don't exist (configurable)
- Authentication uses Application Default Credentials (ADC)

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

[Gemini Batch API Docs](https://ai.google.dev/gemini-api/docs/batch-api) • [Vertex AI Batch Docs](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction) • [Issues](https://github.com/phenschke/gemini-batch/issues)
