"""
Central configuration for gemini-batch library.
"""

# --- Model and Generation Settings ---
MODEL_CONFIG = {
    "default_model": "gemini-2.5-flash",
    "generation_config": {
        "temperature": 0.6,
        "max_output_tokens": 8192,
    },
}

# --- Batch Processing Settings ---
BATCH_CONFIG = {
    "poll_interval": 60,  # seconds
    "completed_states": {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED'
    },
    "default_jsonl_dir": ".gemini_batch",  # Default directory for JSONL request files
    "default_results_dir": ".gemini_batch",  # Default directory for JSONL result files
}

# --- Image Processing Settings ---
IMAGE_PROCESSING_CONFIG = {
    "default_dpi": 200,
    "grayscale": False,
    "crop_sides": 0,
}

# --- Media Resolution Settings ---
MEDIA_RESOLUTION_OPTIONS = {
    "LOW": "MEDIA_RESOLUTION_LOW",       # Lower token usage, faster/cheaper, less detail
    "MEDIUM": "MEDIA_RESOLUTION_MEDIUM", # Balanced detail, cost, and speed
    "HIGH": "MEDIA_RESOLUTION_HIGH",     # Higher token usage, more detail, increased latency/cost
}

# --- Vertex AI Settings ---
# Configuration for Google Cloud Vertex AI backend (alternative to Gemini Developer API)
# Environment variable fallbacks:
#   GOOGLE_GENAI_USE_VERTEXAI: "true" to enable Vertex AI
#   GOOGLE_CLOUD_PROJECT: GCP project ID
#   GOOGLE_CLOUD_LOCATION: GCP region (default: us-central1)
VERTEXAI_CONFIG = {
    "project": None,  # GCP project ID (falls back to GOOGLE_CLOUD_PROJECT env var)
    "location": "us-central1",  # GCP region (falls back to GOOGLE_CLOUD_LOCATION env var)
    "gcs_bucket": None,  # GCS bucket name for file storage (auto-created if doesn't exist)
    "auto_create_bucket": True,  # Whether to auto-create GCS bucket if it doesn't exist
    "bucket_location": "US",  # Location for auto-created buckets
}
