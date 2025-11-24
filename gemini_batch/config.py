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
    "default_jsonl_dir": ".tmp",  # Default directory for temporary JSONL request files
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
