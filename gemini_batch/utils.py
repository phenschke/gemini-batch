"""
Core utilities for gemini-batch: client management, image processing, and utilities.
"""

import os
import io
import re
import mimetypes
import tempfile
import uuid
from typing import Union, Optional, List, Type, Dict, Any, Tuple, get_origin, get_args
from pathlib import Path
from pydantic import BaseModel
from PIL import Image
from google import genai
from google.genai import types
from google.genai import _transformers
import pymupdf as fitz
import logging

from . import config
from .types import TokenStatistics

# Optional GCS import for Vertex AI support
try:
    from google.cloud import storage as gcs_storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    gcs_storage = None

# Optional async GCS import for parallel uploads (Vertex AI)
try:
    from gcloud.aio.storage import Storage as AsyncGCSStorage
    HAS_ASYNC_GCS = True
except ImportError:
    HAS_ASYNC_GCS = False
    AsyncGCSStorage = None

# Logging configuration
def setup_logging(level: int = logging.INFO, format_string: str = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s", filename: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    handlers = []
    if filename:
        handlers.append(logging.FileHandler(filename))
    handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )

    # Suppress verbose HTTP request logs from httpx/httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logging.getLogger("gemini_batch")

logger = setup_logging()


def is_list_schema(schema: Any) -> bool:
    """
    Check if schema is a List[BaseModel] type.

    Args:
        schema: The schema to check (could be a type, generic alias, or None)

    Returns:
        True if schema is List[SomeBaseModel], False otherwise
    """
    if schema is None:
        return False
    origin = get_origin(schema)
    if origin is list:
        args = get_args(schema)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return True
    return False


def create_list_wrapper(item_type: Type[BaseModel]) -> Type[BaseModel]:
    """
    Create a wrapper Pydantic model for a list of items.

    The Google GenAI SDK doesn't support List[T] as a top-level schema,
    so we wrap it in a model with an 'items' field.

    Args:
        item_type: The Pydantic BaseModel class for list items

    Returns:
        A new Pydantic model class with an 'items: List[item_type]' field
    """
    from pydantic import create_model
    return create_model('ListWrapper', items=(List[item_type], ...))


# API key configuration
def get_api_key() -> str:
    """Get Gemini API key from environment."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return api_key


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON from text that may contain explanatory content or markdown formatting.

    This function handles common patterns where LLMs wrap JSON in additional text:
    - Markdown code blocks: ```json {...} ```
    - Explanatory prefixes: "Here's the result: {...}"
    - Trailing text after JSON

    Args:
        text: Raw text that may contain JSON

    Returns:
        Cleaned JSON string, or None if no valid JSON structure found

    Examples:
        >>> extract_json_from_text('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> extract_json_from_text('Here is the data: {"key": "value"}')
        '{"key": "value"}'
        >>> extract_json_from_text('{"key": "value"}')
        '{"key": "value"}'
    """
    if not text or not isinstance(text, str):
        return None

    # Strip whitespace
    text = text.strip()

    # Pattern 1: Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    markdown_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    markdown_match = re.search(markdown_pattern, text, re.DOTALL)
    if markdown_match:
        extracted = markdown_match.group(1).strip()
        if extracted:
            logger.debug("Extracted JSON from markdown code block")
            return extracted

    # Pattern 2: Find JSON object {...} or array [...]
    # Look for the first { or [ and find its matching closing bracket
    json_start_obj = text.find('{')
    json_start_arr = text.find('[')

    # Determine which comes first (if any)
    json_start = -1
    is_object = True

    if json_start_obj != -1 and json_start_arr != -1:
        json_start = min(json_start_obj, json_start_arr)
        is_object = (json_start == json_start_obj)
    elif json_start_obj != -1:
        json_start = json_start_obj
        is_object = True
    elif json_start_arr != -1:
        json_start = json_start_arr
        is_object = False

    if json_start == -1:
        # No JSON structure found
        return None

    # Find matching closing bracket
    open_bracket = '{' if is_object else '['
    close_bracket = '}' if is_object else ']'

    depth = 0
    in_string = False
    escape_next = False
    json_end = -1

    for i in range(json_start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == open_bracket:
                depth += 1
            elif char == close_bracket:
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

    if json_end != -1:
        extracted = text[json_start:json_end]
        if json_start > 0:
            logger.debug(f"Extracted JSON from text (prefix removed: '{text[:json_start][:50]}...')")
        return extracted

    # If we couldn't find a complete JSON structure, return None
    return None


class GeminiClient:
    """
    Wrapper for Gemini client with reusable connection.
    
    Supports both Gemini Developer API (default) and Vertex AI backends.
    Backend selection:
    - Explicitly via vertexai=True parameter
    - Via GOOGLE_GENAI_USE_VERTEXAI="true" environment variable
    
    For Vertex AI, authentication uses Application Default Credentials (ADC).
    Set up with: gcloud auth application-default login
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        vertexai: Optional[bool] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (for Developer API). Falls back to GEMINI_API_KEY env var.
            vertexai: If True, use Vertex AI backend. If None, auto-detect from
                      GOOGLE_GENAI_USE_VERTEXAI env var.
            project: GCP project ID (for Vertex AI). Falls back to GOOGLE_CLOUD_PROJECT env var.
            location: GCP region (for Vertex AI). Falls back to GOOGLE_CLOUD_LOCATION env var,
                      then to config default (us-central1).
            gcs_bucket: GCS bucket name for file storage (Vertex AI only). Falls back to
                        config default. Will be auto-created if it doesn't exist.
        """
        # Determine backend
        if vertexai is None:
            self.vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
        else:
            self.vertexai = vertexai
        
        if self.vertexai:
            # Vertex AI configuration
            self.project = (
                project
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or config.VERTEXAI_CONFIG["project"]
            )
            self.location = (
                location
                or os.getenv("GOOGLE_CLOUD_LOCATION")
                or config.VERTEXAI_CONFIG["location"]
            )
            self.gcs_bucket = gcs_bucket or config.VERTEXAI_CONFIG["gcs_bucket"]
            self.api_key = None
            
            if not self.project:
                raise ValueError(
                    "Vertex AI requires a GCP project. Set via project parameter, "
                    "GOOGLE_CLOUD_PROJECT env var, or config.VERTEXAI_CONFIG['project']"
                )
            
            logger.info(f"Using Vertex AI backend: project={self.project}, location={self.location}")
        else:
            # Gemini Developer API configuration
            self.api_key = api_key or get_api_key()
            self.project = None
            self.location = None
            self.gcs_bucket = None
            logger.info("Using Gemini Developer API backend")
        
        self._client = None
        self._gcs_client = None

    @property
    def client(self) -> genai.Client:
        """Get or create Gemini client."""
        if self._client is None:
            if self.vertexai:
                # Vertex AI uses default API version (not v1alpha)
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location,
                )
            else:
                self._client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
        return self._client
    
    @property
    def gcs_client(self):
        """Get or create GCS client (Vertex AI only)."""
        if not self.vertexai:
            raise RuntimeError("GCS client is only available in Vertex AI mode")
        
        if not HAS_GCS:
            raise ImportError(
                "google-cloud-storage is required for Vertex AI support. "
                "Install it with: pip install gemini-batch[vertexai]"
            )
        
        if self._gcs_client is None:
            self._gcs_client = gcs_storage.Client(project=self.project)
        return self._gcs_client
    
    def get_gcs_bucket_name(self) -> str:
        """
        Get or generate GCS bucket name for this client.
        
        Returns:
            Bucket name string. If gcs_bucket was provided, returns that.
            Otherwise generates: gemini-batch-{project_id}
        """
        if not self.vertexai:
            raise RuntimeError("GCS bucket is only available in Vertex AI mode")
        
        if self.gcs_bucket:
            return self.gcs_bucket
        
        # Generate default bucket name
        return f"gemini-batch-{self.project}"
    
    def ensure_gcs_bucket(self, bucket_name: Optional[str] = None) -> str:
        """
        Ensure GCS bucket exists, creating it if necessary.
        
        Args:
            bucket_name: Bucket name to use. If None, uses get_gcs_bucket_name().
        
        Returns:
            Bucket name that was ensured to exist.
        """
        if not self.vertexai:
            raise RuntimeError("GCS bucket operations are only available in Vertex AI mode")
        
        bucket_name = bucket_name or self.get_gcs_bucket_name()
        gcs = self.gcs_client
        
        try:
            bucket = gcs.get_bucket(bucket_name)
            logger.debug(f"Using existing GCS bucket: {bucket_name}")
        except Exception:
            # Bucket doesn't exist, create it
            if not config.VERTEXAI_CONFIG.get("auto_create_bucket", True):
                raise ValueError(
                    f"GCS bucket '{bucket_name}' does not exist and auto_create_bucket is disabled. "
                    f"Create the bucket manually or enable auto_create_bucket in config."
                )
            
            bucket_location = config.VERTEXAI_CONFIG.get("bucket_location", "US")
            logger.info(f"Creating GCS bucket: {bucket_name} in location {bucket_location}")
            bucket = gcs.create_bucket(bucket_name, location=bucket_location)
            logger.info(f"Created GCS bucket: {bucket_name}")
        
        # Update client's bucket reference
        self.gcs_bucket = bucket_name
        return bucket_name


# ============================================================================
# GCS Utilities (for Vertex AI backend)
# ============================================================================

def upload_to_gcs(
    gemini_client: "GeminiClient",
    file: Union[Path, Image.Image, bytes, str],
    destination_blob_name: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        gemini_client: GeminiClient instance (must be in Vertex AI mode)
        file: File to upload - pathlib.Path, PIL.Image.Image, bytes, or str (file path)
        destination_blob_name: Name for the blob in GCS. Auto-generated if None.
        bucket_name: GCS bucket name. Uses client's bucket if None.
    
    Returns:
        Dictionary with 'uri' (gs:// URI) and 'mime_type'
    
    Raises:
        RuntimeError: If client is not in Vertex AI mode
        ImportError: If google-cloud-storage is not installed
    """
    if not gemini_client.vertexai:
        raise RuntimeError("upload_to_gcs requires Vertex AI mode")
    
    if not HAS_GCS:
        raise ImportError(
            "google-cloud-storage is required for Vertex AI support. "
            "Install it with: pip install gemini-batch[vertexai]"
        )
    
    # Ensure bucket exists
    bucket_name = gemini_client.ensure_gcs_bucket(bucket_name)
    gcs = gemini_client.gcs_client
    bucket = gcs.bucket(bucket_name)
    
    # Determine file content and MIME type
    if isinstance(file, str):
        file = Path(file)
    
    if isinstance(file, Path):
        if not file.exists():
            raise ValueError(f"File not found: {file}")
        
        mime_type, _ = mimetypes.guess_type(str(file))
        mime_type = mime_type or "application/octet-stream"
        
        if destination_blob_name is None:
            destination_blob_name = f"uploads/{uuid.uuid4()}_{file.name}"
        
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(str(file), content_type=mime_type)
        logger.debug(f"Uploaded {file} to gs://{bucket_name}/{destination_blob_name}")
        
    elif isinstance(file, Image.Image):
        mime_type = "image/png"
        
        if destination_blob_name is None:
            destination_blob_name = f"uploads/{uuid.uuid4()}.png"
        
        # Save image to bytes
        img_buffer = io.BytesIO()
        file.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(img_buffer, content_type=mime_type)
        logger.debug(f"Uploaded PIL Image to gs://{bucket_name}/{destination_blob_name}")
        
    elif isinstance(file, bytes):
        # Try to detect if it's an image
        mime_type = "application/octet-stream"
        suffix = ".bin"
        
        # Simple magic byte detection for common image formats
        if file[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = "image/png"
            suffix = ".png"
        elif file[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
            suffix = ".jpg"
        elif file[:6] in (b'GIF87a', b'GIF89a'):
            mime_type = "image/gif"
            suffix = ".gif"
        
        if destination_blob_name is None:
            destination_blob_name = f"uploads/{uuid.uuid4()}{suffix}"
        
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(file, content_type=mime_type)
        logger.debug(f"Uploaded bytes to gs://{bucket_name}/{destination_blob_name}")
        
    else:
        raise ValueError(
            f"Unsupported file type: {type(file)}. "
            f"Supported types: str, pathlib.Path, PIL.Image.Image, bytes"
        )
    
    return {
        "uri": f"gs://{bucket_name}/{destination_blob_name}",
        "mime_type": mime_type
    }


def download_from_gcs(
    gemini_client: "GeminiClient",
    gcs_uri: str,
    destination_path: Optional[str] = None,
) -> str:
    """
    Download a file from Google Cloud Storage.
    
    Args:
        gemini_client: GeminiClient instance (must be in Vertex AI mode)
        gcs_uri: GCS URI (gs://bucket/path/to/file)
        destination_path: Local path to save file. Auto-generated if None.
    
    Returns:
        Path to the downloaded file
    
    Raises:
        RuntimeError: If client is not in Vertex AI mode
        ValueError: If gcs_uri is invalid
    """
    if not gemini_client.vertexai:
        raise RuntimeError("download_from_gcs requires Vertex AI mode")
    
    if not HAS_GCS:
        raise ImportError(
            "google-cloud-storage is required for Vertex AI support. "
            "Install it with: pip install gemini-batch[vertexai]"
        )
    
    # Parse GCS URI
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}. Must start with 'gs://'")
    
    parts = gcs_uri[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}. Expected format: gs://bucket/path")
    
    bucket_name, blob_name = parts
    
    gcs = gemini_client.gcs_client
    bucket = gcs.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    if destination_path is None:
        # Create temp file with appropriate extension
        suffix = Path(blob_name).suffix or ".bin"
        fd, destination_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
    
    blob.download_to_filename(destination_path)
    logger.info(f"Downloaded {gcs_uri} to {destination_path}")
    
    return destination_path


def list_gcs_blobs(
    gemini_client: "GeminiClient",
    prefix: str,
    bucket_name: Optional[str] = None,
) -> List[str]:
    """
    List blobs in a GCS bucket with a given prefix.
    
    Args:
        gemini_client: GeminiClient instance (must be in Vertex AI mode)
        prefix: Prefix to filter blobs
        bucket_name: GCS bucket name. Uses client's bucket if None.
    
    Returns:
        List of gs:// URIs for matching blobs
    """
    if not gemini_client.vertexai:
        raise RuntimeError("list_gcs_blobs requires Vertex AI mode")
    
    if not HAS_GCS:
        raise ImportError(
            "google-cloud-storage is required for Vertex AI support. "
            "Install it with: pip install gemini-batch[vertexai]"
        )
    
    bucket_name = bucket_name or gemini_client.get_gcs_bucket_name()
    gcs = gemini_client.gcs_client
    bucket = gcs.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    return [f"gs://{bucket_name}/{blob.name}" for blob in blobs]


def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """
    Parse a GCS URI into bucket name and path.
    
    Args:
        gcs_uri: GCS URI (gs://bucket/path/to/object or gs://bucket/path/)
    
    Returns:
        Tuple of (bucket_name, path). Path may be empty if only bucket is specified.
    
    Raises:
        ValueError: If gcs_uri is not a valid GCS URI
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}. Must start with 'gs://'")
    
    # Remove gs:// prefix
    uri_without_prefix = gcs_uri[5:]
    
    # Split into bucket and path
    parts = uri_without_prefix.split("/", 1)
    bucket_name = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    
    if not bucket_name:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}. Bucket name is empty.")
    
    return bucket_name, path


def upload_file_for_batch(
    file: Union[Path, Image.Image, bytes],
    gemini_client: "GeminiClient",
) -> Dict[str, str]:
    """
    Upload a file for batch processing, using appropriate storage based on backend.
    
    Routes to:
    - GCS (gs:// URIs) for Vertex AI backend
    - File API (files/ URIs) for Gemini Developer API backend
    
    Args:
        file: File to upload - pathlib.Path, PIL.Image.Image, or bytes
        gemini_client: GeminiClient instance
    
    Returns:
        Dictionary with 'uri' and 'mime_type' keys
    """
    if gemini_client.vertexai:
        return upload_to_gcs(gemini_client, file)
    else:
        return upload_file_to_gemini(file, gemini_client.client)


def upload_file_to_gemini(
    file: Union[Path, Image.Image, bytes],
    client: genai.Client
) -> Dict[str, str]:
    """
    Upload a file to Gemini File API and return URI and MIME type.

    Args:
        file: File to upload - pathlib.Path (file path), PIL.Image.Image, or bytes
        client: Gemini client instance

    Returns:
        Dictionary with 'uri' and 'mime_type' keys

    Raises:
        ValueError: If file type is unsupported or file not found
    """
    # Handle different input types
    if isinstance(file, Path):
        # File path - upload directly
        file_path = str(file)
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        logger.debug(f"Uploading file: {file_path}")
        uploaded_file = client.files.upload(file=file_path)
        logger.debug(f"Uploaded file: {uploaded_file.name} with MIME type: {uploaded_file.mime_type}")

        return {
            "uri": uploaded_file.uri,
            "mime_type": uploaded_file.mime_type
        }

    elif isinstance(file, Image.Image):
        # PIL Image - save to temp file and upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            file.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            logger.debug(f"Uploading PIL Image as PNG")
            uploaded_file = client.files.upload(file=tmp_path)
            logger.debug(f"Uploaded image: {uploaded_file.name} with MIME type: {uploaded_file.mime_type}")

            return {
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type
            }
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    elif isinstance(file, bytes):
        # Raw bytes - save to temp file and upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file)
            tmp_path = tmp.name

        try:
            logger.debug(f"Uploading bytes as PNG")
            uploaded_file = client.files.upload(file=tmp_path)
            logger.debug(f"Uploaded bytes: {uploaded_file.name} with MIME type: {uploaded_file.mime_type}")

            return {
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type
            }
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    else:
        raise ValueError(
            f"Unsupported file type: {type(file)}. "
            f"Supported types: pathlib.Path (file path), PIL.Image.Image, bytes"
        )


# ============================================================================
# Async Upload Functions (for parallel file uploads)
# ============================================================================

async def upload_file_to_gemini_async(
    file: Union[Path, Image.Image, bytes],
    client: genai.Client
) -> Dict[str, str]:
    """
    Async upload a file to Gemini File API using client.aio.files.upload().

    Args:
        file: File to upload - pathlib.Path (file path), PIL.Image.Image, or bytes
        client: Gemini client instance

    Returns:
        Dictionary with 'uri' and 'mime_type' keys

    Raises:
        ValueError: If file type is unsupported or file not found
    """
    if isinstance(file, Path):
        file_path = str(file)
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        logger.debug(f"Async uploading file: {file_path}")
        uploaded_file = await client.aio.files.upload(file=file_path)
        logger.debug(f"Uploaded file: {uploaded_file.name} with MIME type: {uploaded_file.mime_type}")

        if not uploaded_file.uri or not uploaded_file.mime_type:
            raise ValueError(f"Upload failed: missing uri or mime_type for {file_path}")

        return {
            "uri": uploaded_file.uri,
            "mime_type": uploaded_file.mime_type
        }

    elif isinstance(file, Image.Image):
        # PIL Image - save to temp file and upload
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            file.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            logger.debug("Async uploading PIL Image as PNG")
            uploaded_file = await client.aio.files.upload(file=tmp_path)
            logger.debug(f"Uploaded image: {uploaded_file.name} with MIME type: {uploaded_file.mime_type}")

            if not uploaded_file.uri or not uploaded_file.mime_type:
                raise ValueError("Upload failed: missing uri or mime_type for PIL Image")

            return {
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type
            }
        finally:
            os.unlink(tmp_path)

    elif isinstance(file, bytes):
        # Raw bytes - save to temp file and upload
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file)
            tmp_path = tmp.name

        try:
            logger.debug("Async uploading bytes as PNG")
            uploaded_file = await client.aio.files.upload(file=tmp_path)
            logger.debug(f"Uploaded bytes: {uploaded_file.name} with MIME type: {uploaded_file.mime_type}")

            if not uploaded_file.uri or not uploaded_file.mime_type:
                raise ValueError("Upload failed: missing uri or mime_type for bytes")

            return {
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type
            }
        finally:
            os.unlink(tmp_path)

    else:
        raise ValueError(
            f"Unsupported file type: {type(file)}. "
            f"Supported types: pathlib.Path (file path), PIL.Image.Image, bytes"
        )


async def upload_to_gcs_async(
    gemini_client: "GeminiClient",
    file: Union[Path, Image.Image, bytes, str],
    destination_blob_name: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Async upload a file to Google Cloud Storage using gcloud-aio-storage.

    Args:
        gemini_client: GeminiClient instance (must be in Vertex AI mode)
        file: File to upload - pathlib.Path, PIL.Image.Image, bytes, or str (file path)
        destination_blob_name: Name for the blob in GCS. Auto-generated if None.
        bucket_name: GCS bucket name. Uses client's bucket if None.

    Returns:
        Dictionary with 'uri' (gs:// URI) and 'mime_type'

    Raises:
        RuntimeError: If client is not in Vertex AI mode
        ImportError: If gcloud-aio-storage is not installed
    """
    if not gemini_client.vertexai:
        raise RuntimeError("upload_to_gcs_async requires Vertex AI mode")

    if not HAS_ASYNC_GCS:
        raise ImportError(
            "gcloud-aio-storage is required for async Vertex AI uploads. "
            "Install it with: pip install gemini-batch[vertexai]"
        )

    # Ensure bucket exists (sync call, but only happens once per client)
    bucket_name = gemini_client.ensure_gcs_bucket(bucket_name)

    # Determine file content and MIME type
    if isinstance(file, str):
        file = Path(file)

    if isinstance(file, Path):
        if not file.exists():
            raise ValueError(f"File not found: {file}")

        mime_type, _ = mimetypes.guess_type(str(file))
        mime_type = mime_type or "application/octet-stream"

        if destination_blob_name is None:
            destination_blob_name = f"uploads/{uuid.uuid4()}_{file.name}"

        # Read file content
        with open(file, 'rb') as f:
            file_data = f.read()

    elif isinstance(file, Image.Image):
        mime_type = "image/png"

        if destination_blob_name is None:
            destination_blob_name = f"uploads/{uuid.uuid4()}.png"

        # Save image to bytes
        img_buffer = io.BytesIO()
        file.save(img_buffer, format="PNG")
        file_data = img_buffer.getvalue()

    elif isinstance(file, bytes):
        # Try to detect if it's an image
        mime_type = "application/octet-stream"
        suffix = ".bin"

        # Simple magic byte detection for common image formats
        if file[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = "image/png"
            suffix = ".png"
        elif file[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
            suffix = ".jpg"
        elif file[:6] in (b'GIF87a', b'GIF89a'):
            mime_type = "image/gif"
            suffix = ".gif"

        if destination_blob_name is None:
            destination_blob_name = f"uploads/{uuid.uuid4()}{suffix}"

        file_data = file

    else:
        raise ValueError(
            f"Unsupported file type: {type(file)}. "
            f"Supported types: str, pathlib.Path, PIL.Image.Image, bytes"
        )

    # Upload using async GCS client
    upload_timeout = config.VERTEXAI_CONFIG.get("upload_timeout", 300)
    async with AsyncGCSStorage() as gcs:
        await gcs.upload(
            bucket_name,
            destination_blob_name,
            file_data,
            content_type=mime_type,
            timeout=upload_timeout,
        )

    logger.debug(f"Async uploaded to gs://{bucket_name}/{destination_blob_name}")

    return {
        "uri": f"gs://{bucket_name}/{destination_blob_name}",
        "mime_type": mime_type
    }


async def upload_files_parallel(
    files: List[Tuple[int, int, Union[Path, Image.Image, bytes]]],
    gemini_client: "GeminiClient",
    max_concurrent: int = 10,
    show_progress: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Dict[Tuple[int, int], Dict[str, str]]:
    """
    Upload multiple files in parallel using asyncio.gather().

    Args:
        files: List of tuples (prompt_idx, part_idx, file) where file is Path, PIL Image, or bytes
        gemini_client: GeminiClient instance
        max_concurrent: Maximum number of concurrent uploads (default: 10)
        show_progress: Whether to show a progress bar (default: True)
        max_retries: Maximum number of retry attempts for transient errors (default: 3)
        retry_delay: Base delay in seconds between retries, doubles each attempt (default: 1.0)

    Returns:
        Dictionary mapping (prompt_idx, part_idx) to {"uri": ..., "mime_type": ...}

    Raises:
        Exception: If any upload fails after all retries (fail-fast behavior)
    """
    import asyncio
    from tqdm.asyncio import tqdm_asyncio

    if not files:
        return {}

    semaphore = asyncio.Semaphore(max_concurrent)

    async def upload_with_semaphore(
        prompt_idx: int,
        part_idx: int,
        file: Union[Path, Image.Image, bytes]
    ) -> Tuple[Tuple[int, int], Dict[str, str]]:
        async with semaphore:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    if gemini_client.vertexai:
                        result = await upload_to_gcs_async(gemini_client, file)
                    else:
                        result = await upload_file_to_gemini_async(file, gemini_client.client)
                    return ((prompt_idx, part_idx), result)
                except (ConnectionResetError, OSError) as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Upload failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Upload failed after {max_retries + 1} attempts: {e}"
                        )
                        raise
                except Exception as e:
                    # Check if it's a retryable aiohttp/network error
                    error_name = type(e).__name__
                    error_msg = str(e).lower()
                    is_retryable = (
                        # aiohttp client errors
                        "Client" in error_name and ("OS" in error_name or "Connector" in error_name)
                        # Network-related error messages
                        or "connection" in error_msg
                        or "connect" in error_msg
                        or "network" in error_msg
                        or "reset" in error_msg
                        or "timeout" in error_msg
                    )
                    if is_retryable:
                        last_error = e
                        if attempt < max_retries:
                            delay = retry_delay * (2 ** attempt)
                            logger.warning(
                                f"Upload failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                                f"Retrying in {delay:.1f}s..."
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                f"Upload failed after {max_retries + 1} attempts: {e}"
                            )
                            raise
                    else:
                        # Non-retryable error
                        raise
            # Should never reach here, but just in case
            raise last_error

    # Create upload tasks
    tasks = [
        upload_with_semaphore(prompt_idx, part_idx, file)
        for prompt_idx, part_idx, file in files
    ]

    logger.info(f"Uploading {len(files)} files in parallel (max_concurrent={max_concurrent})")

    # Run all uploads concurrently with optional progress bar
    if show_progress:
        results = await tqdm_asyncio.gather(
            *tasks,
            desc="Uploading files",
            unit="file"
        )
    else:
        results = await asyncio.gather(*tasks)

    logger.info(f"Completed uploading {len(files)} files")

    # Convert list of tuples to dictionary
    return dict(results)


def _convert_page_to_image(page: fitz.Page, doc: fitz.Document, dpi: int) -> Image.Image:
    """
    Converts a single PDF page to a PIL Image.
    Tries to extract an embedded image first, otherwise renders the page.
    """
    img = None
    img_list = page.get_images(full=True)
    
    if img_list:
        largest_img_xref = -1
        max_area = 0
        for img_info in img_list:
            width, height = img_info[2], img_info[3]
            area = width * height
            if area > max_area:
                max_area = area
                largest_img_xref = img_info[0]

        if largest_img_xref != -1:
            base_image = doc.extract_image(largest_img_xref)
            if base_image and "image" in base_image:
                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes))
                logger.info(f"Extracting largest embedded image from page {page.number + 1}. Image size: {img.width}x{img.height}")

    if img is None:
        logger.info(f"No embedded image found on page {page.number + 1}. Rendering page with DPI={dpi}.")
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        
    return img


def pdf_pages_to_images(
    pdf_path: str,
    dpi: int = config.IMAGE_PROCESSING_CONFIG["default_dpi"],
    max_pages: Optional[int] = None,
    start_page: int = 1,
    grayscale: bool = False,
    display: bool = False,
    crop_sides: int = 0
) -> List[Image.Image]:
    """
    Convert PDF pages to a list of PIL Images.

    Args:
        pdf_path: Path to PDF (scanned pages).
        dpi: Render resolution for pages that are not image-based.
        max_pages: Optional cap on number of pages.
        start_page: Page index to start from (1-based).
        grayscale: Convert to grayscale to reduce tokens.
        display: Whether to display the images using matplotlib.
        crop_sides: Number of pixels to crop from left and right sides.
    Returns:
        List of PIL Image objects in page order.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    images: List[Image.Image] = []
    start_page_idx = max(0, start_page - 1)
    end_page_idx = doc.page_count if max_pages is None else min(doc.page_count, start_page_idx + max_pages)
    
    logger.info(f"Converting pages {start_page_idx + 1} to {end_page_idx} to images (dpi for rendering fallback={dpi})")
    
    for page_index in range(start_page_idx, end_page_idx):
        logger.info(f"Processing page {page_index + 1} of {doc.page_count}")
        page = doc.load_page(page_index)
        img = _convert_page_to_image(page, doc, dpi)

        if grayscale:
            img = img.convert("L")
        if crop_sides > 0:
            left = crop_sides
            upper = 0
            right = img.width - crop_sides
            lower = img.height
            if right > left:
                img = img.crop((left, upper, right, lower))
        
        images.append(img)
        
        if display:
            _show_image_popup(img, page_index + 1)
            
    logger.info(f"Converted {len(images)} pages to images.")
    doc.close()
    
    return images



def _show_image_popup(img: Image.Image, page_number: int) -> None:
    """
    Show a single image as a popup window and block until closed, at original resolution.
    Args:
        img: PIL Image to display.
        page_number: Page number for the window title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for displaying images. "
            "Install it with: pip install matplotlib"
        )

    dpi = 100  # Use a standard DPI for display
    width, height = img.size
    figsize = (width / dpi, height / dpi)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
    plt.title(f'Page {page_number}')
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)


def build_generation_config(
    response_schema: Optional[Union[types.Schema, Type[BaseModel]]] = None,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    media_resolution: Optional[str] = None,
    model: Optional[str] = None,
) -> types.GenerateContentConfig:
    """
    Builds a generation configuration for the Gemini API.

    Args:
        response_schema: The schema for the response (either types.Schema or Pydantic BaseModel class).
        thinking_budget: The thinking budget (int, for Flash models). Cannot be used with thinking_level.
            Note: For gemini-3* models, thinking_budget=0 is automatically converted to
            thinking_level="MINIMAL" as these models don't support thinking_budget.
        thinking_level: The thinking level (str: "low" or "high", for Gemini 3.0 Pro). Cannot be used with thinking_budget.
        temperature: The temperature.
        top_p: The top-p value.
        top_k: The top-k value.
        max_output_tokens: The maximum number of output tokens.
        media_resolution: The media resolution for image/video inputs. Valid values:
            - "MEDIA_RESOLUTION_LOW": Lower token usage, faster/cheaper, less detail (70-280 tokens/frame)
            - "MEDIA_RESOLUTION_MEDIUM": Balanced detail, cost, and speed (560 tokens for images)
            - "MEDIA_RESOLUTION_HIGH": Higher token usage, more detail, increased latency/cost (1120 tokens for images)
            Controls the maximum number of tokens allocated for media inputs.
        model: The model name (used for automatic thinking config conversion for gemini-3* models).

    Returns:
        A types.GenerateContentConfig object representing the generation configuration.

    Raises:
        ValueError: If both thinking_budget and thinking_level are provided.
    """
    gen_config_dict = {
        "temperature": temperature if temperature is not None else config.MODEL_CONFIG["generation_config"]["temperature"],
        "max_output_tokens": max_output_tokens if max_output_tokens is not None else config.MODEL_CONFIG["generation_config"]["max_output_tokens"],
    }

    if response_schema is not None:
        gen_config_dict["response_mime_type"] = "application/json"
        gen_config_dict["response_json_schema"] = _transformers.t_schema(client=None, origin=response_schema)

    # Auto-convert thinking_budget=0 to thinking_level="MINIMAL" for gemini-3* models
    is_gemini_3_model = model is not None and model.startswith("gemini-3")
    if is_gemini_3_model and thinking_budget == 0 and thinking_level is None:
        print(f"[gemini-batch] Note: thinking_budget=0 auto-converted to thinking_level='MINIMAL' for {model}")
        thinking_level = "MINIMAL"
        thinking_budget = None

    # Validate that only one thinking parameter is provided
    if thinking_budget is not None and thinking_level is not None:
        raise ValueError("Cannot specify both thinking_budget and thinking_level. Use thinking_budget for Flash models or thinking_level for Gemini 3.0 Pro.")

    if thinking_budget is not None:
        include_thoughts = False if thinking_budget == 0 else None
        gen_config_dict["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget, include_thoughts=include_thoughts)
    elif thinking_level is not None:
        gen_config_dict["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

    if top_p is not None:
        gen_config_dict["top_p"] = top_p
    if top_k is not None:
        gen_config_dict["top_k"] = top_k

    if media_resolution is not None:
        gen_config_dict["media_resolution"] = media_resolution

    return types.GenerateContentConfig(**gen_config_dict)


def calculate_token_statistics(
    metadata_list: List[Union[Dict[str, Any], None]],
    verbose: bool = False
) -> TokenStatistics:
    """
    Calculate token usage statistics from batch processing metadata.

    This function aggregates token counts from the metadata returned by
    batch_process(return_metadata=True), providing both total and average
    statistics across all requests.

    Args:
        metadata_list: List of metadata dictionaries from batch_process(), where
            each entry is either a dict with 'usageMetadata' or None (for failed requests).
            Structure: {'usageMetadata': {'totalTokenCount': int, ...}, 'modelVersion': str}
        verbose: If True, prints a summary of totals and averages per prompt to stdout.

    Returns:
        TokenStatistics object containing aggregated token usage statistics including:
        - Request counts (total, successful, failed)
        - Total token counts across all successful requests
        - Average tokens per successful request

    Examples:
        >>> from gemini_batch import batch_process, calculate_token_statistics
        >>> results, metadata = batch_process(prompts, schema, return_metadata=True)
        >>> stats = calculate_token_statistics(metadata)
        >>> print(f"Total tokens: {stats.total_tokens}")
        >>> print(f"Average per request: {stats.avg_total_tokens}")
        >>> print(f"Success rate: {stats.successful_requests}/{stats.total_requests}")
        >>>
        >>> # With verbose output
        >>> stats = calculate_token_statistics(metadata, verbose=True)
        # Prints summary table with totals and averages
    """
    # Initialize counters
    total_requests = len(metadata_list)
    successful_requests = 0
    failed_requests = 0

    # Initialize token totals
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_tokens = 0
    total_cached_tokens = 0
    total_thoughts_tokens = 0

    # Single-pass iteration through metadata
    for metadata in metadata_list:
        # Check if entry is None or missing usageMetadata
        if metadata is None:
            failed_requests += 1
            continue

        if 'usageMetadata' not in metadata:
            failed_requests += 1
            logger.warning("Metadata entry missing 'usageMetadata' field, treating as failed request")
            continue

        # Entry is successful, extract usage data
        successful_requests += 1
        usage = metadata['usageMetadata']

        # Accumulate token counts (handle None values by treating as 0)
        total_prompt_tokens += (usage.get('promptTokenCount') or 0)
        total_candidates_tokens += (usage.get('candidatesTokenCount') or 0)
        total_tokens += (usage.get('totalTokenCount') or 0)
        total_cached_tokens += (usage.get('cachedContentTokenCount') or 0)
        total_thoughts_tokens += (usage.get('thoughtsTokenCount') or 0)

    # Calculate averages (None if no successful requests)
    if successful_requests > 0:
        avg_prompt_tokens = total_prompt_tokens / successful_requests
        avg_candidates_tokens = total_candidates_tokens / successful_requests
        avg_total_tokens = total_tokens / successful_requests
        avg_cached_tokens = total_cached_tokens / successful_requests
        avg_thoughts_tokens = total_thoughts_tokens / successful_requests
    else:
        avg_prompt_tokens = None
        avg_candidates_tokens = None
        avg_total_tokens = None
        avg_cached_tokens = None
        avg_thoughts_tokens = None

    # Print verbose output if requested
    if verbose:
        print("\n" + "=" * 60)
        print("TOKEN STATISTICS SUMMARY")
        print("=" * 60)
        print(f"Requests: {successful_requests}/{total_requests} successful ({failed_requests} failed)")
        print("-" * 60)
        print(f"{'Metric':<25} {'Total':>15} {'Avg/Prompt':>15}")
        print("-" * 60)
        print(f"{'Prompt Tokens':<25} {total_prompt_tokens:>15,} {avg_prompt_tokens:>15,.1f}" if avg_prompt_tokens else f"{'Prompt Tokens':<25} {total_prompt_tokens:>15,} {'N/A':>15}")
        print(f"{'Cached Tokens':<25} {total_cached_tokens:>15,} {avg_cached_tokens:>15,.1f}" if avg_cached_tokens else f"{'Cached Tokens':<25} {total_cached_tokens:>15,} {'N/A':>15}")
        print(f"{'Output Tokens':<25} {total_candidates_tokens:>15,} {avg_candidates_tokens:>15,.1f}" if avg_candidates_tokens else f"{'Output Tokens':<25} {total_candidates_tokens:>15,} {'N/A':>15}")
        print(f"{'Reason Tokens':<25} {total_thoughts_tokens:>15,} {avg_thoughts_tokens:>15,.1f}" if avg_thoughts_tokens else f"{'Reason Tokens':<25} {total_thoughts_tokens:>15,} {'N/A':>15}")
        print("-" * 60)
        print(f"{'TOTAL TOKENS':<25} {total_tokens:>15,} {avg_total_tokens:>15,.1f}" if avg_total_tokens else f"{'TOTAL TOKENS':<25} {total_tokens:>15,} {'N/A':>15}")
        print("=" * 60 + "\n")

    return TokenStatistics(
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        total_prompt_tokens=total_prompt_tokens,
        total_candidates_tokens=total_candidates_tokens,
        total_tokens=total_tokens,
        total_cached_tokens=total_cached_tokens,
        total_thoughts_tokens=total_thoughts_tokens,
        avg_prompt_tokens=avg_prompt_tokens,
        avg_candidates_tokens=avg_candidates_tokens,
        avg_total_tokens=avg_total_tokens,
        avg_cached_tokens=avg_cached_tokens,
        avg_thoughts_tokens=avg_thoughts_tokens,
    )
