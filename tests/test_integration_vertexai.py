"""
Integration tests with live Vertex AI API.

Tests the Vertex AI backend for batch processing, which uses Google Cloud Storage
instead of the Gemini File API.

Prerequisites:
1. Google Cloud project with Vertex AI API enabled
2. Application Default Credentials configured: gcloud auth application-default login
3. Environment variables set:
   - GOOGLE_CLOUD_PROJECT: Your GCP project ID
   - Optionally: VERTEX_AI_GCS_BUCKET for a specific bucket (auto-created otherwise)

Run with: pytest tests/test_integration_vertexai.py -v -s -m integration
"""
import pytest
import os
from pydantic import BaseModel
from PIL import Image, ImageDraw
import io
from typing import List

from gemini_batch.utils import HAS_GCS

# Skip all tests in this module if prerequisites not met
SKIP_REASON = None
if not os.getenv("GOOGLE_CLOUD_PROJECT"):
    SKIP_REASON = "GOOGLE_CLOUD_PROJECT not set - skipping Vertex AI integration tests"
elif not HAS_GCS:
    SKIP_REASON = "google-cloud-storage not installed - run: pip install gemini-batch[vertexai]"
else:
    # Check if credentials are available
    try:
        from google.auth import default
        default()
    except Exception:
        SKIP_REASON = "Google Cloud credentials not configured - run: gcloud auth application-default login"

# Skip all tests if prerequisites not met
if SKIP_REASON:
    pytest.skip(SKIP_REASON, allow_module_level=True)

# Now import the modules that require credentials
from gemini_batch import batch_process, GeminiClient
from gemini_batch.utils import (
    upload_to_gcs,
    download_from_gcs,
    list_gcs_blobs,
    upload_file_for_batch,
)

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def vertexai_client():
    """Create a Vertex AI client for testing."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    bucket = os.getenv("VERTEX_AI_GCS_BUCKET")  # Optional
    
    client = GeminiClient(
        vertexai=True,
        project=project,
        location="us-central1",
        gcs_bucket=bucket,
    )
    return client


@pytest.fixture
def model():
    """Use the cheapest model for testing that supports batch on Vertex AI."""
    return "gemini-2.5-flash-lite"


# =============================================================================
# GCS Utility Tests
# =============================================================================

class TestGCSUtilities:
    """Test GCS upload/download utilities for Vertex AI."""

    def test_upload_text_file_to_gcs(self, vertexai_client, tmp_path):
        """Test uploading a text file to GCS."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello from Vertex AI integration test!")
        
        result = upload_to_gcs(
            vertexai_client,
            test_file,
            destination_blob_name="integration-tests/test_upload.txt",
        )
        
        assert "uri" in result
        assert result["uri"].startswith("gs://")
        assert "mime_type" in result
        assert result["mime_type"] == "text/plain"
        
        print(f"\n✓ Uploaded text file to: {result['uri']}")

    def test_upload_image_bytes_to_gcs(self, vertexai_client):
        """Test uploading image bytes to GCS."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()
        
        result = upload_to_gcs(
            vertexai_client,
            img_bytes,
        )
        
        assert "uri" in result
        assert result["uri"].startswith("gs://")
        assert result["mime_type"] == "image/png"
        
        print(f"\n✓ Uploaded image bytes to: {result['uri']}")

    def test_upload_pil_image_to_gcs(self, vertexai_client):
        """Test uploading a PIL Image to GCS."""
        img = Image.new('RGB', (100, 100), color='blue')
        
        result = upload_to_gcs(
            vertexai_client,
            img,
            destination_blob_name="integration-tests/test_pil_image.png",
        )
        
        assert "uri" in result
        assert result["uri"].startswith("gs://")
        assert result["mime_type"] == "image/png"
        
        print(f"\n✓ Uploaded PIL Image to: {result['uri']}")

    def test_download_from_gcs(self, vertexai_client, tmp_path):
        """Test downloading a file from GCS."""
        # First upload a file
        test_content = "Download test content"
        test_file = tmp_path / "upload_for_download.txt"
        test_file.write_text(test_content)
        
        upload_result = upload_to_gcs(
            vertexai_client,
            test_file,
            destination_blob_name="integration-tests/download_test.txt",
        )
        
        # Now download it
        download_path = str(tmp_path / "downloaded.txt")
        result_path = download_from_gcs(
            vertexai_client,
            upload_result["uri"],
            download_path,
        )
        
        assert os.path.exists(result_path)
        with open(result_path, 'r') as f:
            assert f.read() == test_content
        
        print(f"\n✓ Downloaded file from GCS to: {result_path}")

    def test_list_gcs_blobs(self, vertexai_client):
        """Test listing blobs in GCS bucket."""
        # First ensure there's something to list
        img = Image.new('RGB', (50, 50), color='green')
        upload_to_gcs(
            vertexai_client,
            img,
            destination_blob_name="integration-tests/list_test/image1.png",
        )
        
        blobs = list_gcs_blobs(
            vertexai_client,
            prefix="integration-tests/list_test/",
        )
        
        assert len(blobs) >= 1
        assert all(uri.startswith("gs://") for uri in blobs)
        
        print(f"\n✓ Listed {len(blobs)} blobs with prefix 'integration-tests/list_test/'")

    def test_upload_file_for_batch_routes_to_gcs(self, vertexai_client):
        """Test that upload_file_for_batch routes to GCS for Vertex AI client."""
        img = Image.new('RGB', (100, 100), color='yellow')
        
        result = upload_file_for_batch(img, vertexai_client)
        
        # Should return gs:// URI for Vertex AI
        assert result["uri"].startswith("gs://")
        assert result["mime_type"] == "image/png"
        
        print(f"\n✓ upload_file_for_batch correctly routed to GCS: {result['uri']}")


# =============================================================================
# Vertex AI Batch Processing Tests
# =============================================================================

class TestVertexAIBatchProcessing:
    """Test batch processing with Vertex AI backend."""

    def test_text_to_simple_structured_vertexai(self, model):
        """Test basic batch processing with Vertex AI backend."""
        
        class Answer(BaseModel):
            result: str

        prompts = [
            ["What is 2+2? Answer with just the number."],
            ["What is 3+3? Answer with just the number."],
        ]

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        bucket = os.getenv("VERTEX_AI_GCS_BUCKET")

        results = batch_process(
            prompts=prompts,
            schema=Answer,
            model=model,
            wait=True,
            temperature=0.1,
            vertexai=True,
            project=project,
            gcs_bucket=bucket,
        )

        assert len(results) == 2
        assert isinstance(results[0], Answer)
        assert isinstance(results[1], Answer)
        assert "4" in results[0].result
        assert "6" in results[1].result

        print(f"\n✓ Vertex AI Text→Structured: Got {results[0].result} and {results[1].result}")

    def test_text_to_rich_structured_vertexai(self, model):
        """Test Vertex AI batch processing with multi-field structured output."""

        class MathAnswer(BaseModel):
            answer: int
            explanation: str

        prompts = [
            ["What is 5+5? Answer and explain."],
            ["What is 10+10? Answer and explain."],
        ]

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        bucket = os.getenv("VERTEX_AI_GCS_BUCKET")

        results = batch_process(
            prompts=prompts,
            schema=MathAnswer,
            model=model,
            wait=True,
            temperature=0.1,
            vertexai=True,
            project=project,
            gcs_bucket=bucket,
        )

        assert len(results) == 2
        assert isinstance(results[0], MathAnswer)
        assert isinstance(results[1], MathAnswer)
        assert results[0].answer == 10
        assert results[1].answer == 20
        assert len(results[0].explanation) > 0

        print(f"\n✓ Vertex AI Rich Structured: {results[0].model_dump()}")

    def test_multimodal_vertexai(self, model):
        """Test Vertex AI batch processing with image input."""

        class ImageContent(BaseModel):
            primary_text: str
            description: str

        def create_test_image(text: str) -> bytes:
            img = Image.new('RGB', (300, 150), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((50, 60), text, fill='black')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

        prompts = [
            ["Read the text in this image:", create_test_image("HELLO")],
            ["Read the text in this image:", create_test_image("WORLD")],
        ]

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        bucket = os.getenv("VERTEX_AI_GCS_BUCKET")

        results = batch_process(
            prompts=prompts,
            schema=ImageContent,
            model=model,
            wait=True,
            temperature=0.1,
            vertexai=True,
            project=project,
            gcs_bucket=bucket,
        )

        assert len(results) == 2
        assert isinstance(results[0], ImageContent)
        assert isinstance(results[1], ImageContent)
        assert "hello" in results[0].primary_text.lower()
        assert "world" in results[1].primary_text.lower()

        print(f"\n✓ Vertex AI Multimodal: {results[0].model_dump()}")

    def test_vertexai_with_metadata(self, model):
        """Test Vertex AI batch processing with metadata return."""

        class SimpleAnswer(BaseModel):
            answer: str

        prompts = [
            ["What is 1+1?"],
            ["What is 2+2?"],
        ]

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        bucket = os.getenv("VERTEX_AI_GCS_BUCKET")

        results, metadata = batch_process(
            prompts=prompts,
            schema=SimpleAnswer,
            model=model,
            wait=True,
            temperature=0.1,
            vertexai=True,
            project=project,
            gcs_bucket=bucket,
            return_metadata=True,
        )

        assert len(results) == 2
        assert len(metadata) == 2
        
        # Check metadata structure
        for i, meta in enumerate(metadata):
            assert meta is not None, f"Metadata for result {i} should not be None"
            assert 'usageMetadata' in meta, f"Metadata {i} should contain usageMetadata"
            usage = meta['usageMetadata']
            assert 'totalTokenCount' in usage or usage.get('totalTokenCount') is not None

        print(f"\n✓ Vertex AI with metadata: {metadata[0]}")

    def test_vertexai_raw_text_mode(self, model):
        """Test Vertex AI batch processing without schema (raw text output)."""

        prompts = [
            ["Say 'hello' and nothing else."],
            ["Say 'world' and nothing else."],
        ]

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        bucket = os.getenv("VERTEX_AI_GCS_BUCKET")

        results = batch_process(
            prompts=prompts,
            schema=None,  # Raw text mode
            model=model,
            wait=True,
            temperature=0.1,
            vertexai=True,
            project=project,
            gcs_bucket=bucket,
        )

        assert len(results) == 2
        assert isinstance(results[0], str)
        assert isinstance(results[1], str)
        assert "hello" in results[0].lower()
        assert "world" in results[1].lower()

        print(f"\n✓ Vertex AI raw text: '{results[0]}' and '{results[1]}'")


# =============================================================================
# Environment Auto-Detection Tests
# =============================================================================

class TestEnvironmentAutoDetection:
    """Test that Vertex AI can be enabled via environment variables."""

    def test_vertexai_env_var_detection(self, model):
        """Test that GOOGLE_GENAI_USE_VERTEXAI=true enables Vertex AI mode."""
        # Save current env state
        original_env = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
        
        try:
            # Enable via env var
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            
            # Create client without explicit vertexai=True
            client = GeminiClient()
            
            assert client.vertexai is True, "Client should be in Vertex AI mode"
            assert client.project is not None, "Project should be set"
            
            print(f"\n✓ Auto-detected Vertex AI from env var: project={client.project}")
            
        finally:
            # Restore original env state
            if original_env is None:
                os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
            else:
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = original_env

    def test_vertexai_explicit_override(self):
        """Test that explicit vertexai=False overrides env var."""
        # Save current env state
        original_env = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
        original_api_key = os.environ.get("GEMINI_API_KEY")
        
        try:
            # Set env var to enable Vertex AI
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            
            # But explicitly disable it and provide API key
            if original_api_key:
                client = GeminiClient(vertexai=False)
                assert client.vertexai is False, "Explicit vertexai=False should override env var"
                print("\n✓ Explicit vertexai=False correctly overrides env var")
            else:
                pytest.skip("GEMINI_API_KEY not set, cannot test override")
                
        finally:
            # Restore original env state
            if original_env is None:
                os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
            else:
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = original_env


# =============================================================================
# GCS Bucket Auto-Creation Tests
# =============================================================================

class TestGCSBucketManagement:
    """Test GCS bucket auto-creation and management."""

    def test_bucket_auto_creation(self):
        """Test that GCS bucket is auto-created if it doesn't exist."""
        import uuid
        
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        # Use a unique bucket name to test creation
        test_bucket = f"gemini-batch-test-{uuid.uuid4().hex[:8]}"
        
        client = GeminiClient(
            vertexai=True,
            project=project,
            location="us-central1",
            gcs_bucket=test_bucket,
        )
        
        # This should auto-create the bucket
        bucket_name = client.ensure_gcs_bucket()
        
        assert bucket_name == test_bucket
        
        # Verify bucket exists by listing (should not raise)
        gcs = client.gcs_client
        bucket = gcs.bucket(bucket_name)
        assert bucket.exists()
        
        print(f"\n✓ Auto-created bucket: {bucket_name}")
        
        # Cleanup: delete the test bucket
        try:
            bucket.delete(force=True)
            print(f"  Cleaned up test bucket: {bucket_name}")
        except Exception as e:
            print(f"  Warning: Could not delete test bucket: {e}")

    def test_default_bucket_name_generation(self):
        """Test default bucket name generation based on project."""
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        client = GeminiClient(
            vertexai=True,
            project=project,
            location="us-central1",
            gcs_bucket=None,  # No bucket specified
        )
        
        bucket_name = client.get_gcs_bucket_name()
        
        assert bucket_name == f"gemini-batch-{project}"
        
        print(f"\n✓ Generated default bucket name: {bucket_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
