"""
Unit tests for Vertex AI utilities.

These tests mock GCS and Vertex AI interactions to test the logic
without requiring live API access.

Run with: pytest tests/test_vertexai.py -v
"""
import pytest
import os
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
from PIL import Image
import io

from gemini_batch import GeminiClient
from gemini_batch.utils import HAS_GCS


class TestGeminiClientVertexAI:
    """Test GeminiClient Vertex AI mode configuration."""

    def test_vertexai_mode_explicit(self):
        """Test explicit Vertex AI mode configuration."""
        with patch.dict(os.environ, {
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GEMINI_API_KEY": "",  # Clear API key
        }, clear=False):
            client = GeminiClient(
                vertexai=True,
                project="my-project",
                location="us-west1",
                gcs_bucket="my-bucket",
            )
            
            assert client.vertexai is True
            assert client.project == "my-project"
            assert client.location == "us-west1"
            assert client.gcs_bucket == "my-bucket"
            assert client.api_key is None

    def test_vertexai_mode_from_env(self):
        """Test Vertex AI mode auto-detection from environment variable."""
        with patch.dict(os.environ, {
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_CLOUD_PROJECT": "env-project",
            "GOOGLE_CLOUD_LOCATION": "europe-west1",
        }, clear=False):
            client = GeminiClient()
            
            assert client.vertexai is True
            assert client.project == "env-project"
            assert client.location == "europe-west1"

    def test_vertexai_mode_requires_project(self):
        """Test that Vertex AI mode requires a project."""
        with patch.dict(os.environ, {
            "GOOGLE_CLOUD_PROJECT": "",
        }, clear=False):
            with pytest.raises(ValueError, match="Vertex AI requires a GCP project"):
                GeminiClient(vertexai=True, project=None)

    def test_developer_api_mode_default(self):
        """Test default Developer API mode."""
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-api-key",
            "GOOGLE_GENAI_USE_VERTEXAI": "",
        }, clear=False):
            client = GeminiClient()
            
            assert client.vertexai is False
            assert client.api_key == "test-api-key"
            assert client.project is None
            assert client.gcs_bucket is None

    def test_explicit_vertexai_false_overrides_env(self):
        """Test that explicit vertexai=False overrides environment variable."""
        with patch.dict(os.environ, {
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GEMINI_API_KEY": "test-api-key",
        }, clear=False):
            client = GeminiClient(vertexai=False)
            
            assert client.vertexai is False

    def test_gcs_bucket_name_generation(self):
        """Test GCS bucket name generation."""
        with patch.dict(os.environ, {
            "GOOGLE_CLOUD_PROJECT": "test-project",
        }, clear=False):
            client = GeminiClient(
                vertexai=True,
                project="test-project",
            )
            
            bucket_name = client.get_gcs_bucket_name()
            assert bucket_name == "gemini-batch-test-project"

    def test_gcs_bucket_name_explicit(self):
        """Test explicit GCS bucket name."""
        with patch.dict(os.environ, {
            "GOOGLE_CLOUD_PROJECT": "test-project",
        }, clear=False):
            client = GeminiClient(
                vertexai=True,
                project="test-project",
                gcs_bucket="custom-bucket",
            )
            
            bucket_name = client.get_gcs_bucket_name()
            assert bucket_name == "custom-bucket"

    def test_gcs_client_requires_vertexai_mode(self):
        """Test that gcs_client property requires Vertex AI mode."""
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-api-key",
            "GOOGLE_GENAI_USE_VERTEXAI": "",
        }, clear=False):
            client = GeminiClient(vertexai=False)
            
            with pytest.raises(RuntimeError, match="GCS client is only available in Vertex AI mode"):
                _ = client.gcs_client

    def test_get_gcs_bucket_name_requires_vertexai_mode(self):
        """Test that get_gcs_bucket_name requires Vertex AI mode."""
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-api-key",
            "GOOGLE_GENAI_USE_VERTEXAI": "",
        }, clear=False):
            client = GeminiClient(vertexai=False)
            
            with pytest.raises(RuntimeError, match="GCS bucket is only available in Vertex AI mode"):
                client.get_gcs_bucket_name()


@pytest.mark.skipif(not HAS_GCS, reason="google-cloud-storage not installed")
class TestGCSUtilitiesMocked:
    """Test GCS utilities with mocked GCS client."""

    @pytest.fixture
    def mock_gcs_client(self):
        """Create a mock GCS client."""
        mock = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        mock.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_bucket.exists.return_value = True
        mock.get_bucket.return_value = mock_bucket
        
        return mock

    @pytest.fixture
    def vertexai_client_mocked(self, mock_gcs_client):
        """Create a GeminiClient with mocked GCS client."""
        with patch.dict(os.environ, {
            "GOOGLE_CLOUD_PROJECT": "test-project",
        }, clear=False):
            client = GeminiClient(
                vertexai=True,
                project="test-project",
                gcs_bucket="test-bucket",
            )
            client._gcs_client = mock_gcs_client
            return client

    def test_upload_to_gcs_path(self, vertexai_client_mocked, tmp_path):
        """Test uploading a file path to GCS."""
        from gemini_batch.utils import upload_to_gcs
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        result = upload_to_gcs(
            vertexai_client_mocked,
            test_file,
            destination_blob_name="test/file.txt",
        )
        
        assert "uri" in result
        assert result["uri"] == "gs://test-bucket/test/file.txt"
        assert "mime_type" in result

    def test_upload_to_gcs_pil_image(self, vertexai_client_mocked):
        """Test uploading a PIL Image to GCS."""
        from gemini_batch.utils import upload_to_gcs
        
        img = Image.new('RGB', (100, 100), color='red')
        
        result = upload_to_gcs(
            vertexai_client_mocked,
            img,
            destination_blob_name="test/image.png",
        )
        
        assert "uri" in result
        assert result["uri"] == "gs://test-bucket/test/image.png"
        assert result["mime_type"] == "image/png"

    def test_upload_to_gcs_bytes_png(self, vertexai_client_mocked):
        """Test uploading PNG bytes to GCS."""
        from gemini_batch.utils import upload_to_gcs
        
        # Create PNG bytes
        img = Image.new('RGB', (10, 10), color='blue')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        
        result = upload_to_gcs(
            vertexai_client_mocked,
            png_bytes,
            destination_blob_name="test/bytes.png",
        )
        
        assert result["mime_type"] == "image/png"

    def test_upload_to_gcs_bytes_jpeg(self, vertexai_client_mocked):
        """Test uploading JPEG bytes to GCS."""
        from gemini_batch.utils import upload_to_gcs
        
        # Create JPEG bytes
        img = Image.new('RGB', (10, 10), color='green')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        jpeg_bytes = buf.getvalue()
        
        result = upload_to_gcs(
            vertexai_client_mocked,
            jpeg_bytes,
            destination_blob_name="test/bytes.jpg",
        )
        
        assert result["mime_type"] == "image/jpeg"

    def test_upload_to_gcs_requires_vertexai_mode(self, tmp_path):
        """Test that upload_to_gcs requires Vertex AI mode."""
        from gemini_batch.utils import upload_to_gcs
        
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-key",
            "GOOGLE_GENAI_USE_VERTEXAI": "",
        }, clear=False):
            client = GeminiClient(vertexai=False)
            test_file = tmp_path / "test.txt"
            test_file.write_text("test")
            
            with pytest.raises(RuntimeError, match="upload_to_gcs requires Vertex AI mode"):
                upload_to_gcs(client, test_file)

    def test_download_from_gcs(self, vertexai_client_mocked, tmp_path):
        """Test downloading a file from GCS."""
        from gemini_batch.utils import download_from_gcs
        
        download_path = str(tmp_path / "downloaded.txt")
        
        result = download_from_gcs(
            vertexai_client_mocked,
            "gs://test-bucket/test/file.txt",
            download_path,
        )
        
        assert result == download_path
        # Verify download_to_filename was called
        vertexai_client_mocked._gcs_client.bucket.return_value.blob.return_value.download_to_filename.assert_called_once()

    def test_download_from_gcs_invalid_uri(self, vertexai_client_mocked):
        """Test download_from_gcs with invalid URI."""
        from gemini_batch.utils import download_from_gcs
        
        with pytest.raises(ValueError, match="Invalid GCS URI"):
            download_from_gcs(vertexai_client_mocked, "invalid-uri")
        
        with pytest.raises(ValueError, match="Invalid GCS URI"):
            download_from_gcs(vertexai_client_mocked, "gs://bucket-only")

    def test_list_gcs_blobs(self, vertexai_client_mocked):
        """Test listing blobs in GCS."""
        from gemini_batch.utils import list_gcs_blobs
        
        # Mock blob listing
        mock_blob1 = MagicMock()
        mock_blob1.name = "prefix/file1.txt"
        mock_blob2 = MagicMock()
        mock_blob2.name = "prefix/file2.txt"
        
        vertexai_client_mocked._gcs_client.bucket.return_value.list_blobs.return_value = [
            mock_blob1, mock_blob2
        ]
        
        result = list_gcs_blobs(
            vertexai_client_mocked,
            prefix="prefix/",
        )
        
        assert len(result) == 2
        assert "gs://test-bucket/prefix/file1.txt" in result
        assert "gs://test-bucket/prefix/file2.txt" in result


class TestUploadFileForBatch:
    """Test upload_file_for_batch routing."""

    def test_routes_to_gcs_for_vertexai(self):
        """Test that upload_file_for_batch routes to GCS for Vertex AI."""
        from gemini_batch.utils import upload_file_for_batch
        
        with patch.dict(os.environ, {
            "GOOGLE_CLOUD_PROJECT": "test-project",
        }, clear=False):
            client = GeminiClient(
                vertexai=True,
                project="test-project",
                gcs_bucket="test-bucket",
            )
            
            with patch('gemini_batch.utils.upload_to_gcs') as mock_upload:
                mock_upload.return_value = {"uri": "gs://test-bucket/file.png", "mime_type": "image/png"}
                
                img = Image.new('RGB', (10, 10))
                result = upload_file_for_batch(img, client)
                
                mock_upload.assert_called_once()
                assert result["uri"].startswith("gs://")

    def test_routes_to_file_api_for_developer(self):
        """Test that upload_file_for_batch routes to File API for Developer API."""
        from gemini_batch.utils import upload_file_for_batch
        
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-key",
            "GOOGLE_GENAI_USE_VERTEXAI": "",
        }, clear=False):
            client = GeminiClient(vertexai=False)
            
            with patch('gemini_batch.utils.upload_file_to_gemini') as mock_upload:
                mock_upload.return_value = {"uri": "files/abc123", "mime_type": "image/png"}
                
                img = Image.new('RGB', (10, 10))
                result = upload_file_for_batch(img, client)
                
                mock_upload.assert_called_once()
                assert not result["uri"].startswith("gs://")


class TestBatchJobCreationVertexAI:
    """Test batch job creation with Vertex AI."""

    @pytest.mark.skipif(not HAS_GCS, reason="google-cloud-storage not installed")
    def test_create_batch_job_uploads_to_gcs(self):
        """Test that create_batch_job uploads JSONL to GCS for Vertex AI."""
        from gemini_batch.batch import create_batch_job
        
        with patch.dict(os.environ, {
            "GOOGLE_CLOUD_PROJECT": "test-project",
        }, clear=False):
            # Create mock client
            mock_genai_client = MagicMock()
            mock_batch_job = MagicMock()
            mock_batch_job.name = "projects/test/batches/123"
            mock_genai_client.batches.create.return_value = mock_batch_job
            
            client = GeminiClient(
                vertexai=True,
                project="test-project",
                gcs_bucket="test-bucket",
            )
            client._client = mock_genai_client
            
            # Mock GCS
            mock_gcs = MagicMock()
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_gcs.bucket.return_value = mock_bucket
            mock_gcs.get_bucket.return_value = mock_bucket
            mock_bucket.blob.return_value = mock_blob
            client._gcs_client = mock_gcs
            
            requests = [
                {"key": "req1", "request": {"contents": [{"parts": [{"text": "test"}]}]}}
            ]
            
            job_name = create_batch_job(
                requests=requests,
                model_name="gemini-2.0-flash",
                client=client,
            )
            
            assert job_name == "projects/test/batches/123"
            
            # Verify batches.create was called with GCS URI
            call_args = mock_genai_client.batches.create.call_args
            assert call_args.kwargs['src'].startswith("gs://")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
