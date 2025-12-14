"""Tests for batch embedding functionality."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path

from gemini_batch.embedding import (
    create_embedding_batch_job,
    download_embedding_results,
    parse_embedding_results,
    batch_embed,
)
from gemini_batch import config


class TestCreateEmbeddingBatchJob:
    """Tests for create_embedding_batch_job."""

    @patch('gemini_batch.embedding.GeminiClient')
    def test_creates_jsonl_with_embedding_format(self, mock_client_class, tmp_path):
        """Test that JSONL uses embedding format (content, task_type)."""
        mock_client = MagicMock()
        mock_client.vertexai = False
        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/embed-job-123"
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/uploaded-123"
        mock_client.client.files.upload.return_value = mock_uploaded_file
        mock_client.client.batches.create_embeddings.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        texts = ["Hello world", "Test embedding"]

        job_name = create_embedding_batch_job(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
            jsonl_dir=str(tmp_path),
            client=mock_client,
        )

        # Verify job name returned
        assert job_name == "batches/embed-job-123"

        # Find and verify JSONL content
        jsonl_files = list(tmp_path.glob("embed_*_requests.jsonl"))
        assert len(jsonl_files) == 1

        with open(jsonl_files[0], "r") as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Verify embedding format
        req1 = json.loads(lines[0])
        assert req1["key"] == "0"
        assert "content" in req1["request"]
        assert "task_type" in req1["request"]
        assert req1["request"]["task_type"] == "RETRIEVAL_DOCUMENT"
        assert req1["request"]["content"]["parts"][0]["text"] == "Hello world"

    @patch('gemini_batch.embedding.GeminiClient')
    def test_uses_create_embeddings_api(self, mock_client_class, tmp_path):
        """Test that create_embeddings API is used (not create)."""
        mock_client = MagicMock()
        mock_client.vertexai = False
        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/embed-job-123"
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/uploaded-123"
        mock_client.client.files.upload.return_value = mock_uploaded_file
        mock_client.client.batches.create_embeddings.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        create_embedding_batch_job(
            ["Test text"],
            jsonl_dir=str(tmp_path),
            client=mock_client,
        )

        # Verify create_embeddings was called, not create
        mock_client.client.batches.create_embeddings.assert_called_once()
        mock_client.client.batches.create.assert_not_called()

    @patch('gemini_batch.embedding.GeminiClient')
    def test_validates_task_type(self, mock_client_class, tmp_path):
        """Test that invalid task_type raises error."""
        mock_client = MagicMock()
        mock_client.vertexai = False
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Invalid task_type"):
            create_embedding_batch_job(
                ["Test text"],
                task_type="INVALID_TASK",
                jsonl_dir=str(tmp_path),
                client=mock_client,
            )

    def test_vertexai_not_supported(self):
        """Test that Vertex AI raises clear error."""
        mock_client = MagicMock()
        mock_client.vertexai = True

        with pytest.raises(ValueError, match="not supported with Vertex AI"):
            create_embedding_batch_job(
                ["Test text"],
                client=mock_client,
            )

    @patch('gemini_batch.embedding.GeminiClient')
    def test_all_valid_task_types(self, mock_client_class, tmp_path):
        """Test that all valid task types are accepted."""
        mock_client = MagicMock()
        mock_client.vertexai = False
        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/embed-job-123"
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/uploaded-123"
        mock_client.client.files.upload.return_value = mock_uploaded_file
        mock_client.client.batches.create_embeddings.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        for task_type in config.EMBEDDING_CONFIG["valid_task_types"]:
            # Should not raise
            create_embedding_batch_job(
                ["Test text"],
                task_type=task_type,
                jsonl_dir=str(tmp_path),
                client=mock_client,
            )


class TestDownloadEmbeddingResults:
    """Tests for download_embedding_results."""

    @patch('gemini_batch.embedding.GeminiClient')
    def test_download_from_file_api(self, mock_client_class, tmp_path):
        """Test downloading embedding results from File API."""
        mock_client = MagicMock()
        mock_client.vertexai = False
        mock_batch_job = MagicMock()
        mock_state = MagicMock()
        mock_state.name = 'JOB_STATE_SUCCEEDED'
        mock_batch_job.state = mock_state
        mock_batch_job.display_name = "embed-1234567890"
        mock_dest = MagicMock()
        mock_dest.file_name = "files/results-123"
        mock_dest.gcs_uri = None
        mock_batch_job.dest = mock_dest
        file_content = b'{"key": "0", "response": {"embedding": {"values": [0.1, 0.2]}}}'
        mock_client.client.files.download.return_value = file_content
        mock_client.client.batches.get.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        result_path = download_embedding_results(
            "batches/embed-job",
            output_dir=str(tmp_path),
            client=mock_client,
        )

        assert Path(result_path).exists()
        assert Path(result_path).read_bytes() == file_content

    @patch('gemini_batch.embedding.GeminiClient')
    def test_error_when_not_succeeded(self, mock_client_class):
        """Test error when downloading results from non-succeeded job."""
        mock_client = MagicMock()
        mock_batch_job = MagicMock()
        mock_state = MagicMock()
        mock_state.name = 'JOB_STATE_RUNNING'
        mock_batch_job.state = mock_state
        mock_client.client.batches.get.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Job not succeeded"):
            download_embedding_results("batches/embed-job", client=mock_client)


class TestParseEmbeddingResults:
    """Tests for parse_embedding_results."""

    def test_parses_dict_format(self, tmp_path):
        """Test parsing embedding vectors from dict format results."""
        results_file = tmp_path / "results.jsonl"
        results_content = [
            {"key": "0", "response": {"embedding": {"values": [0.1, 0.2, 0.3]}}},
            {"key": "1", "response": {"embedding": {"values": [0.4, 0.5, 0.6]}}},
        ]
        with open(results_file, "w") as f:
            for r in results_content:
                f.write(json.dumps(r) + "\n")

        embeddings = parse_embedding_results(str(results_file))

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    def test_preserves_order_by_key(self, tmp_path):
        """Test that results are sorted by key to preserve order."""
        results_file = tmp_path / "results.jsonl"
        # Results out of order
        results_content = [
            {"key": "2", "response": {"embedding": {"values": [0.7, 0.8, 0.9]}}},
            {"key": "0", "response": {"embedding": {"values": [0.1, 0.2, 0.3]}}},
            {"key": "1", "response": {"embedding": {"values": [0.4, 0.5, 0.6]}}},
        ]
        with open(results_file, "w") as f:
            for r in results_content:
                f.write(json.dumps(r) + "\n")

        embeddings = parse_embedding_results(str(results_file))

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]  # key "0"
        assert embeddings[1] == [0.4, 0.5, 0.6]  # key "1"
        assert embeddings[2] == [0.7, 0.8, 0.9]  # key "2"

    def test_preserves_alignment_on_errors(self, tmp_path):
        """Test that None is returned for failed requests to preserve alignment."""
        results_file = tmp_path / "results.jsonl"
        results_content = [
            {"key": "0", "response": {"embedding": {"values": [0.1, 0.2, 0.3]}}},
            {"key": "1", "error": "Request failed"},
            {"key": "2", "response": {"embedding": {"values": [0.7, 0.8, 0.9]}}},
        ]
        with open(results_file, "w") as f:
            for r in results_content:
                f.write(json.dumps(r) + "\n")

        embeddings = parse_embedding_results(str(results_file))

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] is None  # Failed request
        assert embeddings[2] == [0.7, 0.8, 0.9]

    def test_handles_malformed_json(self, tmp_path):
        """Test that malformed JSON lines result in None (sorted to end)."""
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write('{"key": "0", "response": {"embedding": {"values": [0.1, 0.2]}}}\n')
            f.write('not valid json\n')
            f.write('{"key": "1", "response": {"embedding": {"values": [0.3, 0.4]}}}\n')

        embeddings = parse_embedding_results(str(results_file))

        # Malformed JSON (no key) gets sorted to end
        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2]  # key "0"
        assert embeddings[1] == [0.3, 0.4]  # key "1"
        assert embeddings[2] is None  # Malformed JSON (sorted to end)

    def test_returns_metadata_when_requested(self, tmp_path):
        """Test return_metadata=True returns tuple."""
        results_file = tmp_path / "results.jsonl"
        results_content = [
            {
                "key": "0",
                "response": {
                    "embedding": {"values": [0.1, 0.2]},
                    "usageMetadata": {"totalTokenCount": 5, "promptTokenCount": 5}
                }
            },
        ]
        with open(results_file, "w") as f:
            for r in results_content:
                f.write(json.dumps(r) + "\n")

        embeddings, metadata = parse_embedding_results(str(results_file), return_metadata=True)

        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2]
        assert len(metadata) == 1
        assert metadata[0]["usageMetadata"]["totalTokenCount"] == 5

    def test_empty_file(self, tmp_path):
        """Test parsing empty file."""
        results_file = tmp_path / "results.jsonl"
        results_file.touch()

        embeddings = parse_embedding_results(str(results_file))

        assert embeddings == []


class TestBatchEmbed:
    """Tests for batch_embed main function."""

    @patch('gemini_batch.embedding.GeminiClient')
    def test_wait_false_returns_job_name(self, mock_client_class, tmp_path):
        """Test wait=False returns job name immediately."""
        mock_client = MagicMock()
        mock_client.vertexai = False
        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/embed-job-123"
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/uploaded-123"
        mock_client.client.files.upload.return_value = mock_uploaded_file
        mock_client.client.batches.create_embeddings.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        result = batch_embed(
            ["Test text"],
            wait=False,
            jsonl_dir=str(tmp_path),
        )

        assert result == "batches/embed-job-123"

    @patch('gemini_batch.embedding.download_embedding_results')
    @patch('gemini_batch.embedding.monitor_batch_job')
    @patch('gemini_batch.embedding.create_embedding_batch_job')
    @patch('gemini_batch.embedding.GeminiClient')
    def test_wait_true_returns_embeddings(
        self, mock_client_class, mock_create, mock_monitor, mock_download, tmp_path
    ):
        """Test wait=True returns parsed embeddings."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_create.return_value = "batches/embed-job-123"
        mock_monitor.return_value = "JOB_STATE_SUCCEEDED"

        # Create mock results file
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write('{"key": "0", "response": {"embedding": {"values": [0.1, 0.2, 0.3]}}}\n')
        mock_download.return_value = str(results_file)

        embeddings = batch_embed(
            ["Test text"],
            wait=True,
            jsonl_dir=str(tmp_path),
            output_dir=str(tmp_path),
        )

        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]

    @patch('gemini_batch.embedding.download_embedding_results')
    @patch('gemini_batch.embedding.monitor_batch_job')
    @patch('gemini_batch.embedding.create_embedding_batch_job')
    @patch('gemini_batch.embedding.GeminiClient')
    def test_raises_on_failed_job(
        self, mock_client_class, mock_create, mock_monitor, mock_download
    ):
        """Test RuntimeError raised when job fails."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_create.return_value = "batches/embed-job-123"
        mock_monitor.return_value = "JOB_STATE_FAILED"

        with pytest.raises(RuntimeError, match="failed with state"):
            batch_embed(["Test text"], wait=True)

    @patch('gemini_batch.embedding.GeminiClient')
    def test_validates_task_type(self, mock_client_class):
        """Test invalid task_type raises error in batch_embed."""
        mock_client = MagicMock()
        mock_client.vertexai = False
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Invalid task_type"):
            batch_embed(["Test text"], task_type="INVALID_TYPE")

    @patch('gemini_batch.embedding.download_embedding_results')
    @patch('gemini_batch.embedding.monitor_batch_job')
    @patch('gemini_batch.embedding.create_embedding_batch_job')
    @patch('gemini_batch.embedding.GeminiClient')
    def test_return_metadata(
        self, mock_client_class, mock_create, mock_monitor, mock_download, tmp_path
    ):
        """Test return_metadata=True returns tuple with metadata."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_create.return_value = "batches/embed-job-123"
        mock_monitor.return_value = "JOB_STATE_SUCCEEDED"

        # Create mock results file with metadata
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write('{"key": "0", "response": {"embedding": {"values": [0.1, 0.2]}, "usageMetadata": {"totalTokenCount": 10}}}\n')
        mock_download.return_value = str(results_file)

        embeddings, metadata = batch_embed(
            ["Test text"],
            wait=True,
            return_metadata=True,
            jsonl_dir=str(tmp_path),
            output_dir=str(tmp_path),
        )

        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2]
        assert len(metadata) == 1
        assert metadata[0]["usageMetadata"]["totalTokenCount"] == 10
