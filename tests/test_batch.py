"""Tests for batch processing functionality."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
from pydantic import BaseModel

from gemini_batch import batch
from gemini_batch.batch import (
    create_batch_job,
    monitor_batch_job,
    download_batch_results,
    get_inline_results,
    parse_batch_results,
    get_batch_job_output_uri,
    resume_batch_job,
    _is_retryable_network_error,
    _extract_text_from_parts,
)


class TestSchema(BaseModel):
    name: str
    value: int


class TestExtractTextFromParts:
    """Tests for _extract_text_from_parts helper function."""

    def test_single_part_dict(self):
        """Test extracting text from a single dict part."""
        parts = [{"text": "Hello world"}]
        assert _extract_text_from_parts(parts) == "Hello world"

    def test_single_part_object(self):
        """Test extracting text from a single object part with text attribute."""
        part = Mock()
        part.text = "Hello world"
        parts = [part]
        assert _extract_text_from_parts(parts) == "Hello world"

    def test_multiple_parts_no_thought(self):
        """Test that first part is used when it's not a thought marker."""
        parts = [
            {"text": "This is the main content"},
            {"text": "This is secondary content"}
        ]
        assert _extract_text_from_parts(parts) == "This is the main content"

    def test_thought_marker_skipped_dict(self):
        """Test that 'thought' marker part is skipped (dict format)."""
        parts = [
            {"text": " thought\n"},
            {"text": "Actual content here"}
        ]
        assert _extract_text_from_parts(parts) == "Actual content here"

    def test_thought_marker_skipped_object(self):
        """Test that 'thought' marker part is skipped (object format)."""
        part1 = Mock()
        part1.text = " thought\n"
        part2 = Mock()
        part2.text = "Actual content here"
        parts = [part1, part2]
        assert _extract_text_from_parts(parts) == "Actual content here"

    def test_thought_marker_case_insensitive(self):
        """Test that thought detection is case insensitive."""
        parts = [
            {"text": " THOUGHT\n"},
            {"text": "Actual content here"}
        ]
        assert _extract_text_from_parts(parts) == "Actual content here"

    def test_thought_in_long_text_not_skipped(self):
        """Test that 'thought' in text longer than 20 chars is not skipped."""
        parts = [
            {"text": "This is a longer text with thought in it"},
            {"text": "Secondary content"}
        ]
        assert _extract_text_from_parts(parts) == "This is a longer text with thought in it"

    def test_short_text_without_thought_not_skipped(self):
        """Test that short text without 'thought' is not skipped."""
        parts = [
            {"text": "OK\n"},
            {"text": "Secondary content"}
        ]
        assert _extract_text_from_parts(parts) == "OK\n"

    def test_empty_parts_raises(self):
        """Test that empty parts list raises ValueError."""
        with pytest.raises(ValueError, match="No parts in response"):
            _extract_text_from_parts([])

    def test_no_text_attribute_raises(self):
        """Test that part without text attribute raises ValueError."""
        part = Mock(spec=[])  # No text attribute
        parts = [part]
        with pytest.raises(ValueError, match="First part has no text attribute"):
            _extract_text_from_parts(parts)

    def test_second_part_no_text_falls_back(self):
        """Test that if second part has no text, falls back to first part."""
        part2 = Mock(spec=[])  # No text attribute
        parts = [
            {"text": " thought\n"},
            part2
        ]
        # Should fall back to first part since second has no text
        assert _extract_text_from_parts(parts) == " thought\n"


@patch('gemini_batch.batch.GeminiClient')
def test_create_batch_job_file(mock_client_class, tmp_path):
    """Test batch job creation with file-based mode."""
    mock_client = MagicMock()
    mock_client.vertexai = False  # Explicitly set Developer API mode
    mock_batch_job = MagicMock()
    mock_batch_job.name = "batches/test-job-456"
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "files/uploaded-123"
    mock_client.client.files.upload.return_value = mock_uploaded_file
    mock_client.client.batches.create.return_value = mock_batch_job
    mock_client_class.return_value = mock_client

    requests = [
        {"key": "req1", "request": {"contents": [{"text": "test1"}]}},
    ]

    job_name = create_batch_job(requests, jsonl_dir=str(tmp_path))

    assert job_name == "batches/test-job-456"
    mock_client.client.files.upload.assert_called_once()
    mock_client.client.batches.create.assert_called_once()


@patch('gemini_batch.batch.GeminiClient')
@patch('time.sleep', return_value=None)
def test_monitor_batch_job_success(mock_sleep, mock_client_class):
    """Test monitoring a batch job to completion."""
    mock_client = MagicMock()
    mock_batch_job = MagicMock()
    mock_state = MagicMock()
    mock_state.name = 'JOB_STATE_SUCCEEDED'
    mock_batch_job.state = mock_state

    # Mock batch_stats to avoid comparison errors
    mock_stats = MagicMock()
    mock_stats.failed_request_count = 0
    mock_batch_job.batch_stats = mock_stats

    mock_client.client.batches.get.return_value = mock_batch_job
    mock_client_class.return_value = mock_client

    state = monitor_batch_job("batches/test-job")

    assert state == 'JOB_STATE_SUCCEEDED'
    mock_client.client.batches.get.assert_called()


@patch('gemini_batch.batch.GeminiClient')
@patch('time.sleep', return_value=None)
def test_monitor_batch_job_failed(mock_sleep, mock_client_class):
    """Test monitoring a failed batch job."""
    mock_client = MagicMock()
    mock_batch_job = MagicMock()
    mock_state = MagicMock()
    mock_state.name = 'JOB_STATE_FAILED'
    mock_batch_job.state = mock_state
    mock_batch_job.error = "Some error"
    mock_client.client.batches.get.return_value = mock_batch_job
    mock_client_class.return_value = mock_client

    state = monitor_batch_job("batches/test-job")

    assert state == 'JOB_STATE_FAILED'


@patch('gemini_batch.batch.GeminiClient')
@patch('time.sleep', return_value=None)
def test_monitor_batch_job_retries_on_network_error(mock_sleep, mock_client_class):
    """Test that monitor_batch_job retries on transient network errors."""
    mock_client = MagicMock()
    mock_batch_job = MagicMock()
    mock_state = MagicMock()
    mock_state.name = 'JOB_STATE_SUCCEEDED'
    mock_batch_job.state = mock_state
    mock_stats = MagicMock()
    mock_stats.failed_request_count = 0
    mock_batch_job.batch_stats = mock_stats

    # First two calls raise network error, third succeeds
    class MockConnectError(Exception):
        pass
    MockConnectError.__name__ = 'ConnectError'

    mock_client.client.batches.get.side_effect = [
        MockConnectError("[Errno 101] Network is unreachable"),
        MockConnectError("[Errno 101] Network is unreachable"),
        mock_batch_job,
    ]
    mock_client_class.return_value = mock_client

    state = monitor_batch_job("batches/test-job", retry_count=3, retry_delay=1.0)

    assert state == 'JOB_STATE_SUCCEEDED'
    assert mock_client.client.batches.get.call_count == 3
    # Verify exponential backoff: first retry waits 1s, second waits 2s
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


@patch('gemini_batch.batch.GeminiClient')
@patch('time.sleep', return_value=None)
def test_monitor_batch_job_exceeds_max_retries(mock_sleep, mock_client_class):
    """Test that monitor_batch_job raises after exhausting retries."""
    mock_client = MagicMock()

    # All calls raise network error
    class MockConnectError(Exception):
        pass
    MockConnectError.__name__ = 'ConnectError'

    mock_client.client.batches.get.side_effect = MockConnectError("[Errno 101] Network is unreachable")
    mock_client_class.return_value = mock_client

    with pytest.raises(Exception) as exc_info:
        monitor_batch_job("batches/test-job", retry_count=2, retry_delay=1.0)

    assert "Network is unreachable" in str(exc_info.value)
    # Initial call + 2 retries = 3 calls total
    assert mock_client.client.batches.get.call_count == 3


def test_is_retryable_network_error():
    """Test _is_retryable_network_error helper function."""
    # Test ConnectError by type name
    class ConnectError(Exception):
        pass
    assert _is_retryable_network_error(ConnectError("test")) is True

    # Test ConnectionResetError
    assert _is_retryable_network_error(ConnectionResetError()) is True

    # Test by error message
    assert _is_retryable_network_error(Exception("Network is unreachable")) is True
    assert _is_retryable_network_error(Exception("[Errno 101] something")) is True
    assert _is_retryable_network_error(Exception("connection refused")) is True

    # Test non-retryable errors
    assert _is_retryable_network_error(ValueError("invalid value")) is False
    assert _is_retryable_network_error(KeyError("missing key")) is False


@patch('gemini_batch.batch.GeminiClient')
def test_download_batch_results(mock_client_class, tmp_path):
    """Test downloading batch results."""
    mock_client = MagicMock()
    mock_client.vertexai = False  # Explicitly set Developer API mode
    mock_batch_job = MagicMock()
    mock_state = MagicMock()
    mock_state.name = 'JOB_STATE_SUCCEEDED'
    mock_batch_job.state = mock_state
    mock_batch_job.display_name = "test-results"
    mock_dest = MagicMock()
    mock_dest.file_name = "files/results-123"
    mock_dest.gcs_uri = None  # No GCS URI for Developer API
    mock_batch_job.dest = mock_dest
    file_content = b'{"key": "test", "response": {"text": "result"}}'
    mock_client.client.files.download.return_value = file_content
    mock_client.client.batches.get.return_value = mock_batch_job
    mock_client_class.return_value = mock_client

    result_path = download_batch_results("batches/test-job", output_dir=str(tmp_path))

    assert Path(result_path).exists()
    assert Path(result_path).read_bytes() == file_content


@patch('gemini_batch.batch.GeminiClient')
def test_download_batch_results_not_succeeded(mock_client_class):
    """Test error when downloading results from non-succeeded job."""
    mock_client = MagicMock()
    mock_batch_job = MagicMock()
    mock_state = MagicMock()
    mock_state.name = 'JOB_STATE_RUNNING'
    mock_batch_job.state = mock_state
    mock_client.client.batches.get.return_value = mock_batch_job
    mock_client_class.return_value = mock_client

    with pytest.raises(ValueError, match="Job not succeeded"):
        download_batch_results("batches/test-job")


@patch('gemini_batch.batch.GeminiClient')
def test_get_inline_results(mock_client_class):
    """Test getting inline results."""
    mock_client = MagicMock()
    mock_batch_job = MagicMock()
    mock_state = MagicMock()
    mock_state.name = 'JOB_STATE_SUCCEEDED'
    mock_batch_job.state = mock_state

    # Create mock inline responses
    mock_response1 = MagicMock()
    mock_response1.response = {"text": "result1"}
    mock_response1.error = None

    mock_response2 = MagicMock()
    mock_response2.response = None
    mock_response2.error = "Error occurred"

    mock_dest = MagicMock()
    mock_dest.inlined_responses = [mock_response1, mock_response2]
    mock_dest.file_name = None
    mock_batch_job.dest = mock_dest

    mock_client.client.batches.get.return_value = mock_batch_job
    mock_client_class.return_value = mock_client

    results = get_inline_results("batches/test-job")

    assert len(results) == 2
    assert results[0]['index'] == 0
    assert results[0]['response'] == {"text": "result1"}
    assert results[1]['index'] == 1
    assert 'error' in results[1]


def test_parse_batch_results_from_file(tmp_path):
    """Test parsing results from JSONL file."""
    # Create mock JSONL file
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Test", "value": 42}'
                        }]
                    }
                }]
            }
        },
        {
            "key": "req2",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Test2", "value": 99}'
                        }]
                    }
                }]
            }
        }
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    parsed = parse_batch_results(str(results_file), TestSchema)

    assert len(parsed) == 2
    assert isinstance(parsed[0], TestSchema)
    assert parsed[0].name == "Test"
    assert parsed[0].value == 42
    assert parsed[1].name == "Test2"
    assert parsed[1].value == 99


def test_parse_batch_results_from_list():
    """Test parsing results from list of dicts."""
    results_data = [
        {
            "index": 0,
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Inline", "value": 123}'
                        }]
                    }
                }]
            }
        }
    ]

    parsed = parse_batch_results(results_data, TestSchema)

    assert len(parsed) == 1
    assert parsed[0].name == "Inline"
    assert parsed[0].value == 123


def test_parse_batch_results_with_errors(tmp_path):
    """Test parsing results with errors - preserves alignment with None."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "error": "Request failed"
        },
        {
            "key": "req2",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Good", "value": 1}'
                        }]
                    }
                }]
            }
        }
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    parsed = parse_batch_results(str(results_file), TestSchema)

    # Should preserve alignment - None for error, result for good
    assert len(parsed) == 2
    assert parsed[0] is None  # Failed request
    assert parsed[1].name == "Good"
    assert parsed[1].value == 1


def test_parse_batch_results_validation_error(tmp_path):
    """Test parsing with schema validation errors - now non-fatal, returns None for invalid."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Test", "value": "not_an_int"}'  # Invalid
                        }]
                    }
                }]
            }
        }
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    # Validation errors are now non-fatal - should return list with None instead of raising
    parsed = parse_batch_results(str(results_file), TestSchema, validate=True)
    assert len(parsed) == 1
    assert parsed[0] is None  # Validation error results in None


def test_parse_batch_results_raw_text(tmp_path):
    """Test parsing results as raw text when schema is None."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": "This is a raw text response"
                        }]
                    }
                }]
            }
        },
        {
            "key": "req2",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": "Another text response"
                        }]
                    }
                }]
            }
        }
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    parsed = parse_batch_results(str(results_file), schema=None)

    assert len(parsed) == 2
    assert isinstance(parsed[0], str)
    assert isinstance(parsed[1], str)
    assert parsed[0] == "This is a raw text response"
    assert parsed[1] == "Another text response"


def test_parse_batch_results_raw_text_from_list():
    """Test parsing raw text results from list of dicts."""
    results_data = [
        {
            "index": 0,
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": "Raw inline response"
                        }]
                    }
                }]
            }
        }
    ]

    parsed = parse_batch_results(results_data, schema=None)

    assert len(parsed) == 1
    assert isinstance(parsed[0], str)
    assert parsed[0] == "Raw inline response"


def test_parse_batch_results_with_metadata(tmp_path):
    """Test parsing results with return_metadata=True."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Test1", "value": 10}'
                        }]
                    }
                }],
                "usageMetadata": {
                    "totalTokenCount": 100,
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 75,
                    "cachedContentTokenCount": 15,
                    "thoughtsTokenCount": 5
                },
                "modelVersion": "gemini-2.5-flash"
            }
        },
        {
            "key": "req2",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Test2", "value": 20}'
                        }]
                    }
                }],
                "usageMetadata": {
                    "totalTokenCount": 150,
                    "promptTokenCount": 30,
                    "candidatesTokenCount": 120,
                    "cachedContentTokenCount": 0,
                    "thoughtsTokenCount": 0
                },
                "modelVersion": "gemini-2.5-flash"
            }
        }
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    parsed, metadata = parse_batch_results(str(results_file), TestSchema, return_metadata=True)

    # Verify results
    assert len(parsed) == 2
    assert isinstance(parsed[0], TestSchema)
    assert parsed[0].name == "Test1"
    assert parsed[0].value == 10
    assert parsed[1].name == "Test2"
    assert parsed[1].value == 20

    # Verify metadata
    assert len(metadata) == 2
    assert metadata[0]['usageMetadata']['totalTokenCount'] == 100
    assert metadata[0]['usageMetadata']['cachedContentTokenCount'] == 15
    assert metadata[0]['usageMetadata']['thoughtsTokenCount'] == 5
    assert metadata[0]['modelVersion'] == "gemini-2.5-flash"
    assert metadata[1]['usageMetadata']['totalTokenCount'] == 150
    assert metadata[1]['usageMetadata']['cachedContentTokenCount'] == 0
    assert metadata[1]['usageMetadata']['thoughtsTokenCount'] == 0
    assert metadata[1]['modelVersion'] == "gemini-2.5-flash"


def test_parse_batch_results_with_metadata_raw_text(tmp_path):
    """Test parsing raw text with return_metadata=True."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": "Raw response text"
                        }]
                    }
                }],
                "usageMetadata": {
                    "totalTokenCount": 50,
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 40,
                    "thoughtsTokenCount": None
                }
            }
        }
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    parsed, metadata = parse_batch_results(str(results_file), schema=None, return_metadata=True)

    # Verify results
    assert len(parsed) == 1
    assert isinstance(parsed[0], str)
    assert parsed[0] == "Raw response text"

    # Verify metadata
    assert len(metadata) == 1
    assert metadata[0]['usageMetadata']['totalTokenCount'] == 50
    assert metadata[0]['usageMetadata']['thoughtsTokenCount'] is None


def test_parse_batch_results_with_metadata_errors(tmp_path):
    """Test that metadata preserves alignment with None for error responses."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "error": "Request failed"
        },
        {
            "key": "req2",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Good", "value": 1}'
                        }]
                    }
                }],
                "usageMetadata": {
                    "totalTokenCount": 60
                }
            }
        }
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    parsed, metadata = parse_batch_results(str(results_file), TestSchema, return_metadata=True)

    # Should preserve alignment with None for error
    assert len(parsed) == 2
    assert parsed[0] is None  # Error
    assert parsed[1].name == "Good"

    # Metadata should also preserve alignment
    assert len(metadata) == 2
    assert metadata[0] is None  # No metadata for error
    assert metadata[1]['usageMetadata']['totalTokenCount'] == 60


def test_parse_batch_results_with_markdown_wrapped_json(tmp_path):
    """Test parsing JSON wrapped in markdown code blocks."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '```json\n{"name": "Test1", "value": 10}\n```'
                        }]
                    }
                }]
            }
        },
        {
            "key": "req2",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '```\n{"name": "Test2", "value": 20}\n```'
                        }]
                    }
                }]
            }
        }
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    parsed = parse_batch_results(str(results_file), TestSchema)

    assert len(parsed) == 2
    assert parsed[0].name == "Test1"
    assert parsed[0].value == 10
    assert parsed[1].name == "Test2"
    assert parsed[1].value == 20


def test_parse_batch_results_with_explanatory_text(tmp_path):
    """Test parsing JSON with explanatory prefix and suffix text."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": 'Here is the extracted data: {"name": "Test1", "value": 10}'
                        }]
                    }
                }]
            }
        },
        {
            "key": "req2",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": 'The result is: {"name": "Test2", "value": 20} as requested.'
                        }]
                    }
                }]
            }
        }
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    parsed = parse_batch_results(str(results_file), TestSchema)

    assert len(parsed) == 2
    assert parsed[0].name == "Test1"
    assert parsed[0].value == 10
    assert parsed[1].name == "Test2"
    assert parsed[1].value == 20


def test_parse_batch_results_with_malformed_jsonl_lines(tmp_path):
    """Test that malformed JSONL lines preserve alignment with None."""
    results_file = tmp_path / "results.jsonl"

    # Write file with some malformed lines
    with open(results_file, 'w') as f:
        # Good line
        f.write(json.dumps({
            "key": "req1",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Good1", "value": 1}'
                        }]
                    }
                }]
            }
        }) + '\n')

        # Malformed line (invalid JSON)
        f.write('{"key": "req2", INVALID JSON HERE}\n')

        # Another good line
        f.write(json.dumps({
            "key": "req3",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Good2", "value": 2}'
                        }]
                    }
                }]
            }
        }) + '\n')

        # Empty line (becomes None)
        f.write('\n')

        # Another malformed line
        f.write('not json at all\n')

        # Final good line
        f.write(json.dumps({
            "key": "req4",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Good3", "value": 3}'
                        }]
                    }
                }]
            }
        }) + '\n')

    # Should parse successfully, preserving alignment with None for malformed lines
    parsed = parse_batch_results(str(results_file), TestSchema)

    # Should get 6 results total (3 good, 3 None)
    assert len(parsed) == 6
    assert parsed[0].name == "Good1"
    assert parsed[0].value == 1
    assert parsed[1] is None  # Malformed JSON
    assert parsed[2].name == "Good2"
    assert parsed[2].value == 2
    assert parsed[3] is None  # Empty line
    assert parsed[4] is None  # Malformed JSON
    assert parsed[5].name == "Good3"
    assert parsed[5].value == 3


def test_parse_batch_results_validation_error_non_fatal(tmp_path):
    """Test that schema validation errors don't stop entire batch processing - preserves alignment."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Good1", "value": 1}'
                        }]
                    }
                }]
            }
        },
        {
            "key": "req2",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Bad", "value": "not_an_int"}'  # Invalid type
                        }]
                    }
                }]
            }
        },
        {
            "key": "req3",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Good2", "value": 2}'
                        }]
                    }
                }]
            }
        }
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    # With validation errors now being non-fatal, this should not raise
    parsed = parse_batch_results(str(results_file), TestSchema, validate=True)

    # Should preserve alignment with None for validation error
    assert len(parsed) == 3
    assert parsed[0].name == "Good1"
    assert parsed[0].value == 1
    assert parsed[1] is None  # Validation error
    assert parsed[2].name == "Good2"
    assert parsed[2].value == 2


def test_parse_batch_results_complex_mixed_issues(tmp_path):
    """Test parsing with multiple types of issues in one batch."""
    results_file = tmp_path / "results.jsonl"

    with open(results_file, 'w') as f:
        # 1. Good result
        f.write(json.dumps({
            "key": "req1",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Good", "value": 1}'
                        }]
                    }
                }]
            }
        }) + '\n')

        # 2. Result with markdown wrapper
        f.write(json.dumps({
            "key": "req2",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '```json\n{"name": "Markdown", "value": 2}\n```'
                        }]
                    }
                }]
            }
        }) + '\n')

        # 3. Malformed JSONL line
        f.write('CORRUPTED LINE\n')

        # 4. Result with explanatory text
        f.write(json.dumps({
            "key": "req4",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": 'Result: {"name": "Prefixed", "value": 3} done!'
                        }]
                    }
                }]
            }
        }) + '\n')

        # 5. Result with validation error
        f.write(json.dumps({
            "key": "req5",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Invalid", "value": "bad_type"}'
                        }]
                    }
                }]
            }
        }) + '\n')

        # 6. Result with error field
        f.write(json.dumps({
            "key": "req6",
            "error": "API error occurred"
        }) + '\n')

        # 7. Another good result
        f.write(json.dumps({
            "key": "req7",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"name": "Final", "value": 7}'
                        }]
                    }
                }]
            }
        }) + '\n')

    # Should successfully parse despite multiple issues - preserves alignment
    parsed = parse_batch_results(str(results_file), TestSchema, validate=True)

    # Should get 7 results total (3 successful, 3 None, 1 successful)
    assert len(parsed) == 7
    assert parsed[0].name == "Good"
    assert parsed[1].name == "Markdown"
    assert parsed[2] is None  # Malformed JSONL line
    assert parsed[3].name == "Prefixed"
    assert parsed[4] is None  # Validation error
    assert parsed[5] is None  # Error field
    assert parsed[6].name == "Final"


def test_parse_batch_results_preserves_alignment(tmp_path):
    """Test that output length always matches input length."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {"key": "req1", "response": {"candidates": [{"content": {"parts": [{"text": '{"name": "R1", "value": 1}'}]}}]}},
        {"key": "req2", "error": "Failed"},
        {"key": "req3", "response": {"candidates": [{"content": {"parts": [{"text": '{"name": "R3", "value": 3}'}]}}]}},
        {"key": "req4", "response": {"candidates": [{"content": {"parts": [{"text": '{"name": "R4", "value": "bad"}'}]}}]}},  # Validation error
        {"key": "req5", "response": {"candidates": [{"content": {"parts": [{"text": '{"name": "R5", "value": 5}'}]}}]}},
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    parsed = parse_batch_results(str(results_file), TestSchema)

    # Verify alignment is preserved
    assert len(parsed) == 5
    assert parsed[0].name == "R1"
    assert parsed[1] is None  # Error
    assert parsed[2].name == "R3"
    assert parsed[3] is None  # Validation error
    assert parsed[4].name == "R5"


def test_parse_batch_results_alignment_with_metadata(tmp_path):
    """Test that metadata list also preserves alignment."""
    results_file = tmp_path / "results.jsonl"
    results_data = [
        {
            "key": "req1",
            "response": {
                "candidates": [{"content": {"parts": [{"text": '{"name": "R1", "value": 1}'}]}}],
                "usageMetadata": {"totalTokenCount": 100},
                "modelVersion": "model-v1"
            }
        },
        {"key": "req2", "error": "Failed"},
        {
            "key": "req3",
            "response": {
                "candidates": [{"content": {"parts": [{"text": '{"name": "R3", "value": 3}'}]}}],
                "usageMetadata": {"totalTokenCount": 200},
                "modelVersion": "model-v1"
            }
        },
    ]

    with open(results_file, 'w') as f:
        for line in results_data:
            f.write(json.dumps(line) + '\n')

    parsed, metadata = parse_batch_results(str(results_file), TestSchema, return_metadata=True)

    # Verify both lists have same length and alignment
    assert len(parsed) == 3
    assert len(metadata) == 3
    assert parsed[0].name == "R1"
    assert metadata[0]['usageMetadata']['totalTokenCount'] == 100
    assert parsed[1] is None
    assert metadata[1] is None  # None metadata for failed result
    assert parsed[2].name == "R3"
    assert metadata[2]['usageMetadata']['totalTokenCount'] == 200


def test_parse_batch_results_with_genai_response_objects():
    """Test parsing results with genai response objects (not dicts) including cachedContentTokenCount."""
    # Create mock genai response objects
    mock_usage_metadata = Mock()
    mock_usage_metadata.total_token_count = 150
    mock_usage_metadata.prompt_token_count = 50
    mock_usage_metadata.candidates_token_count = 90
    mock_usage_metadata.cached_content_token_count = 30
    mock_usage_metadata.thoughts_token_count = 10

    mock_part = Mock()
    mock_part.text = '{"name": "Test", "value": 42}'

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_candidate = Mock()
    mock_candidate.content = mock_content

    mock_response = Mock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = mock_usage_metadata
    mock_response.model_version = "gemini-2.0-flash"

    results_data = [
        {
            "key": "req1",
            "response": mock_response
        }
    ]

    parsed, metadata = parse_batch_results(results_data, TestSchema, return_metadata=True)

    # Verify parsed result
    assert len(parsed) == 1
    assert parsed[0].name == "Test"
    assert parsed[0].value == 42

    # Verify metadata includes all fields including cachedContentTokenCount
    assert len(metadata) == 1
    assert metadata[0]['usageMetadata']['totalTokenCount'] == 150
    assert metadata[0]['usageMetadata']['promptTokenCount'] == 50
    assert metadata[0]['usageMetadata']['candidatesTokenCount'] == 90
    assert metadata[0]['usageMetadata']['cachedContentTokenCount'] == 30
    assert metadata[0]['usageMetadata']['thoughtsTokenCount'] == 10
    assert metadata[0]['modelVersion'] == "gemini-2.0-flash"


def test_extract_timestamp_from_display_name():
    """Test timestamp extraction from display_name."""
    from gemini_batch.batch import extract_timestamp_from_display_name

    # Valid patterns
    assert extract_timestamp_from_display_name("batch-1700000000") == 1700000000
    assert extract_timestamp_from_display_name("batch-1234567890") == 1234567890
    assert extract_timestamp_from_display_name("batch-999") == 999

    # Invalid patterns - should return None
    assert extract_timestamp_from_display_name("my-custom-job") is None
    assert extract_timestamp_from_display_name("batch-abc") is None
    assert extract_timestamp_from_display_name("") is None
    assert extract_timestamp_from_display_name(None) is None
    assert extract_timestamp_from_display_name("no-timestamp-here") is None


@patch('gemini_batch.batch.GeminiClient')
@patch('time.time', return_value=1700000000)
def test_timestamp_alignment_in_filenames(mock_time, mock_client_class, tmp_path):
    """Test that request and result files use the same timestamp."""
    mock_client = MagicMock()
    mock_client.vertexai = False  # Explicitly set Developer API mode
    mock_batch_job = MagicMock()
    mock_batch_job.name = "batches/test-job-456"
    mock_batch_job.display_name = "batch-1700000000"  # Display name with timestamp
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "files/uploaded-123"
    mock_client.client.files.upload.return_value = mock_uploaded_file
    mock_client.client.batches.create.return_value = mock_batch_job
    mock_client_class.return_value = mock_client

    requests = [
        {"key": "req1", "request": {"contents": [{"text": "test1"}]}},
    ]

    # Create batch job - should create batch_1700000000_requests.jsonl
    job_name = create_batch_job(requests, jsonl_dir=str(tmp_path))

    # Verify display_name was set with timestamp
    create_call_args = mock_client.client.batches.create.call_args
    assert "batch-1700000000" in str(create_call_args)

    # Now test download_batch_results with the same job
    mock_state = MagicMock()
    mock_state.name = 'JOB_STATE_SUCCEEDED'
    mock_batch_job.state = mock_state
    mock_dest = MagicMock()
    mock_dest.file_name = "files/results-123"
    mock_dest.gcs_uri = None  # No GCS URI for Developer API
    mock_batch_job.dest = mock_dest
    file_content = b'{"key": "test", "response": {"text": "result"}}'
    mock_client.client.files.download.return_value = file_content
    mock_client.client.batches.get.return_value = mock_batch_job

    # Download results - should create batch_1700000000_results.jsonl
    result_path = download_batch_results(job_name, output_dir=str(tmp_path))

    # Verify result filename uses the timestamp from display_name
    assert "batch_1700000000_results.jsonl" in result_path


@patch('gemini_batch.batch.GeminiClient')
def test_download_batch_results_with_timestamp_extraction(mock_client_class, tmp_path):
    """Test that download_batch_results extracts timestamp from display_name."""
    mock_client = MagicMock()
    mock_client.vertexai = False  # Explicitly set Developer API mode
    mock_batch_job = MagicMock()
    mock_state = MagicMock()
    mock_state.name = 'JOB_STATE_SUCCEEDED'
    mock_batch_job.state = mock_state
    mock_batch_job.display_name = "batch-1234567890"  # Contains timestamp
    mock_dest = MagicMock()
    mock_dest.file_name = "files/results-123"
    mock_dest.gcs_uri = None  # No GCS URI for Developer API
    mock_batch_job.dest = mock_dest
    file_content = b'{"key": "test", "response": {"text": "result"}}'
    mock_client.client.files.download.return_value = file_content
    mock_client.client.batches.get.return_value = mock_batch_job
    mock_client_class.return_value = mock_client

    result_path = download_batch_results("batches/test-job", output_dir=str(tmp_path))

    # Should create batch_1234567890_results.jsonl
    assert "batch_1234567890_results.jsonl" in result_path
    assert Path(result_path).exists()


@patch('gemini_batch.batch.GeminiClient')
def test_download_batch_results_fallback_for_custom_display_name(mock_client_class, tmp_path):
    """Test fallback behavior when display_name doesn't match pattern."""
    mock_client = MagicMock()
    mock_client.vertexai = False  # Explicitly set Developer API mode
    mock_batch_job = MagicMock()
    mock_state = MagicMock()
    mock_state.name = 'JOB_STATE_SUCCEEDED'
    mock_batch_job.state = mock_state
    mock_batch_job.display_name = "my-custom-job"  # No timestamp pattern
    mock_dest = MagicMock()
    mock_dest.file_name = "files/results-123"
    mock_dest.gcs_uri = None  # No GCS URI for Developer API
    mock_batch_job.dest = mock_dest
    file_content = b'{"key": "test", "response": {"text": "result"}}'
    mock_client.client.files.download.return_value = file_content
    mock_client.client.batches.get.return_value = mock_batch_job
    mock_client_class.return_value = mock_client

    result_path = download_batch_results("batches/test-job", output_dir=str(tmp_path))

    # Should fall back to display_name + .jsonl
    assert "my-custom-job.jsonl" in result_path
    assert Path(result_path).exists()


@patch('gemini_batch.utils.GeminiClient')
def test_batch_process_part_media_resolution_format(mock_client_class, tmp_path):
    """Test that part_media_resolution is correctly formatted as object with level key.

    Per Google's docs, media_resolution on a part should be:
    {"media_resolution": {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"}}
    Not just a string.
    """
    from unittest.mock import AsyncMock
    from gemini_batch import batch_process

    mock_client = MagicMock()
    mock_client.vertexai = False

    # Mock async file upload (used by parallel upload)
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.uri = "https://generativelanguage.googleapis.com/v1/files/uploaded-123"
    mock_uploaded_file.mime_type = "image/png"
    mock_client.client.aio.files.upload = AsyncMock(return_value=mock_uploaded_file)

    # Mock batch creation - capture the JSONL file content
    captured_jsonl_content = []
    mock_batch_job = MagicMock()
    mock_batch_job.name = "batches/test-job-789"

    def capture_batch_create(**kwargs):
        # Read the uploaded JSONL file to capture content
        if 'src' in kwargs:
            src = kwargs['src']
            if hasattr(src, 'uploaded_file') and hasattr(src.uploaded_file, 'file'):
                # Handle types.UploadedFileSource
                pass
        return mock_batch_job

    mock_client.client.batches.create.side_effect = capture_batch_create
    mock_client.client.batches.create.return_value = mock_batch_job

    mock_client_class.return_value = mock_client

    # Create a test image file
    test_image = tmp_path / "test.png"
    # Create a minimal valid PNG file
    import struct
    import zlib
    def create_minimal_png():
        # PNG signature
        signature = b'\x89PNG\r\n\x1a\n'
        # IHDR chunk (1x1 pixel, 8-bit grayscale)
        width = 1
        height = 1
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        # IDAT chunk (compressed scanline with filter byte 0 and one gray pixel)
        raw_data = b'\x00\x00'  # filter=0, gray=0
        compressed = zlib.compress(raw_data)
        idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
        idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
        # IEND chunk
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        return signature + ihdr + idat + iend

    test_image.write_bytes(create_minimal_png())

    # Use wait=False to avoid monitoring
    try:
        batch_process(
            prompts=[["Describe this image:", test_image]],
            schema=TestSchema,
            wait=False,
            part_media_resolution="MEDIA_RESOLUTION_ULTRA_HIGH",
            jsonl_dir=str(tmp_path),
        )
    except Exception:
        pass  # We just want to verify the file upload call

    # Find the JSONL file that was created
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) > 0, "No JSONL file was created"

    # Read and parse the JSONL content
    jsonl_content = jsonl_files[0].read_text()
    request_data = json.loads(jsonl_content.strip())

    # Verify the file part has media_resolution in the correct format
    parts = request_data["request"]["contents"][0]["parts"]
    file_part = None
    for part in parts:
        if "file_data" in part:
            file_part = part
            break

    assert file_part is not None, "No file_data part found in request"
    assert "media_resolution" in file_part, "media_resolution not found in file part"

    # THE KEY ASSERTION: media_resolution should be an object with "level" key
    # NOT just a string
    media_res = file_part["media_resolution"]
    assert isinstance(media_res, dict), f"media_resolution should be a dict, got {type(media_res)}"
    assert "level" in media_res, f"media_resolution should have 'level' key, got {media_res}"
    assert media_res["level"] == "MEDIA_RESOLUTION_ULTRA_HIGH", f"Expected MEDIA_RESOLUTION_ULTRA_HIGH, got {media_res['level']}"


# Tests for image deduplication
class TestContentHash:
    """Tests for _compute_content_hash function."""

    def test_hash_bytes_consistent(self):
        """Same bytes content produces same hash."""
        from gemini_batch import _compute_content_hash

        data = b"test image data" * 100
        hash1 = _compute_content_hash(data)
        hash2 = _compute_content_hash(data)

        assert hash1 == hash2

    def test_hash_bytes_different(self):
        """Different bytes content produces different hash."""
        from gemini_batch import _compute_content_hash

        data1 = b"test image data 1" * 100
        data2 = b"test image data 2" * 100
        hash1 = _compute_content_hash(data1)
        hash2 = _compute_content_hash(data2)

        assert hash1 != hash2

    def test_hash_path_consistent(self, tmp_path):
        """Same file produces same hash."""
        from gemini_batch import _compute_content_hash

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"file content" * 100)

        hash1 = _compute_content_hash(test_file)
        hash2 = _compute_content_hash(test_file)

        assert hash1 == hash2

    def test_hash_path_different_files(self, tmp_path):
        """Different files produce different hashes."""
        from gemini_batch import _compute_content_hash

        file1 = tmp_path / "test1.bin"
        file1.write_bytes(b"content 1" * 100)
        file2 = tmp_path / "test2.bin"
        file2.write_bytes(b"content 2" * 100)

        hash1 = _compute_content_hash(file1)
        hash2 = _compute_content_hash(file2)

        assert hash1 != hash2

    def test_hash_pil_image_consistent(self):
        """Same PIL image produces same hash."""
        from gemini_batch import _compute_content_hash
        from PIL import Image

        img = Image.new('RGB', (100, 100), color='red')

        hash1 = _compute_content_hash(img)
        hash2 = _compute_content_hash(img)

        assert hash1 == hash2

    def test_hash_pil_image_different(self):
        """Different PIL images produce different hashes."""
        from gemini_batch import _compute_content_hash
        from PIL import Image

        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')

        hash1 = _compute_content_hash(img1)
        hash2 = _compute_content_hash(img2)

        assert hash1 != hash2

    def test_hash_size_matters(self, tmp_path):
        """Files with same prefix but different size have different hashes."""
        from gemini_batch import _compute_content_hash

        # Create two files with same first 64KB but different sizes
        chunk = b"x" * 65536  # 64KB
        file1 = tmp_path / "file1.bin"
        file1.write_bytes(chunk)  # Exactly 64KB
        file2 = tmp_path / "file2.bin"
        file2.write_bytes(chunk + b"extra")  # 64KB + extra

        hash1 = _compute_content_hash(file1)
        hash2 = _compute_content_hash(file2)

        assert hash1 != hash2


@patch('gemini_batch.utils.GeminiClient')
def test_batch_process_deduplicates_identical_images(mock_client_class, tmp_path):
    """Test that identical images are only uploaded once."""
    from unittest.mock import AsyncMock
    from gemini_batch import batch_process

    mock_client = MagicMock()
    mock_client.vertexai = False

    # Track upload calls
    upload_call_count = 0
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.uri = "https://generativelanguage.googleapis.com/v1/files/uploaded-123"
    mock_uploaded_file.mime_type = "image/png"

    async def mock_upload(*args, **kwargs):
        nonlocal upload_call_count
        upload_call_count += 1
        return mock_uploaded_file

    mock_client.client.aio.files.upload = mock_upload

    mock_batch_job = MagicMock()
    mock_batch_job.name = "batches/test-job"
    mock_client.client.batches.create.return_value = mock_batch_job

    mock_client_class.return_value = mock_client

    # Create a test image file
    test_image = tmp_path / "test.png"
    # Create a minimal valid PNG
    from PIL import Image
    img = Image.new('RGB', (10, 10), color='red')
    img.save(test_image)

    # Use the SAME image in 3 prompts
    prompts = [
        ["Prompt 1:", test_image],
        ["Prompt 2:", test_image],
        ["Prompt 3:", test_image],
    ]

    try:
        batch_process(
            prompts=prompts,
            schema=TestSchema,
            wait=False,
            jsonl_dir=str(tmp_path),
        )
    except Exception:
        pass  # We just want to verify upload deduplication

    # Should only upload once despite appearing in 3 prompts
    assert upload_call_count == 1, f"Expected 1 upload, got {upload_call_count}"


@patch('gemini_batch.utils.GeminiClient')
def test_batch_process_uploads_different_images(mock_client_class, tmp_path):
    """Test that different images are each uploaded."""
    from unittest.mock import AsyncMock
    from gemini_batch import batch_process

    mock_client = MagicMock()
    mock_client.vertexai = False

    # Track upload calls
    upload_call_count = 0
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.uri = "https://generativelanguage.googleapis.com/v1/files/uploaded-123"
    mock_uploaded_file.mime_type = "image/png"

    async def mock_upload(*args, **kwargs):
        nonlocal upload_call_count
        upload_call_count += 1
        return mock_uploaded_file

    mock_client.client.aio.files.upload = mock_upload

    mock_batch_job = MagicMock()
    mock_batch_job.name = "batches/test-job"
    mock_client.client.batches.create.return_value = mock_batch_job

    mock_client_class.return_value = mock_client

    # Create 3 DIFFERENT test images
    from PIL import Image
    img1 = tmp_path / "test1.png"
    Image.new('RGB', (10, 10), color='red').save(img1)
    img2 = tmp_path / "test2.png"
    Image.new('RGB', (10, 10), color='green').save(img2)
    img3 = tmp_path / "test3.png"
    Image.new('RGB', (10, 10), color='blue').save(img3)

    prompts = [
        ["Prompt 1:", img1],
        ["Prompt 2:", img2],
        ["Prompt 3:", img3],
    ]

    try:
        batch_process(
            prompts=prompts,
            schema=TestSchema,
            wait=False,
            jsonl_dir=str(tmp_path),
        )
    except Exception:
        pass

    # Should upload 3 times for 3 different images
    assert upload_call_count == 3, f"Expected 3 uploads, got {upload_call_count}"


@patch('gemini_batch.utils.GeminiClient')
def test_batch_process_deduplicates_bytes(mock_client_class, tmp_path):
    """Test that identical bytes content is only uploaded once."""
    from unittest.mock import AsyncMock
    from gemini_batch import batch_process

    mock_client = MagicMock()
    mock_client.vertexai = False

    upload_call_count = 0
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.uri = "https://generativelanguage.googleapis.com/v1/files/uploaded-123"
    mock_uploaded_file.mime_type = "image/png"

    async def mock_upload(*args, **kwargs):
        nonlocal upload_call_count
        upload_call_count += 1
        return mock_uploaded_file

    mock_client.client.aio.files.upload = mock_upload

    mock_batch_job = MagicMock()
    mock_batch_job.name = "batches/test-job"
    mock_client.client.batches.create.return_value = mock_batch_job

    mock_client_class.return_value = mock_client

    # Create bytes image content
    from PIL import Image
    import io
    img = Image.new('RGB', (10, 10), color='red')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    image_bytes = buf.getvalue()

    # Use the SAME bytes in 3 prompts
    prompts = [
        ["Prompt 1:", image_bytes],
        ["Prompt 2:", image_bytes],
        ["Prompt 3:", image_bytes],
    ]

    try:
        batch_process(
            prompts=prompts,
            schema=TestSchema,
            wait=False,
            jsonl_dir=str(tmp_path),
        )
    except Exception:
        pass

    # Should only upload once
    assert upload_call_count == 1, f"Expected 1 upload, got {upload_call_count}"


# Tests for types.Part objects in prompts
class TestPartObjects:
    """Tests for using google.genai.types.Part objects in prompts."""

    @patch('gemini_batch.utils.GeminiClient')
    def test_part_with_file_data_and_media_resolution(self, mock_client_class, tmp_path):
        """Test Part with file_data and media_resolution passes through correctly."""
        from google.genai import types
        from gemini_batch import batch_process

        mock_client = MagicMock()
        mock_client.vertexai = False

        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/test-job"
        mock_client.client.batches.create.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        # Create a Part with file_data and media_resolution
        part = types.Part(
            file_data=types.FileData(
                file_uri="https://example.com/files/image.png",
                mime_type="image/png"
            ),
            media_resolution=types.PartMediaResolution(level="MEDIA_RESOLUTION_ULTRA_HIGH")
        )

        prompts = [["Describe this image:", part]]

        try:
            batch_process(
                prompts=prompts,
                schema=TestSchema,
                wait=False,
                jsonl_dir=str(tmp_path),
            )
        except Exception:
            pass

        # Find and read the JSONL file
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) > 0, "No JSONL file was created"

        jsonl_content = jsonl_files[0].read_text()
        request_data = json.loads(jsonl_content.strip())

        parts = request_data["request"]["contents"][0]["parts"]

        # First part should be text
        assert parts[0] == {"text": "Describe this image:"}

        # Second part should have file_data and media_resolution preserved
        file_part = parts[1]
        assert "file_data" in file_part
        assert file_part["file_data"]["file_uri"] == "https://example.com/files/image.png"
        assert file_part["file_data"]["mime_type"] == "image/png"
        assert "media_resolution" in file_part
        assert file_part["media_resolution"]["level"] == "MEDIA_RESOLUTION_ULTRA_HIGH"

    @patch('gemini_batch.utils.GeminiClient')
    def test_part_with_text(self, mock_client_class, tmp_path):
        """Test Part with text content works correctly."""
        from google.genai import types
        from gemini_batch import batch_process

        mock_client = MagicMock()
        mock_client.vertexai = False

        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/test-job"
        mock_client.client.batches.create.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        # Create a Part with text
        part = types.Part(text="This is some text content")

        prompts = [["First part:", part, "last part"]]

        try:
            batch_process(
                prompts=prompts,
                schema=TestSchema,
                wait=False,
                jsonl_dir=str(tmp_path),
            )
        except Exception:
            pass

        # Find and read the JSONL file
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) > 0, "No JSONL file was created"

        jsonl_content = jsonl_files[0].read_text()
        request_data = json.loads(jsonl_content.strip())

        parts = request_data["request"]["contents"][0]["parts"]

        assert parts[0] == {"text": "First part:"}
        assert parts[1] == {"text": "This is some text content"}
        assert parts[2] == {"text": "last part"}

    @patch('gemini_batch.utils.GeminiClient')
    def test_part_media_resolution_precedence(self, mock_client_class, tmp_path):
        """Test that per-part media_resolution overrides global part_media_resolution."""
        from google.genai import types
        from gemini_batch import batch_process

        mock_client = MagicMock()
        mock_client.vertexai = False

        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/test-job"
        mock_client.client.batches.create.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        # Part with its own media_resolution
        part_with_res = types.Part(
            file_data=types.FileData(
                file_uri="https://example.com/files/image1.png",
                mime_type="image/png"
            ),
            media_resolution=types.PartMediaResolution(level="MEDIA_RESOLUTION_LOW")
        )

        # Part without media_resolution (should get global)
        part_without_res = types.Part(
            file_data=types.FileData(
                file_uri="https://example.com/files/image2.png",
                mime_type="image/png"
            )
        )

        prompts = [[part_with_res, part_without_res]]

        try:
            batch_process(
                prompts=prompts,
                schema=TestSchema,
                wait=False,
                jsonl_dir=str(tmp_path),
                part_media_resolution="MEDIA_RESOLUTION_HIGH",  # Global default
            )
        except Exception:
            pass

        # Find and read the JSONL file
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) > 0, "No JSONL file was created"

        jsonl_content = jsonl_files[0].read_text()
        request_data = json.loads(jsonl_content.strip())

        parts = request_data["request"]["contents"][0]["parts"]

        # First part should keep its own LOW resolution
        assert parts[0]["media_resolution"]["level"] == "MEDIA_RESOLUTION_LOW"

        # Second part should get the global HIGH resolution
        assert parts[1]["media_resolution"]["level"] == "MEDIA_RESOLUTION_HIGH"

    def test_part_with_inline_data_raises_error(self):
        """Test that Part with inline_data raises ValueError."""
        from google.genai import types
        from gemini_batch import batch_process

        # Create a Part with inline_data
        part = types.Part(
            inline_data=types.Blob(
                data=b"some image bytes",
                mime_type="image/png"
            )
        )

        prompts = [["Describe this:", part]]

        with pytest.raises(ValueError) as exc_info:
            batch_process(
                prompts=prompts,
                schema=TestSchema,
                wait=False,
            )

        assert "inline_data are not supported" in str(exc_info.value)

    def test_part_with_file_data_no_uri_raises_error(self):
        """Test that Part with file_data but no file_uri raises ValueError."""
        from google.genai import types
        from gemini_batch import batch_process

        # Create a Part with file_data but no file_uri
        part = types.Part(
            file_data=types.FileData(
                mime_type="image/png"
                # file_uri is missing/None
            )
        )

        prompts = [["Describe this:", part]]

        with pytest.raises(ValueError) as exc_info:
            batch_process(
                prompts=prompts,
                schema=TestSchema,
                wait=False,
            )

        assert "file_uri" in str(exc_info.value)


# Tests for resume functionality
class TestResumeBatchJob:
    """Tests for get_batch_job_output_uri and resume_batch_job functions."""

    @patch('gemini_batch.batch.list_gcs_blobs')
    @patch('gemini_batch.batch.GeminiClient')
    def test_get_batch_job_output_uri_success(self, mock_client_class, mock_list_blobs):
        """Test getting GCS output URI from a Vertex AI batch job."""
        mock_client = MagicMock()
        mock_client.vertexai = True
        mock_client.gcs_bucket = "test-bucket"

        mock_batch_job = MagicMock()
        mock_dest = MagicMock()
        mock_dest.gcs_uri = "gs://test-bucket/batch-results/batch-123/"
        mock_batch_job.dest = mock_dest
        mock_client.client.batches.get.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        # Mock list_gcs_blobs to return a predictions file
        mock_list_blobs.return_value = [
            "gs://test-bucket/batch-results/batch-123/predictions.jsonl"
        ]

        result = get_batch_job_output_uri("projects/test/locations/us-central1/batchJobs/123")

        assert result == "gs://test-bucket/batch-results/batch-123/predictions.jsonl"
        mock_client.client.batches.get.assert_called_once_with(name="projects/test/locations/us-central1/batchJobs/123")

    @patch('gemini_batch.batch.GeminiClient')
    def test_get_batch_job_output_uri_no_dest(self, mock_client_class):
        """Test error when job has no destination."""
        mock_client = MagicMock()
        mock_batch_job = MagicMock()
        mock_batch_job.dest = None
        mock_client.client.batches.get.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="has no destination configured"):
            get_batch_job_output_uri("projects/test/batchJobs/123")

    @patch('gemini_batch.batch.GeminiClient')
    def test_get_batch_job_output_uri_not_vertexai(self, mock_client_class):
        """Test error when job is not a Vertex AI job (no GCS destination)."""
        mock_client = MagicMock()
        mock_batch_job = MagicMock()
        mock_dest = MagicMock()
        mock_dest.gcs_uri = None  # Not a Vertex AI job
        mock_dest.file_name = "files/results-123"  # Has File API destination instead
        mock_batch_job.dest = mock_dest
        mock_client.client.batches.get.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="not a Vertex AI job"):
            get_batch_job_output_uri("batches/test-job")

    @patch('gemini_batch.batch.list_gcs_blobs')
    @patch('gemini_batch.batch.GeminiClient')
    def test_get_batch_job_output_uri_default_path(self, mock_client_class, mock_list_blobs):
        """Test fallback to default predictions.jsonl path when no files found."""
        mock_client = MagicMock()
        mock_client.gcs_bucket = "test-bucket"

        mock_batch_job = MagicMock()
        mock_dest = MagicMock()
        mock_dest.gcs_uri = "gs://test-bucket/batch-results/batch-456/"
        mock_batch_job.dest = mock_dest
        mock_client.client.batches.get.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        # Mock list_gcs_blobs to return empty list
        mock_list_blobs.return_value = []

        result = get_batch_job_output_uri("projects/test/batchJobs/456")

        # Should return default path
        assert result == "gs://test-bucket/batch-results/batch-456/predictions.jsonl"

    @patch('gemini_batch.batch.list_gcs_blobs')
    @patch('gemini_batch.batch.GeminiClient')
    def test_get_batch_job_output_uri_different_bucket(self, mock_client_class, mock_list_blobs):
        """Test that get_batch_job_output_uri works when job bucket differs from client bucket.
        
        This is a regression test - previously the code used client.gcs_bucket to strip
        the bucket prefix, but the bucket might be different or client.gcs_bucket might be None.
        """
        mock_client = MagicMock()
        mock_client.vertexai = True
        # Client has different bucket (or could be None)
        mock_client.gcs_bucket = "client-default-bucket"

        mock_batch_job = MagicMock()
        mock_dest = MagicMock()
        # Job output is in a different bucket
        mock_dest.gcs_uri = "gs://job-output-bucket/batch-results/batch-789/"
        mock_batch_job.dest = mock_dest
        mock_client.client.batches.get.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        # Mock list_gcs_blobs to return a predictions file in timestamped subfolder
        mock_list_blobs.return_value = [
            "gs://job-output-bucket/batch-results/batch-789/2024-01-17_12:34:56/predictions.jsonl"
        ]

        result = get_batch_job_output_uri("projects/test/batchJobs/789")

        assert result == "gs://job-output-bucket/batch-results/batch-789/2024-01-17_12:34:56/predictions.jsonl"
        # Verify list_gcs_blobs was called with the correct bucket from the job's GCS URI
        mock_list_blobs.assert_called_once()
        call_args = mock_list_blobs.call_args
        assert call_args.kwargs['bucket_name'] == "job-output-bucket"
        assert call_args.kwargs['prefix'] == "batch-results/batch-789/"

    @patch('gemini_batch.batch.get_batch_job_output_uri')
    @patch('gemini_batch.batch.GeminiClient')
    @patch('time.time', return_value=1700000000)
    def test_resume_batch_job_with_job_name(self, mock_time, mock_client_class, mock_get_output_uri):
        """Test resuming a batch job using job name lookup."""
        mock_client = MagicMock()
        mock_client.vertexai = True
        mock_client.ensure_gcs_bucket.return_value = "test-bucket"

        mock_new_job = MagicMock()
        mock_new_job.name = "projects/test/locations/us-central1/batchJobs/456"
        mock_client.client.batches.create.return_value = mock_new_job
        mock_client_class.return_value = mock_client

        # Mock the output URI lookup
        mock_get_output_uri.return_value = "gs://test-bucket/batch-results/batch-123/predictions.jsonl"

        result = resume_batch_job("projects/test/locations/us-central1/batchJobs/123")

        assert result == "projects/test/locations/us-central1/batchJobs/456"
        mock_get_output_uri.assert_called_once()
        mock_client.client.batches.create.assert_called_once()

        # Verify the create call used the output URI as source
        create_call = mock_client.client.batches.create.call_args
        assert create_call.kwargs['src'] == "gs://test-bucket/batch-results/batch-123/predictions.jsonl"

    @patch('gemini_batch.batch.GeminiClient')
    @patch('time.time', return_value=1700000000)
    def test_resume_batch_job_with_gcs_uri(self, mock_time, mock_client_class):
        """Test resuming a batch job using direct GCS URI."""
        mock_client = MagicMock()
        mock_client.vertexai = True
        mock_client.ensure_gcs_bucket.return_value = "test-bucket"

        mock_new_job = MagicMock()
        mock_new_job.name = "projects/test/locations/us-central1/batchJobs/789"
        mock_client.client.batches.create.return_value = mock_new_job
        mock_client_class.return_value = mock_client

        # Use direct GCS URI
        result = resume_batch_job("gs://my-bucket/results/predictions.jsonl")

        assert result == "projects/test/locations/us-central1/batchJobs/789"
        mock_client.client.batches.create.assert_called_once()

        # Verify the create call used the GCS URI directly as source
        create_call = mock_client.client.batches.create.call_args
        assert create_call.kwargs['src'] == "gs://my-bucket/results/predictions.jsonl"

    @patch('gemini_batch.batch.GeminiClient')
    def test_resume_batch_job_not_vertexai_error(self, mock_client_class):
        """Test error when trying to resume without Vertex AI backend."""
        mock_client = MagicMock()
        mock_client.vertexai = False  # Not using Vertex AI
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="only available with Vertex AI"):
            resume_batch_job("projects/test/batchJobs/123")

    @patch('gemini_batch.batch.GeminiClient')
    @patch('time.time', return_value=1700000000)
    def test_resume_batch_job_custom_display_name(self, mock_time, mock_client_class):
        """Test resuming with custom display name."""
        mock_client = MagicMock()
        mock_client.vertexai = True
        mock_client.ensure_gcs_bucket.return_value = "test-bucket"

        mock_new_job = MagicMock()
        mock_new_job.name = "projects/test/batchJobs/456"
        mock_client.client.batches.create.return_value = mock_new_job
        mock_client_class.return_value = mock_client

        result = resume_batch_job(
            "gs://bucket/predictions.jsonl",
            job_display_name="my-resumed-job"
        )

        # Verify custom display name was used
        create_call = mock_client.client.batches.create.call_args
        config = create_call.kwargs['config']
        assert config.display_name == "my-resumed-job"

    @patch('gemini_batch.batch.GeminiClient')
    @patch('time.time', return_value=1700000000)
    def test_resume_batch_job_custom_output_path(self, mock_time, mock_client_class):
        """Test resuming with custom GCS output path."""
        mock_client = MagicMock()
        mock_client.vertexai = True
        mock_client.ensure_gcs_bucket.return_value = "test-bucket"

        mock_new_job = MagicMock()
        mock_new_job.name = "projects/test/batchJobs/456"
        mock_client.client.batches.create.return_value = mock_new_job
        mock_client_class.return_value = mock_client

        result = resume_batch_job(
            "gs://bucket/predictions.jsonl",
            gcs_output_path="gs://custom-bucket/custom-output/"
        )

        # Verify custom output path was used
        create_call = mock_client.client.batches.create.call_args
        config = create_call.kwargs['config']
        assert config.dest == "gs://custom-bucket/custom-output/"


class TestBatchProcessResumeFrom:
    """Tests for batch_process with resume_from parameter."""

    @patch('gemini_batch.batch.resume_batch_job')
    @patch('gemini_batch.batch.monitor_batch_job')
    @patch('gemini_batch.batch.download_batch_results')
    @patch('gemini_batch.batch.parse_batch_results')
    @patch('gemini_batch.utils.GeminiClient')
    def test_batch_process_resume_from_wait(
        self, mock_client_class, mock_parse, mock_download, mock_monitor, mock_resume
    ):
        """Test batch_process with resume_from and wait=True."""
        from gemini_batch import batch_process

        mock_client = MagicMock()
        mock_client.vertexai = True
        mock_client_class.return_value = mock_client

        mock_resume.return_value = "projects/test/batchJobs/456"
        mock_monitor.return_value = "JOB_STATE_SUCCEEDED"
        mock_download.return_value = "/tmp/results.jsonl"
        mock_parse.return_value = [TestSchema(name="Test", value=1)]

        results = batch_process(
            prompts=[],  # Empty when resuming
            schema=TestSchema,
            resume_from="projects/test/batchJobs/123",
            vertexai=True,
        )

        assert len(results) == 1
        assert results[0].name == "Test"
        mock_resume.assert_called_once()
        mock_monitor.assert_called_once()
        mock_download.assert_called_once()
        mock_parse.assert_called_once()

    @patch('gemini_batch.batch.resume_batch_job')
    @patch('gemini_batch.utils.GeminiClient')
    def test_batch_process_resume_from_no_wait(self, mock_client_class, mock_resume):
        """Test batch_process with resume_from and wait=False."""
        from gemini_batch import batch_process

        mock_client = MagicMock()
        mock_client.vertexai = True
        mock_client_class.return_value = mock_client

        mock_resume.return_value = "projects/test/batchJobs/456"

        result = batch_process(
            prompts=[],
            schema=TestSchema,
            resume_from="projects/test/batchJobs/123",
            vertexai=True,
            wait=False,
        )

        # Should return job name when wait=False
        assert result == "projects/test/batchJobs/456"
        mock_resume.assert_called_once()

    @patch('gemini_batch.utils.GeminiClient')
    def test_batch_process_resume_from_not_vertexai_error(self, mock_client_class):
        """Test error when using resume_from without Vertex AI backend."""
        from gemini_batch import batch_process

        mock_client = MagicMock()
        mock_client.vertexai = False  # Not using Vertex AI
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="only supported with Vertex AI"):
            batch_process(
                prompts=[],
                schema=TestSchema,
                resume_from="projects/test/batchJobs/123",
            )

    @patch('gemini_batch.batch.resume_batch_job')
    @patch('gemini_batch.batch.monitor_batch_job')
    @patch('gemini_batch.utils.GeminiClient')
    def test_batch_process_resume_from_job_failed(
        self, mock_client_class, mock_monitor, mock_resume
    ):
        """Test error when resumed job fails."""
        from gemini_batch import batch_process

        mock_client = MagicMock()
        mock_client.vertexai = True
        mock_client_class.return_value = mock_client

        mock_resume.return_value = "projects/test/batchJobs/456"
        mock_monitor.return_value = "JOB_STATE_FAILED"  # Job failed

        with pytest.raises(RuntimeError, match="Resumed batch job failed"):
            batch_process(
                prompts=[],
                schema=TestSchema,
                resume_from="projects/test/batchJobs/123",
                vertexai=True,
            )


class TestBatchProcessWithKeys:
    """Tests for batch_process with custom keys parameter."""

    @patch('gemini_batch.utils.GeminiClient')
    def test_batch_process_with_custom_keys(self, mock_client_class, tmp_path):
        """Test that custom keys are included in request keys."""
        from gemini_batch import batch_process

        mock_client = MagicMock()
        mock_client.vertexai = False

        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/test-job"
        mock_client.client.batches.create.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        prompts = [["Text prompt 1"], ["Text prompt 2"]]
        keys = ["doc_001", "doc_002"]

        try:
            batch_process(
                prompts=prompts,
                keys=keys,
                schema=TestSchema,
                wait=False,
                jsonl_dir=str(tmp_path),
            )
        except Exception:
            pass

        # Find and read the JSONL file
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) > 0, "No JSONL file was created"

        jsonl_content = jsonl_files[0].read_text().strip().split('\n')
        assert len(jsonl_content) == 2

        # Verify request keys contain custom keys in format {i}_{key}
        request1 = json.loads(jsonl_content[0])
        request2 = json.loads(jsonl_content[1])

        assert request1["key"] == "0_doc_001"
        assert request2["key"] == "1_doc_002"

    def test_batch_process_keys_length_mismatch(self):
        """Test that ValueError is raised when keys length doesn't match prompts."""
        from gemini_batch import batch_process

        prompts = [["Text 1"], ["Text 2"], ["Text 3"]]
        keys = ["key1", "key2"]  # Only 2 keys for 3 prompts

        with pytest.raises(ValueError) as exc_info:
            batch_process(
                prompts=prompts,
                keys=keys,
                schema=TestSchema,
                wait=False,
            )

        assert "Length of 'keys'" in str(exc_info.value)
        assert "2" in str(exc_info.value)
        assert "3" in str(exc_info.value)

    def test_batch_process_duplicate_keys(self):
        """Test that ValueError is raised for duplicate keys."""
        from gemini_batch import batch_process

        prompts = [["Text 1"], ["Text 2"], ["Text 3"]]
        keys = ["key1", "key2", "key1"]  # Duplicate "key1"

        with pytest.raises(ValueError) as exc_info:
            batch_process(
                prompts=prompts,
                keys=keys,
                schema=TestSchema,
                wait=False,
            )

        assert "unique" in str(exc_info.value)

    def test_batch_process_keys_with_resume_from_error(self):
        """Test that ValueError is raised when both keys and resume_from are provided."""
        from gemini_batch import batch_process

        prompts = [["Text 1"]]
        keys = ["key1"]

        with pytest.raises(ValueError) as exc_info:
            batch_process(
                prompts=prompts,
                keys=keys,
                schema=TestSchema,
                resume_from="gs://bucket/results.jsonl",
                vertexai=True,
                wait=False,
            )

        assert "'keys' parameter cannot be used with 'resume_from'" in str(exc_info.value)

    def test_parse_batch_results_with_custom_keys(self, tmp_path):
        """Test that parse_batch_results correctly sorts results with custom key suffixes."""
        # Create results file with out-of-order results (as Batch API may return)
        results_file = tmp_path / "results.jsonl"
        results_data = [
            # Results in random order
            {
                "key": "2_doc_C",
                "response": {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": '{"name": "Third", "value": 3}'
                            }]
                        }
                    }]
                }
            },
            {
                "key": "0_doc_A",
                "response": {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": '{"name": "First", "value": 1}'
                            }]
                        }
                    }]
                }
            },
            {
                "key": "1_doc_B",
                "response": {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": '{"name": "Second", "value": 2}'
                            }]
                        }
                    }]
                }
            },
        ]

        with open(results_file, 'w') as f:
            for line in results_data:
                f.write(json.dumps(line) + '\n')

        parsed = parse_batch_results(str(results_file), TestSchema)

        # Results should be sorted by index (extracted from {i}_{key})
        assert len(parsed) == 3
        assert parsed[0].name == "First"
        assert parsed[0].value == 1
        assert parsed[1].name == "Second"
        assert parsed[1].value == 2
        assert parsed[2].name == "Third"
        assert parsed[2].value == 3

    @patch('gemini_batch.utils.GeminiClient')
    def test_batch_process_without_keys(self, mock_client_class, tmp_path):
        """Test that batch_process works as before when keys is None."""
        from gemini_batch import batch_process

        mock_client = MagicMock()
        mock_client.vertexai = False

        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/test-job"
        mock_client.client.batches.create.return_value = mock_batch_job
        mock_client_class.return_value = mock_client

        prompts = [["Text prompt 1"], ["Text prompt 2"]]

        try:
            batch_process(
                prompts=prompts,
                schema=TestSchema,
                wait=False,
                jsonl_dir=str(tmp_path),
            )
        except Exception:
            pass

        # Find and read the JSONL file
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) > 0

        jsonl_content = jsonl_files[0].read_text().strip().split('\n')
        assert len(jsonl_content) == 2

        # Verify request keys use default format {i}
        request1 = json.loads(jsonl_content[0])
        request2 = json.loads(jsonl_content[1])

        assert request1["key"] == "0"
        assert request2["key"] == "1"
