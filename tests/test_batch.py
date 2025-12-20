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
)


class TestSchema(BaseModel):
    name: str
    value: int


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
