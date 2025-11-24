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
@patch('builtins.open', create=True)
def test_create_batch_job_file(mock_open, mock_client_class, tmp_path):
    """Test batch job creation with file-based mode."""
    mock_client = MagicMock()
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
    mock_batch_job = MagicMock()
    mock_state = MagicMock()
    mock_state.name = 'JOB_STATE_SUCCEEDED'
    mock_batch_job.state = mock_state
    mock_batch_job.display_name = "test-results"
    mock_dest = MagicMock()
    mock_dest.file_name = "files/results-123"
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
    """Test parsing results with errors."""
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

    # Only the good result should be parsed
    assert len(parsed) == 1
    assert parsed[0].name == "Good"


def test_parse_batch_results_validation_error(tmp_path):
    """Test parsing with schema validation errors - now non-fatal, returns empty list."""
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

    # Validation errors are now non-fatal - should return empty list instead of raising
    parsed = parse_batch_results(str(results_file), TestSchema, validate=True)
    assert len(parsed) == 0


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
    assert metadata[0]['usageMetadata']['thoughtsTokenCount'] == 5
    assert metadata[0]['modelVersion'] == "gemini-2.5-flash"
    assert metadata[1]['usageMetadata']['totalTokenCount'] == 150
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
    """Test that metadata is skipped for error responses."""
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

    # Only the good result should be parsed
    assert len(parsed) == 1
    assert parsed[0].name == "Good"

    # Only one metadata entry for the successful response
    assert len(metadata) == 1
    assert metadata[0]['usageMetadata']['totalTokenCount'] == 60


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
    """Test that malformed JSONL lines are skipped without crashing."""
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

        # Empty line (should be skipped)
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

    # Should parse successfully, skipping malformed lines
    parsed = parse_batch_results(str(results_file), TestSchema)

    # Should only get the 3 good results
    assert len(parsed) == 3
    assert parsed[0].name == "Good1"
    assert parsed[0].value == 1
    assert parsed[1].name == "Good2"
    assert parsed[1].value == 2
    assert parsed[2].name == "Good3"
    assert parsed[2].value == 3


def test_parse_batch_results_validation_error_non_fatal(tmp_path):
    """Test that schema validation errors don't stop entire batch processing."""
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

    # Should get the 2 good results, skip the bad one
    assert len(parsed) == 2
    assert parsed[0].name == "Good1"
    assert parsed[0].value == 1
    assert parsed[1].name == "Good2"
    assert parsed[1].value == 2


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

    # Should successfully parse despite multiple issues
    parsed = parse_batch_results(str(results_file), TestSchema, validate=True)

    # Should get 4 successful results (req1, req2, req4, req7)
    assert len(parsed) == 4
    assert parsed[0].name == "Good"
    assert parsed[1].name == "Markdown"
    assert parsed[2].name == "Prefixed"
    assert parsed[3].name == "Final"
