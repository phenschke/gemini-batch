"""Shared pytest fixtures for tests."""
import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel


@pytest.fixture
def mock_gemini_client():
    """Create a mock Gemini client."""
    mock = MagicMock()
    mock.api_key = "test-api-key"
    return mock


@pytest.fixture
def sample_schema():
    """Sample Pydantic schema for testing."""
    class SampleSchema(BaseModel):
        name: str
        value: int
        description: str = "default"

    return SampleSchema


@pytest.fixture
def sample_batch_requests():
    """Sample batch request data."""
    return [
        {
            "key": "req1",
            "request": {
                "contents": [{
                    "parts": [{"text": "Test input 1"}]
                }]
            }
        },
        {
            "key": "req2",
            "request": {
                "contents": [{
                    "parts": [{"text": "Test input 2"}]
                }]
            }
        }
    ]


@pytest.fixture
def sample_batch_response():
    """Sample batch response data."""
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": '{"name": "Test", "value": 42, "description": "A test"}'
                }]
            }
        }]
    }
