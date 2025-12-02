"""Tests for utility functions."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from PIL import Image
import io

from gemini_batch import utils
from gemini_batch.utils import GeminiClient, upload_file_to_gemini, build_generation_config
from pydantic import BaseModel


class DummySchema(BaseModel):
    name: str
    value: int


def test_get_api_key_from_env(monkeypatch):
    """Test API key retrieval from environment."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    assert utils.get_api_key() == "test-key-123"


def test_get_api_key_missing(monkeypatch):
    """Test error when API key is missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        utils.get_api_key()


def test_gemini_client_init(monkeypatch):
    """Test GeminiClient initialization."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = GeminiClient()
    assert client.api_key == "test-key"


def test_gemini_client_with_explicit_key():
    """Test GeminiClient with explicit API key."""
    client = GeminiClient(api_key="explicit-key")
    assert client.api_key == "explicit-key"


def test_upload_file_to_gemini_from_path(tmp_path):
    """Test file upload from Path object."""
    # Create a test image
    img = Image.new('RGB', (100, 100), color='red')
    img_path = tmp_path / "test.png"
    img.save(img_path)

    # Mock client
    mock_client = Mock()
    mock_uploaded = Mock()
    mock_uploaded.uri = "gs://bucket/file.png"
    mock_uploaded.mime_type = "image/png"
    mock_uploaded.name = "files/test"
    mock_client.files.upload.return_value = mock_uploaded

    # Upload to Gemini with Path object
    result = upload_file_to_gemini(img_path, mock_client)

    assert result["uri"] == "gs://bucket/file.png"
    assert result["mime_type"] == "image/png"
    mock_client.files.upload.assert_called_once_with(file=str(img_path))


def test_upload_file_to_gemini_from_pil():
    """Test file upload from PIL Image."""
    img = Image.new('RGB', (100, 100), color='blue')

    # Mock client
    mock_client = Mock()
    mock_uploaded = Mock()
    mock_uploaded.uri = "gs://bucket/image.png"
    mock_uploaded.mime_type = "image/png"
    mock_uploaded.name = "files/img"
    mock_client.files.upload.return_value = mock_uploaded

    # Upload to Gemini
    result = upload_file_to_gemini(img, mock_client)

    assert result["uri"] == "gs://bucket/image.png"
    assert result["mime_type"] == "image/png"
    mock_client.files.upload.assert_called_once()


def test_upload_file_to_gemini_from_bytes():
    """Test file upload from bytes."""
    img = Image.new('RGB', (100, 100), color='green')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()

    # Mock client
    mock_client = Mock()
    mock_uploaded = Mock()
    mock_uploaded.uri = "gs://bucket/bytes.png"
    mock_uploaded.mime_type = "image/png"
    mock_uploaded.name = "files/bytes"
    mock_client.files.upload.return_value = mock_uploaded

    # Upload to Gemini
    result = upload_file_to_gemini(img_bytes, mock_client)

    assert result["uri"] == "gs://bucket/bytes.png"
    assert result["mime_type"] == "image/png"
    mock_client.files.upload.assert_called_once()


def test_upload_file_invalid_path():
    """Test error handling for invalid file path."""
    from pathlib import Path
    mock_client = Mock()
    with pytest.raises(ValueError, match="File not found"):
        upload_file_to_gemini(Path("/nonexistent/path.jpg"), mock_client)


def test_upload_file_invalid_type():
    """Test error handling for unsupported file type."""
    mock_client = Mock()
    with pytest.raises(ValueError, match="Unsupported file type"):
        upload_file_to_gemini(12345, mock_client)


def test_build_generation_config_basic():
    """Test basic generation config building."""
    config = build_generation_config()

    assert config.temperature is not None
    assert config.max_output_tokens is not None


def test_build_generation_config_with_schema():
    """Test generation config with Pydantic schema."""
    config = build_generation_config(response_schema=DummySchema)

    assert config.response_mime_type == "application/json"
    assert config.response_json_schema is not None


def test_build_generation_config_with_thinking():
    """Test generation config with thinking budget."""
    config = build_generation_config(thinking_budget=1000)

    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 1000


def test_build_generation_config_custom_params():
    """Test generation config with custom parameters."""
    config = build_generation_config(
        temperature=0.9,
        max_output_tokens=4096,
        top_p=0.95,
        top_k=40
    )

    assert config.temperature == 0.9
    assert config.max_output_tokens == 4096
    assert config.top_p == 0.95
    assert config.top_k == 40


def test_build_generation_config_with_media_resolution():
    """Test generation config with media_resolution parameter."""
    config = build_generation_config(
        media_resolution="MEDIA_RESOLUTION_HIGH"
    )

    assert config.media_resolution == "MEDIA_RESOLUTION_HIGH"


def test_build_generation_config_with_media_resolution_and_schema():
    """Test generation config with both media_resolution and response schema."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        value: int

    config = build_generation_config(
        response_schema=TestSchema,
        media_resolution="MEDIA_RESOLUTION_MEDIUM"
    )

    assert config.media_resolution == "MEDIA_RESOLUTION_MEDIUM"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema is not None


@patch('gemini_batch.utils.fitz')
def test_pdf_pages_to_images(mock_fitz, tmp_path):
    """Test PDF to images conversion."""
    # Create mock PDF
    mock_doc = MagicMock()
    mock_doc.page_count = 2
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.get_images.return_value = []
    mock_doc.load_page.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    # Mock pixmap
    mock_pix = MagicMock()
    mock_pix.width = 100
    mock_pix.height = 100
    mock_pix.n = 3
    mock_pix.samples = b'\x00' * (100 * 100 * 3)
    mock_page.get_pixmap.return_value = mock_pix

    # Create a dummy PDF file
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b'dummy pdf content')

    images = utils.pdf_pages_to_images(str(pdf_path), max_pages=1)

    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_pdf_pages_to_images_file_not_found():
    """Test error handling for missing PDF file."""
    with pytest.raises(FileNotFoundError):
        utils.pdf_pages_to_images("/nonexistent/file.pdf")


def test_extract_json_from_text_plain_json():
    """Test extracting plain JSON without any wrapper text."""
    json_str = '{"name": "test", "value": 42}'
    result = utils.extract_json_from_text(json_str)
    assert result == json_str


def test_extract_json_from_text_markdown_code_block():
    """Test extracting JSON from markdown code block."""
    text = '```json\n{"name": "test", "value": 42}\n```'
    result = utils.extract_json_from_text(text)
    assert result == '{"name": "test", "value": 42}'


def test_extract_json_from_text_markdown_code_block_no_language():
    """Test extracting JSON from markdown code block without language specifier."""
    text = '```\n{"name": "test", "value": 42}\n```'
    result = utils.extract_json_from_text(text)
    assert result == '{"name": "test", "value": 42}'


def test_extract_json_from_text_with_prefix():
    """Test extracting JSON with explanatory prefix text."""
    text = 'Here is the extracted data: {"name": "test", "value": 42}'
    result = utils.extract_json_from_text(text)
    assert result == '{"name": "test", "value": 42}'


def test_extract_json_from_text_with_prefix_and_suffix():
    """Test extracting JSON with both prefix and suffix text."""
    text = 'Here is the data: {"name": "test", "value": 42} and that\'s all!'
    result = utils.extract_json_from_text(text)
    assert result == '{"name": "test", "value": 42}'


def test_extract_json_from_text_array():
    """Test extracting JSON array."""
    text = 'Results: [{"id": 1}, {"id": 2}]'
    result = utils.extract_json_from_text(text)
    assert result == '[{"id": 1}, {"id": 2}]'


def test_extract_json_from_text_nested_objects():
    """Test extracting nested JSON objects."""
    text = '{"outer": {"inner": {"deep": "value"}}, "count": 3}'
    result = utils.extract_json_from_text(text)
    assert result == text


def test_extract_json_from_text_with_escaped_quotes():
    """Test extracting JSON with escaped quotes in strings."""
    json_str = '{"message": "He said \\"hello\\""}'
    result = utils.extract_json_from_text(json_str)
    assert result == json_str


def test_extract_json_from_text_no_json():
    """Test when text contains no JSON structure."""
    text = "This is just plain text with no JSON"
    result = utils.extract_json_from_text(text)
    assert result is None


def test_extract_json_from_text_empty_string():
    """Test with empty string."""
    result = utils.extract_json_from_text("")
    assert result is None


def test_extract_json_from_text_none():
    """Test with None input."""
    result = utils.extract_json_from_text(None)
    assert result is None


def test_extract_json_from_text_multiline_markdown():
    """Test extracting JSON from multiline markdown with indentation."""
    text = '''```json
{
  "name": "test",
  "value": 42,
  "nested": {
    "key": "value"
  }
}
```'''
    result = utils.extract_json_from_text(text)
    # Should extract the JSON content without the markdown wrapper
    assert '{"name":"test"' in result.replace('\n', '').replace(' ', '')
    assert result.strip().startswith('{')
    assert result.strip().endswith('}')


def test_calculate_token_statistics_all_successful():
    """Test token statistics with all successful requests."""
    from gemini_batch.utils import calculate_token_statistics

    metadata_list = [
        {
            'usageMetadata': {
                'totalTokenCount': 1500,
                'promptTokenCount': 1000,
                'candidatesTokenCount': 500,
                'cachedContentTokenCount': 100,
                'thoughtsTokenCount': 50,
            },
            'modelVersion': 'gemini-2.5-flash'
        },
        {
            'usageMetadata': {
                'totalTokenCount': 2000,
                'promptTokenCount': 1200,
                'candidatesTokenCount': 800,
                'cachedContentTokenCount': 200,
                'thoughtsTokenCount': 100,
            },
            'modelVersion': 'gemini-2.5-flash'
        },
    ]

    stats = calculate_token_statistics(metadata_list)

    assert stats.total_requests == 2
    assert stats.successful_requests == 2
    assert stats.failed_requests == 0
    assert stats.total_prompt_tokens == 2200
    assert stats.total_candidates_tokens == 1300
    assert stats.total_tokens == 3500
    assert stats.total_cached_tokens == 300
    assert stats.total_thoughts_tokens == 150
    assert stats.avg_prompt_tokens == 1100.0
    assert stats.avg_candidates_tokens == 650.0
    assert stats.avg_total_tokens == 1750.0
    assert stats.avg_cached_tokens == 150.0
    assert stats.avg_thoughts_tokens == 75.0


def test_calculate_token_statistics_with_failures():
    """Test token statistics with mix of successful and failed requests."""
    from gemini_batch.utils import calculate_token_statistics

    metadata_list = [
        {
            'usageMetadata': {
                'totalTokenCount': 1500,
                'promptTokenCount': 1000,
                'candidatesTokenCount': 500,
                'cachedContentTokenCount': 0,
                'thoughtsTokenCount': 0,
            },
            'modelVersion': 'gemini-2.5-flash'
        },
        None,  # Failed request
        {
            'usageMetadata': {
                'totalTokenCount': 2000,
                'promptTokenCount': 1200,
                'candidatesTokenCount': 800,
                'cachedContentTokenCount': 0,
                'thoughtsTokenCount': 0,
            },
            'modelVersion': 'gemini-2.5-flash'
        },
        None,  # Failed request
    ]

    stats = calculate_token_statistics(metadata_list)

    assert stats.total_requests == 4
    assert stats.successful_requests == 2
    assert stats.failed_requests == 2
    assert stats.total_prompt_tokens == 2200
    assert stats.total_candidates_tokens == 1300
    assert stats.total_tokens == 3500
    # Averages should be based on successful requests only
    assert stats.avg_prompt_tokens == 1100.0
    assert stats.avg_candidates_tokens == 650.0
    assert stats.avg_total_tokens == 1750.0


def test_calculate_token_statistics_all_failed():
    """Test token statistics with all failed requests."""
    from gemini_batch.utils import calculate_token_statistics

    metadata_list = [None, None, None]

    stats = calculate_token_statistics(metadata_list)

    assert stats.total_requests == 3
    assert stats.successful_requests == 0
    assert stats.failed_requests == 3
    assert stats.total_prompt_tokens == 0
    assert stats.total_candidates_tokens == 0
    assert stats.total_tokens == 0
    assert stats.total_cached_tokens == 0
    assert stats.total_thoughts_tokens == 0
    # Averages should be None when no successful requests
    assert stats.avg_prompt_tokens is None
    assert stats.avg_candidates_tokens is None
    assert stats.avg_total_tokens is None
    assert stats.avg_cached_tokens is None
    assert stats.avg_thoughts_tokens is None


def test_calculate_token_statistics_empty_list():
    """Test token statistics with empty metadata list."""
    from gemini_batch.utils import calculate_token_statistics

    metadata_list = []

    stats = calculate_token_statistics(metadata_list)

    assert stats.total_requests == 0
    assert stats.successful_requests == 0
    assert stats.failed_requests == 0
    assert stats.total_prompt_tokens == 0
    assert stats.total_candidates_tokens == 0
    assert stats.total_tokens == 0
    assert stats.total_cached_tokens == 0
    assert stats.total_thoughts_tokens == 0
    assert stats.avg_prompt_tokens is None
    assert stats.avg_candidates_tokens is None
    assert stats.avg_total_tokens is None
    assert stats.avg_cached_tokens is None
    assert stats.avg_thoughts_tokens is None


def test_calculate_token_statistics_missing_usage_metadata():
    """Test token statistics with missing usageMetadata field."""
    from gemini_batch.utils import calculate_token_statistics

    metadata_list = [
        {
            'usageMetadata': {
                'totalTokenCount': 1000,
                'promptTokenCount': 600,
                'candidatesTokenCount': 400,
                'cachedContentTokenCount': 0,
                'thoughtsTokenCount': 0,
            },
            'modelVersion': 'gemini-2.5-flash'
        },
        {'modelVersion': 'gemini-2.5-flash'},  # Missing usageMetadata
        {
            'usageMetadata': {
                'totalTokenCount': 1500,
                'promptTokenCount': 900,
                'candidatesTokenCount': 600,
                'cachedContentTokenCount': 0,
                'thoughtsTokenCount': 0,
            },
            'modelVersion': 'gemini-2.5-flash'
        },
    ]

    stats = calculate_token_statistics(metadata_list)

    assert stats.total_requests == 3
    assert stats.successful_requests == 2
    assert stats.failed_requests == 1
    assert stats.total_prompt_tokens == 1500
    assert stats.total_candidates_tokens == 1000
    assert stats.total_tokens == 2500
    assert stats.avg_prompt_tokens == 750.0
    assert stats.avg_candidates_tokens == 500.0
    assert stats.avg_total_tokens == 1250.0


def test_calculate_token_statistics_none_token_values():
    """Test token statistics with None values in token counts."""
    from gemini_batch.utils import calculate_token_statistics

    metadata_list = [
        {
            'usageMetadata': {
                'totalTokenCount': 1000,
                'promptTokenCount': 600,
                'candidatesTokenCount': 400,
                'cachedContentTokenCount': None,  # None value
                'thoughtsTokenCount': None,  # None value
            },
            'modelVersion': 'gemini-2.5-flash'
        },
        {
            'usageMetadata': {
                'totalTokenCount': 1500,
                'promptTokenCount': None,  # None value
                'candidatesTokenCount': 600,
                'cachedContentTokenCount': 50,
                'thoughtsTokenCount': 25,
            },
            'modelVersion': 'gemini-2.5-flash'
        },
    ]

    stats = calculate_token_statistics(metadata_list)

    assert stats.total_requests == 2
    assert stats.successful_requests == 2
    assert stats.failed_requests == 0
    # None values should be treated as 0
    assert stats.total_prompt_tokens == 600  # 600 + 0
    assert stats.total_candidates_tokens == 1000  # 400 + 600
    assert stats.total_tokens == 2500  # 1000 + 1500
    assert stats.total_cached_tokens == 50  # 0 + 50
    assert stats.total_thoughts_tokens == 25  # 0 + 25


def test_calculate_token_statistics_partial_token_data():
    """Test token statistics with missing token type keys."""
    from gemini_batch.utils import calculate_token_statistics

    metadata_list = [
        {
            'usageMetadata': {
                'totalTokenCount': 1000,
                'promptTokenCount': 600,
                'candidatesTokenCount': 400,
                # Missing cachedContentTokenCount
                # Missing thoughtsTokenCount
            },
            'modelVersion': 'gemini-2.5-flash'
        },
        {
            'usageMetadata': {
                'totalTokenCount': 1500,
                # Missing promptTokenCount
                'candidatesTokenCount': 600,
                'cachedContentTokenCount': 50,
                # Missing thoughtsTokenCount
            },
            'modelVersion': 'gemini-2.5-flash'
        },
    ]

    stats = calculate_token_statistics(metadata_list)

    assert stats.total_requests == 2
    assert stats.successful_requests == 2
    assert stats.failed_requests == 0
    # Missing keys should be treated as 0
    assert stats.total_prompt_tokens == 600  # 600 + 0
    assert stats.total_candidates_tokens == 1000  # 400 + 600
    assert stats.total_tokens == 2500  # 1000 + 1500
    assert stats.total_cached_tokens == 50  # 0 + 50
    assert stats.total_thoughts_tokens == 0  # 0 + 0


def test_calculate_token_statistics_with_cached_and_thoughts():
    """Test token statistics focusing on cached and thinking tokens."""
    from gemini_batch.utils import calculate_token_statistics

    metadata_list = [
        {
            'usageMetadata': {
                'totalTokenCount': 2000,
                'promptTokenCount': 1000,
                'candidatesTokenCount': 500,
                'cachedContentTokenCount': 300,  # Significant cached tokens
                'thoughtsTokenCount': 200,  # Thinking tokens
            },
            'modelVersion': 'gemini-2.5-flash'
        },
        {
            'usageMetadata': {
                'totalTokenCount': 2500,
                'promptTokenCount': 1200,
                'candidatesTokenCount': 600,
                'cachedContentTokenCount': 400,  # Significant cached tokens
                'thoughtsTokenCount': 300,  # Thinking tokens
            },
            'modelVersion': 'gemini-2.5-flash'
        },
    ]

    stats = calculate_token_statistics(metadata_list)

    assert stats.total_requests == 2
    assert stats.successful_requests == 2
    assert stats.failed_requests == 0
    assert stats.total_cached_tokens == 700
    assert stats.total_thoughts_tokens == 500
    assert stats.avg_cached_tokens == 350.0
    assert stats.avg_thoughts_tokens == 250.0
    # Verify other fields are also correct
    assert stats.total_tokens == 4500
    assert stats.avg_total_tokens == 2250.0


def test_calculate_token_statistics_verbose(capsys):
    """Test token statistics verbose output."""
    from gemini_batch.utils import calculate_token_statistics

    metadata_list = [
        {
            'usageMetadata': {
                'totalTokenCount': 1500,
                'promptTokenCount': 1000,
                'candidatesTokenCount': 500,
                'cachedContentTokenCount': 100,
                'thoughtsTokenCount': 50,
            },
            'modelVersion': 'gemini-2.5-flash'
        },
        {
            'usageMetadata': {
                'totalTokenCount': 2000,
                'promptTokenCount': 1200,
                'candidatesTokenCount': 800,
                'cachedContentTokenCount': 200,
                'thoughtsTokenCount': 100,
            },
            'modelVersion': 'gemini-2.5-flash'
        },
    ]

    stats = calculate_token_statistics(metadata_list, verbose=True)
    captured = capsys.readouterr()

    # Verify the output contains expected content
    assert "TOKEN STATISTICS SUMMARY" in captured.out
    assert "2/2 successful" in captured.out
    assert "Prompt" in captured.out  # More flexible match
    assert "Candidates" in captured.out or "Output" in captured.out  # May vary
    assert "TOTAL" in captured.out or "Total" in captured.out
    assert "3,500" in captured.out or "3500" in captured.out  # Total tokens

    # Verify stats object is still returned correctly
    assert stats.total_requests == 2
    assert stats.successful_requests == 2
    assert stats.total_tokens == 3500


def test_calculate_token_statistics_verbose_no_successful(capsys):
    """Test token statistics verbose output when all requests failed."""
    from gemini_batch.utils import calculate_token_statistics

    metadata_list = [None, None]

    stats = calculate_token_statistics(metadata_list, verbose=True)
    captured = capsys.readouterr()

    # Verify the output shows N/A for averages
    assert "TOKEN STATISTICS SUMMARY" in captured.out
    assert "0/2 successful" in captured.out
    assert "N/A" in captured.out  # Averages should show N/A

    # Verify stats object is still returned correctly
    assert stats.total_requests == 2
    assert stats.successful_requests == 0
    assert stats.avg_total_tokens is None
