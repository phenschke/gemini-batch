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
