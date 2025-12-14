"""Tests for async batch processing functionality."""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json
from pathlib import Path
from PIL import Image
import io
from pydantic import BaseModel

from gemini_batch.async_batch import (
    AsyncOpenAIClient,
    detect_mime_type,
    encode_media_for_openai,
    build_openai_messages,
    process_single_request,
    process_single_request_with_retry,
    async_process,
    process,
)


class TestSchema(BaseModel):
    name: str
    value: int


class TestDetectMimeType:
    """Tests for MIME type detection from magic bytes."""

    def test_detect_png(self):
        """Test PNG detection."""
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        assert detect_mime_type(png_bytes) == "image/png"

    def test_detect_jpeg(self):
        """Test JPEG detection."""
        jpeg_bytes = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert detect_mime_type(jpeg_bytes) == "image/jpeg"

    def test_detect_gif87a(self):
        """Test GIF87a detection."""
        gif_bytes = b'GIF87a' + b'\x00' * 100
        assert detect_mime_type(gif_bytes) == "image/gif"

    def test_detect_gif89a(self):
        """Test GIF89a detection."""
        gif_bytes = b'GIF89a' + b'\x00' * 100
        assert detect_mime_type(gif_bytes) == "image/gif"

    def test_detect_webp(self):
        """Test WebP detection."""
        webp_bytes = b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 100
        assert detect_mime_type(webp_bytes) == "image/webp"

    def test_detect_unknown(self):
        """Test fallback to PNG for unknown formats."""
        unknown_bytes = b'\x00\x01\x02\x03' + b'\x00' * 100
        assert detect_mime_type(unknown_bytes) == "image/png"


class TestEncodeMediaForOpenAI:
    """Tests for media encoding."""

    def test_encode_path_png(self, tmp_path):
        """Test encoding PNG from file path."""
        # Create a simple PNG file
        img = Image.new('RGB', (10, 10), color='red')
        img_path = tmp_path / "test.png"
        img.save(img_path, format='PNG')

        result = encode_media_for_openai(img_path)

        assert result["type"] == "image_url"
        assert "image_url" in result
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_encode_pil_image(self):
        """Test encoding PIL Image."""
        img = Image.new('RGB', (10, 10), color='blue')

        result = encode_media_for_openai(img)

        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_encode_pil_rgba_image(self):
        """Test encoding RGBA PIL Image (converted to RGB)."""
        img = Image.new('RGBA', (10, 10), color=(255, 0, 0, 128))

        result = encode_media_for_openai(img)

        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_encode_bytes(self):
        """Test encoding raw bytes."""
        # Create PNG bytes
        img = Image.new('RGB', (10, 10), color='green')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()

        result = encode_media_for_openai(img_bytes)

        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_encode_invalid_path(self, tmp_path):
        """Test error for non-existent file."""
        fake_path = tmp_path / "nonexistent.png"

        with pytest.raises(ValueError, match="File not found"):
            encode_media_for_openai(fake_path)


class TestBuildOpenAIMessages:
    """Tests for prompt to messages conversion."""

    def test_text_only_prompt(self):
        """Test converting text-only prompt."""
        prompt_parts = ["Hello, world!"]

        messages = build_openai_messages(prompt_parts)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 1
        assert messages[0]["content"][0] == {"type": "text", "text": "Hello, world!"}

    def test_multiple_text_parts(self):
        """Test converting multiple text parts."""
        prompt_parts = ["Part 1", "Part 2", "Part 3"]

        messages = build_openai_messages(prompt_parts)

        assert len(messages) == 1
        assert len(messages[0]["content"]) == 3
        assert messages[0]["content"][0]["text"] == "Part 1"
        assert messages[0]["content"][1]["text"] == "Part 2"
        assert messages[0]["content"][2]["text"] == "Part 3"

    def test_with_schema_adds_system_message(self):
        """Test that schema adds system message."""
        prompt_parts = ["What is 2+2?"]

        messages = build_openai_messages(prompt_parts, schema=TestSchema)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "JSON" in messages[0]["content"]
        assert "name" in messages[0]["content"]
        assert "value" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_mixed_content_prompt(self, tmp_path):
        """Test converting mixed text and image prompt."""
        # Create test image
        img = Image.new('RGB', (10, 10), color='red')
        img_path = tmp_path / "test.png"
        img.save(img_path, format='PNG')

        prompt_parts = ["Analyze this image:", img_path, "What do you see?"]

        messages = build_openai_messages(prompt_parts)

        assert len(messages) == 1
        assert len(messages[0]["content"]) == 3
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][1]["type"] == "image_url"
        assert messages[0]["content"][2]["type"] == "text"


class TestAsyncOpenAIClient:
    """Tests for AsyncOpenAIClient class."""

    def test_client_initialization(self):
        """Test client initializes with correct settings."""
        client = AsyncOpenAIClient(
            api_key="test-key",
            base_url="https://api.test.com",
            max_concurrent=5,
            timeout=30.0,
        )

        assert client.api_key == "test-key"
        assert client.base_url == "https://api.test.com"
        assert client._timeout == 30.0
        assert client._client is None  # Lazy initialization

    def test_api_key_from_env(self, monkeypatch):
        """Test API key fallback to environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")

        client = AsyncOpenAIClient()

        assert client.api_key == "env-test-key"

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that semaphore properly limits concurrent requests."""
        client = AsyncOpenAIClient(
            api_key="test-key",
            max_concurrent=2,
        )

        # Track concurrent operations
        concurrent_count = 0
        max_concurrent_seen = 0

        async def track_concurrency():
            nonlocal concurrent_count, max_concurrent_seen
            async with client._semaphore:
                concurrent_count += 1
                max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
                await asyncio.sleep(0.1)
                concurrent_count -= 1

        # Run 5 tasks with max_concurrent=2
        await asyncio.gather(*[track_concurrency() for _ in range(5)])

        assert max_concurrent_seen <= 2


class TestProcessSingleRequest:
    """Tests for single request processing."""

    @pytest.mark.asyncio
    async def test_basic_text_response(self):
        """Test basic text response processing."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "test-model"

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(return_value=mock_response)

        result, metadata = await process_single_request(
            client=mock_client,
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            schema=None,
        )

        assert result == "Hello!"
        assert metadata["usageMetadata"]["promptTokenCount"] == 10
        assert metadata["usageMetadata"]["candidatesTokenCount"] == 5
        assert metadata["usageMetadata"]["totalTokenCount"] == 15

    @pytest.mark.asyncio
    async def test_structured_output_parsing(self):
        """Test structured output with Pydantic parsing."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "test", "value": 42}'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "test-model"

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(return_value=mock_response)

        result, metadata = await process_single_request(
            client=mock_client,
            messages=[{"role": "user", "content": "Generate"}],
            model="test-model",
            schema=TestSchema,
        )

        assert isinstance(result, TestSchema)
        assert result.name == "test"
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_json_in_markdown_block(self):
        """Test extracting JSON from markdown code block."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"name": "extracted", "value": 99}\n```'
        mock_response.usage = None
        mock_response.model = "test-model"

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(return_value=mock_response)

        result, metadata = await process_single_request(
            client=mock_client,
            messages=[{"role": "user", "content": "Generate"}],
            model="test-model",
            schema=TestSchema,
        )

        assert isinstance(result, TestSchema)
        assert result.name == "extracted"
        assert result.value == 99

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        """Test that invalid JSON returns None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not JSON at all"
        mock_response.usage = None
        mock_response.model = "test-model"

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(return_value=mock_response)

        result, metadata = await process_single_request(
            client=mock_client,
            messages=[{"role": "user", "content": "Generate"}],
            model="test-model",
            schema=TestSchema,
        )

        assert result is None


class TestProcessSingleRequestWithRetry:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Test successful response on first attempt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success"
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(return_value=mock_response)

        result, metadata = await process_single_request_with_retry(
            client=mock_client,
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            schema=None,
            retry_count=3,
            retry_delay=0.01,
        )

        assert result == "Success"
        assert mock_client.create_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry logic on transient failures."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success after retry"
        mock_response.usage = None

        mock_client = AsyncMock()
        # Fail twice, then succeed
        mock_client.create_completion = AsyncMock(
            side_effect=[
                Exception("Temporary error"),
                Exception("Another error"),
                mock_response,
            ]
        )

        result, metadata = await process_single_request_with_retry(
            client=mock_client,
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            schema=None,
            retry_count=3,
            retry_delay=0.01,
        )

        assert result == "Success after retry"
        assert mock_client.create_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Test that exhausted retries return None."""
        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Persistent error")
        )

        result, metadata = await process_single_request_with_retry(
            client=mock_client,
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            schema=None,
            retry_count=2,
            retry_delay=0.01,
        )

        assert result is None
        assert metadata is None
        assert mock_client.create_completion.call_count == 3  # Initial + 2 retries


class TestAsyncProcess:
    """Tests for main async_process function."""

    @pytest.mark.asyncio
    async def test_empty_prompts(self):
        """Test handling of empty prompts list."""
        results = await async_process(prompts=[], api_key="test-key")
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_prompts_with_metadata(self):
        """Test handling of empty prompts list with return_metadata."""
        results, metadata = await async_process(prompts=[], return_metadata=True, api_key="test-key")
        assert results == []
        assert metadata == []

    @pytest.mark.asyncio
    async def test_basic_processing(self):
        """Test basic text prompt processing."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Result 1"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "test-model"

        with patch('openai.AsyncOpenAI') as mock_openai_class:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_openai

            results = await async_process(
                prompts=[["What is 2+2?"]],
                model="test-model",
                api_key="test-key",
            )

            assert len(results) == 1
            assert results[0] == "Result 1"

    @pytest.mark.asyncio
    async def test_structured_output(self):
        """Test structured output with schema."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "test", "value": 42}'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "test-model"

        with patch('openai.AsyncOpenAI') as mock_openai_class:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_openai

            results = await async_process(
                prompts=[["Generate data"]],
                schema=TestSchema,
                model="test-model",
                api_key="test-key",
            )

            assert len(results) == 1
            assert isinstance(results[0], TestSchema)
            assert results[0].name == "test"
            assert results[0].value == 42

    @pytest.mark.asyncio
    async def test_return_metadata(self):
        """Test metadata is returned when requested."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "test-model"

        with patch('openai.AsyncOpenAI') as mock_openai_class:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_openai

            results, metadata = await async_process(
                prompts=[["Hello"]],
                model="test-model",
                api_key="test-key",
                return_metadata=True,
            )

            assert len(results) == 1
            assert len(metadata) == 1
            assert metadata[0]["usageMetadata"]["totalTokenCount"] == 15

    @pytest.mark.asyncio
    async def test_multiple_prompts(self):
        """Test processing multiple prompts concurrently."""
        responses = []
        for i in range(3):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = f"Result {i}"
            mock_response.usage = None
            responses.append(mock_response)

        with patch('openai.AsyncOpenAI') as mock_openai_class:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = AsyncMock(side_effect=responses)
            mock_openai_class.return_value = mock_openai

            results = await async_process(
                prompts=[["Prompt 1"], ["Prompt 2"], ["Prompt 3"]],
                model="test-model",
                api_key="test-key",
            )

            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_failed_request_returns_none(self):
        """Test that failed requests return None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success"
        mock_response.usage = None

        with patch('openai.AsyncOpenAI') as mock_openai_class:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = AsyncMock(
                side_effect=[
                    mock_response,
                    Exception("API Error"),
                    mock_response,
                ]
            )
            mock_openai_class.return_value = mock_openai

            results = await async_process(
                prompts=[["P1"], ["P2"], ["P3"]],
                model="test-model",
                api_key="test-key",
                retry_count=0,  # Disable retries
            )

            assert len(results) == 3
            assert results[0] == "Success"
            assert results[1] is None  # Failed request
            assert results[2] == "Success"


class TestSyncProcess:
    """Tests for sync_async_process wrapper."""

    def test_sync_wrapper_calls_async(self):
        """Test sync wrapper properly calls async function."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Sync result"
        mock_response.usage = None

        with patch('openai.AsyncOpenAI') as mock_openai_class:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_openai

            results = process(
                prompts=[["Hello"]],
                model="test-model",
                api_key="test-key",
            )

            assert len(results) == 1
            assert results[0] == "Sync result"
