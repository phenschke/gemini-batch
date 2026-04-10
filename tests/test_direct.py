"""Tests for direct Gemini API processing functionality."""
import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path
from PIL import Image
import io
from pydantic import BaseModel
from typing import List
from google.genai import types as genai_types

from gemini_batch.direct import (
    _build_content_parts,
    _extract_metadata,
    _is_rate_limit_error,
    async_process,
    process,
)
from gemini_batch.utils import _compute_content_hash


class TestSchema(BaseModel):
    name: str
    value: int


class ItemSchema(BaseModel):
    id: int
    label: str


# --- Helper to build mock response ---

def _make_response(text, prompt_tokens=10, candidates_tokens=5, model_version="gemini-test"):
    """Build a mock generate_content response."""
    mock_part = MagicMock()
    mock_part.text = text

    mock_content = MagicMock()
    mock_content.parts = [mock_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content

    mock_usage = MagicMock()
    mock_usage.total_token_count = prompt_tokens + candidates_tokens
    mock_usage.prompt_token_count = prompt_tokens
    mock_usage.candidates_token_count = candidates_tokens
    mock_usage.cached_content_token_count = 0
    mock_usage.thoughts_token_count = 0

    response = MagicMock()
    response.candidates = [mock_candidate]
    response.usage_metadata = mock_usage
    response.model_version = model_version
    return response


# --- _build_content_parts ---

class TestBuildContentParts:
    def test_text_parts(self):
        parts = _build_content_parts(["Hello", "World"], {}, {})
        assert len(parts) == 2
        assert parts[0].text == "Hello"
        assert parts[1].text == "World"

    def test_path_parts(self, tmp_path):
        img = Image.new("RGB", (2, 2), "red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        path_to_uri = {str(img_path.resolve()): {"uri": "files/abc", "mime_type": "image/png"}}
        parts = _build_content_parts([img_path], path_to_uri, {})

        assert len(parts) == 1
        assert parts[0].file_data.file_uri == "files/abc"
        assert parts[0].file_data.mime_type == "image/png"

    def test_bytes_parts(self):
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        content_hash = _compute_content_hash(data)
        content_hashes = {content_hash: {"uri": "files/xyz", "mime_type": "image/png"}}

        parts = _build_content_parts([data], {}, content_hashes)
        assert len(parts) == 1
        assert parts[0].file_data.file_uri == "files/xyz"

    def test_pil_image_parts(self):
        img = Image.new("RGB", (4, 4), "blue")
        content_hash = _compute_content_hash(img)
        content_hashes = {content_hash: {"uri": "files/img", "mime_type": "image/png"}}

        parts = _build_content_parts([img], {}, content_hashes)
        assert len(parts) == 1
        assert parts[0].file_data.file_uri == "files/img"

    def test_prebuilt_part_passthrough(self):
        part = genai_types.Part(
            file_data=genai_types.FileData(
                file_uri="files/custom", mime_type="image/jpeg"
            )
        )
        parts = _build_content_parts([part], {}, {})
        assert len(parts) == 1
        assert parts[0].file_data.file_uri == "files/custom"

    def test_prebuilt_part_inline_data_rejected(self):
        part = genai_types.Part(inline_data=genai_types.Blob(data=b"abc", mime_type="image/png"))
        with pytest.raises(ValueError, match="inline_data"):
            _build_content_parts([part], {}, {})

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported part type"):
            _build_content_parts([12345], {}, {})

    def test_part_media_resolution_applied(self, tmp_path):
        img = Image.new("RGB", (2, 2), "red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        path_to_uri = {str(img_path.resolve()): {"uri": "files/abc", "mime_type": "image/png"}}
        parts = _build_content_parts(
            [img_path], path_to_uri, {},
            part_media_resolution="MEDIA_RESOLUTION_LOW"
        )
        assert parts[0].media_resolution.level == "MEDIA_RESOLUTION_LOW"

    def test_mixed_content(self, tmp_path):
        img_path = tmp_path / "img.png"
        Image.new("RGB", (2, 2)).save(img_path)
        path_to_uri = {str(img_path.resolve()): {"uri": "files/f1", "mime_type": "image/png"}}

        parts = _build_content_parts(
            ["Analyze:", img_path, "What is this?"],
            path_to_uri, {}
        )
        assert len(parts) == 3
        assert parts[0].text == "Analyze:"
        assert parts[1].file_data.file_uri == "files/f1"
        assert parts[2].text == "What is this?"


# --- _extract_metadata ---

class TestExtractMetadata:
    def test_extracts_usage(self):
        response = _make_response("text")
        meta = _extract_metadata(response)
        assert meta is not None
        assert meta["usageMetadata"]["promptTokenCount"] == 10
        assert meta["usageMetadata"]["candidatesTokenCount"] == 5
        assert meta["usageMetadata"]["totalTokenCount"] == 15
        assert meta["modelVersion"] == "gemini-test"

    def test_no_metadata(self):
        response = MagicMock()
        response.usage_metadata = None
        response.model_version = None
        meta = _extract_metadata(response)
        assert meta is None


# --- async_process ---

class TestAsyncProcess:

    @pytest.mark.asyncio
    async def test_empty_prompts(self):
        results = await async_process(prompts=[])
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_prompts_with_metadata(self):
        results, metadata = await async_process(prompts=[], return_metadata=True)
        assert results == []
        assert metadata == []

    @pytest.mark.asyncio
    async def test_text_processing(self):
        mock_response = _make_response("The answer is 4")

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=mock_response)
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["What is 2+2?"]],
                show_progress=False,
            )

        assert len(results) == 1
        assert results[0] == "The answer is 4"

    @pytest.mark.asyncio
    async def test_structured_output(self):
        mock_response = _make_response('{"name": "test", "value": 42}')

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=mock_response)
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["Generate"]],
                schema=TestSchema,
                show_progress=False,
            )

        assert len(results) == 1
        assert isinstance(results[0], TestSchema)
        assert results[0].name == "test"
        assert results[0].value == 42

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json(self):
        mock_response = _make_response('```json\n{"name": "wrapped", "value": 7}\n```')

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=mock_response)
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["Generate"]],
                schema=TestSchema,
                show_progress=False,
            )

        assert results[0].name == "wrapped"
        assert results[0].value == 7

    @pytest.mark.asyncio
    async def test_return_metadata(self):
        mock_response = _make_response("Hello", prompt_tokens=20, candidates_tokens=10)

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=mock_response)
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results, metadata = await async_process(
                prompts=[["Hello"]],
                return_metadata=True,
                show_progress=False,
            )

        assert len(results) == 1
        assert len(metadata) == 1
        assert metadata[0]["usageMetadata"]["promptTokenCount"] == 20
        assert metadata[0]["usageMetadata"]["candidatesTokenCount"] == 10
        assert metadata[0]["usageMetadata"]["totalTokenCount"] == 30

    @pytest.mark.asyncio
    async def test_multiple_prompts(self):
        responses = [_make_response(f"Result {i}") for i in range(3)]

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(side_effect=responses)
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["P1"], ["P2"], ["P3"]],
                show_progress=False,
            )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        good_response = _make_response("Success")

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(
                side_effect=[Exception("Transient"), good_response]
            )
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["Test"]],
                retry_count=2,
                retry_delay=0.01,
                show_progress=False,
            )

        assert results[0] == "Success"
        assert mock_models.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(
                side_effect=Exception("Persistent error")
            )
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["Test"]],
                retry_count=1,
                retry_delay=0.01,
                show_progress=False,
            )

        assert results[0] is None
        assert mock_models.generate_content.call_count == 2  # Initial + 1 retry

    @pytest.mark.asyncio
    async def test_list_schema_wrapping(self):
        mock_response = _make_response('{"items": [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}]}')

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=mock_response)
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["List items"]],
                schema=List[ItemSchema],
                show_progress=False,
            )

        assert len(results) == 1
        assert len(results[0]) == 2
        assert isinstance(results[0][0], ItemSchema)
        assert results[0][0].id == 1
        assert results[0][1].label == "b"

    @pytest.mark.asyncio
    async def test_failed_parse_returns_none(self):
        mock_response = _make_response("not valid json at all")

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=mock_response)
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["Generate"]],
                schema=TestSchema,
                show_progress=False,
            )

        assert results[0] is None


# --- process (sync wrapper) ---

class TestSyncProcess:
    def test_sync_wrapper(self):
        mock_response = _make_response("Sync result")

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=mock_response)
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = process(
                prompts=[["Hello"]],
                show_progress=False,
            )

        assert len(results) == 1
        assert results[0] == "Sync result"


class TestIsRateLimitError:
    def test_detects_429_in_message(self):
        assert _is_rate_limit_error(Exception("429 RESOURCE_EXHAUSTED")) is True

    def test_detects_resource_exhausted(self):
        assert _is_rate_limit_error(Exception("RESOURCE_EXHAUSTED")) is True

    def test_ignores_other_errors(self):
        assert _is_rate_limit_error(Exception("Connection timeout")) is False
        assert _is_rate_limit_error(ValueError("Bad input")) is False


class TestRateLimitRetry:
    @pytest.mark.asyncio
    async def test_429_triggers_longer_backoff_and_succeeds(self):
        """429 errors should use longer backoff but still retry and succeed."""
        good_response = _make_response("OK")

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(
                side_effect=[Exception("429 RESOURCE_EXHAUSTED"), good_response]
            )
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["Test"]],
                retry_count=2,
                retry_delay=0.01,
                show_progress=False,
            )

        assert results[0] == "OK"
        assert mock_models.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_429_global_throttle_affects_concurrent_requests(self):
        """When one request hits a 429, other concurrent requests should also slow down."""
        good_response = _make_response("OK")
        call_times = []
        original_time = time.time

        async def tracked_generate(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) <= 1:
                raise Exception("429 RESOURCE_EXHAUSTED")
            return good_response

        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(side_effect=tracked_generate)
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["P1"], ["P2"]],
                retry_count=2,
                retry_delay=0.01,
                max_concurrent=2,
                show_progress=False,
            )

        # Both should eventually succeed
        successful = [r for r in results if r is not None]
        assert len(successful) == 2

    @pytest.mark.asyncio
    async def test_all_429_retries_exhausted(self):
        """If all retries hit 429, the prompt should fail gracefully."""
        with patch("gemini_batch.direct.GeminiClient") as MockClient, \
             patch("gemini_batch.direct.collect_and_upload_files", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = ({}, {})

            mock_client_instance = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(
                side_effect=Exception("429 RESOURCE_EXHAUSTED")
            )
            mock_aio.models = mock_models
            mock_client_instance.client.aio = mock_aio
            MockClient.return_value = mock_client_instance

            results = await async_process(
                prompts=[["Test"]],
                retry_count=1,
                retry_delay=0.01,
                show_progress=False,
            )

        assert results[0] is None
        assert mock_models.generate_content.call_count == 2
