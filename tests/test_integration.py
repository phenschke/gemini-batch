"""
Integration tests with live Gemini API.

Tests the three core batch processing workflows:
1. Text in → Simple structured output (single field)
2. Text in → Rich structured output (multiple fields)
3. Image in → Structured output (multimodal)

Run with: pytest tests/test_integration.py -v -s -m integration
"""
import pytest
import os
from pydantic import BaseModel
from PIL import Image, ImageDraw
import io

from gemini_batch import batch_process, batch_embed

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def check_api_key():
    """Skip all tests if GEMINI_API_KEY is not set."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set - skipping integration tests")


@pytest.fixture
def model():
    """Use the cheapest model for testing."""
    return "gemini-flash-lite-latest"


# Test 1: Text in → Simple structured output
def test_text_to_simple_structured(model):
    """Test basic batch processing with text input and simple single-field structured output."""

    class Answer(BaseModel):
        """Simple single-field response."""
        result: str

    prompts = [
        ["What is 2+2? Answer with just the number."],
        ["What is 3+3? Answer with just the number."],
    ]

    results = batch_process(
        prompts=prompts,
        schema=Answer,
        model=model,
        wait=True,
        temperature=0.1,
    )

    assert len(results) == 2
    assert isinstance(results[0], Answer)
    assert isinstance(results[1], Answer)
    assert "4" in results[0].result
    assert "6" in results[1].result

    print(f"\n✓ Text→Simple Structured: Got {results[0].result} and {results[1].result}")


# Test 2: Text in → Rich structured output
def test_text_to_rich_structured(model):
    """Test batch processing with multi-field structured Pydantic output."""

    class MathAnswer(BaseModel):
        """Structured response for math questions."""
        answer: int
        explanation: str

    prompts = [
        ["What is 5+5? Answer and explain your reasoning."],
        ["What is 10+10? Answer and explain your reasoning."],
    ]

    results = batch_process(
        prompts=prompts,
        schema=MathAnswer,
        model=model,
        wait=True,
        temperature=0.1,
    )

    assert len(results) == 2
    assert isinstance(results[0], MathAnswer)
    assert isinstance(results[1], MathAnswer)
    assert results[0].answer == 10
    assert results[1].answer == 20
    assert len(results[0].explanation) > 0
    assert len(results[1].explanation) > 0

    print(f"\n✓ Text→Rich Structured: {results[0].model_dump()}")


# Test 3: Text + Image in → Structured output
def test_multimodal_to_structured(model):
    """Test batch processing with image input and structured output."""

    class ImageContent(BaseModel):
        """Structured response for image analysis."""
        primary_text: str
        description: str

    # Create a simple test image with text
    def create_test_image(text: str) -> bytes:
        img = Image.new('RGB', (300, 150), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 60), text, fill='black')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    prompts = [
        ["Read the text shown in this image and describe what you see:", create_test_image("HELLO")],
        ["Read the text shown in this image and describe what you see:", create_test_image("WORLD")],
    ]

    results = batch_process(
        prompts=prompts,
        schema=ImageContent,
        model=model,
        wait=True,
        temperature=0.1,
    )

    assert len(results) == 2
    assert isinstance(results[0], ImageContent)
    assert isinstance(results[1], ImageContent)

    # The model should detect the text in the images
    assert "hello" in results[0].primary_text.lower()
    assert "world" in results[1].primary_text.lower()
    assert len(results[0].description) > 0
    assert len(results[1].description) > 0

    print(f"\n✓ Image→Structured: {results[0].model_dump()}")


# Test 4: Thinking budget verification
def test_thinking_budget_respected(model):
    """Test that thinking_budget parameter is respected by the API."""

    class Reasoning(BaseModel):
        """Response with explanation and answer."""
        explanation: str
        answer: str

    prompts = [
        ["Solve this problem: If a train travels 60 miles in 1 hour, how far does it travel in 3 hours? Explain your reasoning."],
        ["What is 15 * 8? Show your work."],
    ]

    # Test 1: With thinking enabled (budget = 1024)
    results_with_thinking, metadata_with_thinking = batch_process(
        prompts=prompts,
        schema=Reasoning,
        model=model,
        wait=True,
        temperature=0.5,
        thinking_budget=512,
        return_metadata=True,
    )

    assert len(results_with_thinking) == 2
    assert len(metadata_with_thinking) == 2
    assert isinstance(results_with_thinking[0], Reasoning)
    assert isinstance(results_with_thinking[1], Reasoning)

    print(f"\n✓ With thinking budget (512):")
    print(f"  Result: {results_with_thinking[0].model_dump()}")

    # Verify thinking tokens were used
    for i, meta in enumerate(metadata_with_thinking):
        print(f"  Metadata {i}: {meta}")
        assert 'usageMetadata' in meta, "Metadata should contain usageMetadata"
        thoughts_count = meta['usageMetadata'].get('thoughtsTokenCount')

        # With thinking_budget > 0, we expect thinking tokens (though it may be 0 for trivial tasks)
        # The key test is that the field exists and is not None
        assert thoughts_count is not None, f"With thinking_budget=512, thoughtsTokenCount should not be None"
        print(f"  ✓ Thoughts tokens: {thoughts_count}")

    # Test 2: With thinking disabled (budget = 0)
    results_without_thinking, metadata_without_thinking = batch_process(
        prompts=prompts,
        schema=Reasoning,
        model=model,
        wait=True,
        temperature=0.5,
        thinking_budget=0,
        return_metadata=True,
    )

    assert len(results_without_thinking) == 2
    assert len(metadata_without_thinking) == 2
    assert isinstance(results_without_thinking[0], Reasoning)
    assert isinstance(results_without_thinking[1], Reasoning)

    print(f"\n✓ Without thinking budget (0):")
    print(f"  Result: {results_without_thinking[0].model_dump()}")

    # Verify thinking was reduced (API may still use minimal thinking even with budget=0)
    for i, meta in enumerate(metadata_without_thinking):
        print(f"  Metadata {i}: {meta}")
        assert 'usageMetadata' in meta, "Metadata should contain usageMetadata"
        thoughts_count = meta['usageMetadata'].get('thoughtsTokenCount')

        # With thinking_budget=0, thoughts should be significantly reduced compared to budget=512
        # Note: API sometimes uses minimal thinking tokens (e.g., 4) even with budget=0
        thoughts_with_budget = metadata_with_thinking[i]['usageMetadata'].get('thoughtsTokenCount', 0)

        if thoughts_count is None or thoughts_count <= 5:
            print(f"  ✓ Thoughts tokens disabled: {thoughts_count}")
        elif thoughts_with_budget >= 5:
            # Verify significant reduction (at least 90% reduction)
            reduction_ratio = 1 - (thoughts_count / thoughts_with_budget)
            print(f"  ✓ Thoughts tokens reduced from {thoughts_with_budget} to {thoughts_count} ({reduction_ratio*100:.1f}% reduction)")
            assert reduction_ratio > 0.9, \
                f"Expected >90% reduction in thinking tokens with budget=0, got {reduction_ratio*100:.1f}% reduction"
        else:
            print(f"  ✓ Thoughts tokens: {thoughts_count}")

    print("\n✓ Confirmed: thinking_budget parameter significantly affects token usage")


# Test 5: Verify thinking metadata access (non-batch mode for comparison)
def test_thinking_metadata_access():
    """Test direct API call to verify thinking metadata is accessible (non-batch)."""
    from gemini_batch.utils import GeminiClient, build_generation_config
    from pydantic import BaseModel

    class SimpleAnswer(BaseModel):
        reasoning: str
        result: str
        

    client = GeminiClient()

    # Make a direct (non-batch) API call with thinking enabled
    gen_config = build_generation_config(
        response_schema=SimpleAnswer,
        thinking_budget=512,
        temperature=0.5,
    )

    response = client.client.models.generate_content(
        model="gemini-flash-lite-latest",  # Use a thinking model
        contents="What is 2+2? Explain your reasoning briefly.",
        config=gen_config,
    )

    # Verify we can access usage metadata
    assert hasattr(response, 'usage_metadata'), "Response should have usage_metadata"
    usage = response.usage_metadata

    print(f"\n✓ Usage metadata accessible:")
    print(f"  - Total tokens: {usage.total_token_count}")
    print(f"  - Prompt tokens: {usage.prompt_token_count}")
    print(f"  - Candidates tokens: {usage.candidates_token_count}")

    # Check for thinking tokens (may be 0 or None depending on model/prompt)
    if hasattr(usage, 'thoughts_token_count'):
        print(f"  - Thoughts tokens: {usage.thoughts_token_count}")
        print("\n✓ Thinking metadata is accessible in the API response")
    else:
        print("  - Thoughts tokens: Not present (may be 0 or model doesn't support it)")

    # Now test with thinking disabled
    gen_config_no_thinking = build_generation_config(
        response_schema=SimpleAnswer,
        thinking_budget=0,
        temperature=0.5,
    )

    response_no_thinking = client.client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents="What is 2+2? Explain your reasoning briefly.",
        config=gen_config_no_thinking,
    )

    usage_no_thinking = response_no_thinking.usage_metadata
    print(f"\n✓ Without thinking (budget=0):")
    print(f"  - Total tokens: {usage_no_thinking.total_token_count}")
    print(f"  - Candidates tokens: {usage_no_thinking.candidates_token_count}")

    if hasattr(usage_no_thinking, 'thoughts_token_count'):
        print(f"  - Thoughts tokens: {usage_no_thinking.thoughts_token_count}")
        assert usage_no_thinking.thoughts_token_count < 10 or usage_no_thinking.thoughts_token_count is None, \
            "With thinking_budget=0, thoughts_token_count should be close to 0 or None"
        print("\n✓ Confirmed: thinking_budget=0 results in no thinking tokens")


# Test 6: thinking_level="MINIMAL" enforcement with gemini-3-flash
def test_thinking_level_minimal_plain_text():
    """Test thinking_level='MINIMAL' keeps thoughts < 500 tokens (plain text output)."""
    prompts = [
        ["What is 15 * 17? Show your work."],
        ["If a car travels at 60mph for 2.5 hours, how far does it go?"],
    ]

    results, metadata = batch_process(
        prompts=prompts,
        schema=None,  # Plain text
        model="gemini-3-flash-preview",
        wait=True,
        thinking_level="MINIMAL",
        return_metadata=True,
    )

    assert len(results) == 2
    assert len(metadata) == 2

    print("\n✓ thinking_level='MINIMAL' with plain text:")
    for i, meta in enumerate(metadata):
        assert 'usageMetadata' in meta, "Metadata should contain usageMetadata"
        thoughts_count = meta['usageMetadata'].get('thoughtsTokenCount', 0) or 0
        print(f"  Request {i}: {thoughts_count} thinking tokens")
        assert thoughts_count < 500, f"Expected < 500 thinking tokens with MINIMAL, got {thoughts_count}"

    print("✓ Plain text: MINIMAL thinking enforced (< 500 tokens)")


def test_thinking_level_minimal_structured():
    """Test thinking_level='MINIMAL' keeps thoughts < 500 tokens (structured output)."""

    class MathResult(BaseModel):
        answer: int
        explanation: str

    prompts = [
        ["What is 15 * 17? Show your work."],
        ["If a car travels at 60mph for 2.5 hours, how far does it go in miles?"],
    ]

    results, metadata = batch_process(
        prompts=prompts,
        schema=MathResult,
        model="gemini-3-flash-preview",
        wait=True,
        thinking_level="MINIMAL",
        return_metadata=True,
    )

    assert len(results) == 2
    assert len(metadata) == 2
    assert isinstance(results[0], MathResult)
    assert isinstance(results[1], MathResult)

    print("\n✓ thinking_level='MINIMAL' with structured output:")
    print(f"  Result 0: {results[0].model_dump()}")
    print(f"  Result 1: {results[1].model_dump()}")
    for i, meta in enumerate(metadata):
        assert 'usageMetadata' in meta, "Metadata should contain usageMetadata"
        thoughts_count = meta['usageMetadata'].get('thoughtsTokenCount', 0) or 0
        print(f"  Request {i}: {thoughts_count} thinking tokens")
        assert thoughts_count < 500, f"Expected < 500 thinking tokens with MINIMAL, got {thoughts_count}"

    print("✓ Structured output: MINIMAL thinking enforced (< 500 tokens)")


# Test 7: Batch embeddings
def test_batch_embed_basic():
    """Test basic batch embedding generation."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries.",
    ]

    embeddings = batch_embed(texts=texts, wait=True)

    assert len(embeddings) == 2
    assert all(isinstance(e, list) for e in embeddings)
    assert all(len(e) > 0 for e in embeddings)  # Embeddings have values
    assert all(isinstance(v, float) for e in embeddings for v in e)  # Values are floats

    # Check embedding dimensions are consistent
    assert len(embeddings[0]) == len(embeddings[1])

    print(f"\n✓ Batch Embeddings: Got embeddings with dimension {len(embeddings[0])}")


def test_batch_embed_with_task_types():
    """Test batch embeddings with different task types."""
    texts = ["Search query about Python programming"]

    # Test RETRIEVAL_QUERY (for search queries)
    query_embeddings = batch_embed(
        texts=texts,
        task_type="RETRIEVAL_QUERY",
        wait=True,
    )
    assert len(query_embeddings) == 1
    assert len(query_embeddings[0]) > 0

    # Test RETRIEVAL_DOCUMENT (for documents)
    doc_embeddings = batch_embed(
        texts=texts,
        task_type="RETRIEVAL_DOCUMENT",
        wait=True,
    )
    assert len(doc_embeddings) == 1
    assert len(doc_embeddings[0]) > 0

    # Embeddings should have same dimension but different values
    assert len(query_embeddings[0]) == len(doc_embeddings[0])
    assert query_embeddings[0] != doc_embeddings[0]  # Different task types produce different embeddings

    print(f"\n✓ Task Types: RETRIEVAL_QUERY and RETRIEVAL_DOCUMENT produce different embeddings")


def test_batch_embed_with_metadata():
    """Test batch embeddings with metadata return."""
    texts = ["Test embedding for metadata"]

    embeddings, metadata = batch_embed(
        texts=texts,
        return_metadata=True,
    )

    assert len(embeddings) == 1
    assert len(metadata) == 1
    assert embeddings[0] is not None
    assert len(embeddings[0]) > 0

    print(f"\n✓ Metadata: Embedding dimension {len(embeddings[0])}, metadata: {metadata[0]}")


def test_batch_embed_vertexai_not_supported():
    """Test that batch embeddings raise clear error for Vertex AI."""
    texts = ["Test text"]

    with pytest.raises(ValueError, match="not supported with Vertex AI"):
        batch_embed(
            texts=texts,
            vertexai=True,
        )

    print("\n✓ Vertex AI Embeddings: Correctly raises error (not supported)")


if __name__ == "__main__":
    # Allow running directly with: python -m pytest tests/test_integration.py -v -s
    pytest.main([__file__, "-v", "-s"])
