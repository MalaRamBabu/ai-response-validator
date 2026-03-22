import pytest
from utils.gemini_client import GeminiClient
from utils.ai_validator  import AIValidator


@pytest.fixture(scope="session")
def gemini():
    """Shared Gemini client for entire test session"""
    return GeminiClient()


@pytest.fixture(scope="session")
def validator():
    """Shared AI validator helper for entire test session"""
    return AIValidator()


# ── Prompt Test Data ──────────────────────────────────────────────────────────

@pytest.fixture
def factual_prompts():
    return [
        "What is the capital of France?",
        "What is 2 + 2?",
        "What is the boiling point of water in Celsius?",
    ]

@pytest.fixture
def coding_prompts():
    return [
        "Write a Python function to add two numbers",
        "Write a Python function to reverse a string",
        "Write a Python function to check if a number is even",
    ]

@pytest.fixture
def greeting_prompts():
    return [
        "Say hello",
        "Good morning! How are you?",
        "Introduce yourself in one sentence",
    ]

@pytest.fixture
def edge_case_prompts():
    return [
        ".",
        "????",
        "123456789",
        "   ",
    ]
