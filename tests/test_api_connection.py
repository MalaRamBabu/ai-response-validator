import pytest


class TestAPIConnection:
    """
    Test Suite 1: API Connection & Basic Response Validation
    Verifies Gemini API is reachable and returns valid responses.
    """

    def test_api_returns_200_status(self, gemini):
        """TC-001: Gemini API returns HTTP 200 for valid prompt"""
        result = gemini.send_prompt("Say hello")
        assert result["status_code"] == 200, \
            f"Expected 200, got {result['status_code']}. Error: {result['error']}"

    def test_api_response_is_not_none(self, gemini, validator):
        """TC-002: Gemini API response text is not None"""
        result = gemini.send_prompt("What is 1 + 1?")
        assert result["text"] is not None, "Response text should not be None"

    def test_api_response_is_not_empty(self, gemini, validator):
        """TC-003: Gemini API response text is not empty"""
        result = gemini.send_prompt("What is the color of the sky?")
        assert validator.is_not_empty(result["text"]), \
            "Response should not be empty"

    def test_api_response_has_valid_structure(self, gemini, validator):
        """TC-004: Gemini response dictionary has all required keys"""
        result = gemini.send_prompt("Say one word")
        assert validator.is_valid_json_structure(result), \
            f"Invalid response structure: {result}"

    def test_api_response_time_under_10_seconds(self, gemini, validator):
        """TC-005: Gemini API responds within 10 seconds"""
        result = gemini.send_prompt("What is Python?")
        assert validator.response_time_is_acceptable(
            result["response_time_seconds"], max_seconds=10.0
        ), f"Response too slow: {result['response_time_seconds']:.2f}s"

    def test_api_response_has_minimum_length(self, gemini, validator):
        """TC-006: Gemini response has meaningful content (min 10 chars)"""
        result = gemini.send_prompt("Describe the Sun in one sentence.")
        assert validator.is_above_min_length(result["text"], min_chars=10), \
            f"Response too short: '{result['text']}'"

    def test_api_response_within_max_length(self, gemini, validator):
        """TC-007: Gemini response does not exceed 5000 characters"""
        result = gemini.send_prompt("What is machine learning?")
        assert validator.is_within_length(result["text"], max_chars=5000), \
            f"Response too long: {len(result['text'])} chars"

    @pytest.mark.parametrize("prompt", [
        "What is 2 + 2?",
        "Name one planet in our solar system.",
        "What language is Python written in?",
    ])
    def test_multiple_simple_prompts_return_200(self, gemini, prompt):
        """TC-008: Multiple simple prompts all return 200 status"""
        result = gemini.send_prompt(prompt)
        assert result["status_code"] == 200, \
            f"Prompt '{prompt}' failed with status {result['status_code']}"
