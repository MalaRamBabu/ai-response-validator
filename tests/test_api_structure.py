import pytest
import time


class TestAPIStructure:
    """
    Test Suite 1: API Structure & Status Validation
    Validates that the Gemini API returns correct HTTP status,
    response structure, and acceptable response times.
    """

    def test_api_returns_200_status(self, gemini, validator):
        """TC-001: Valid prompt returns HTTP 200 status code"""
        response = gemini.get_raw_response("Say hello")
        validator.assert_api_status_200(response)

    def test_api_response_has_correct_structure(self, gemini, validator):
        """TC-002: Response JSON has required keys — candidates, content, parts"""
        response = gemini.get_raw_response("What is Python?")
        validator.assert_api_status_200(response)
        validator.assert_response_structure(response)

    def test_api_response_time_within_limit(self, gemini, validator):
        """TC-003: API responds within 10 seconds"""
        start    = time.time()
        gemini.ask("Say hello in one word")
        elapsed  = time.time() - start
        validator.assert_response_time(elapsed, max_seconds=10.0)

    def test_api_response_is_string(self, gemini, validator):
        """TC-004: Response text is a string type"""
        response = gemini.ask("What is AI?")
        validator.assert_is_string(response)

    def test_api_response_not_empty(self, gemini, validator):
        """TC-005: Response text is not empty"""
        prompt   = "Describe Python in one sentence"
        response = gemini.ask(prompt)
        validator.assert_not_empty(response, prompt)

    def test_api_response_has_minimum_length(self, gemini, validator):
        """TC-006: Response has at least 10 characters"""
        prompt   = "What is testing?"
        response = gemini.ask(prompt)
        validator.assert_min_length(response, min_chars=10, prompt=prompt)

    def test_api_multiple_calls_all_return_responses(self, gemini, validator):
        """TC-007: Three consecutive API calls all return valid responses"""
        prompts = [
            "What is QA testing?",
            "What is automation?",
            "What is Playwright?",
        ]
        for prompt in prompts:
            response = gemini.ask(prompt)
            validator.assert_not_empty(response, prompt)
