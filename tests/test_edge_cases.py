import pytest


class TestEdgeCases:
    """
    Test Suite 4: Edge Case & Boundary Testing
    Tests AI behavior with unusual, minimal, or
    boundary inputs — important for robustness testing.
    """

    def test_very_short_prompt_returns_response(self, gemini, validator):
        """TC-024: Single word prompt still gets a response"""
        prompt   = "Hi"
        response = gemini.ask(prompt)
        validator.assert_not_empty(response, prompt)

    def test_question_mark_only_prompt(self, gemini, validator):
        """TC-025: Prompt with only '?' still returns a response"""
        prompt   = "?"
        response = gemini.ask(prompt)
        validator.assert_not_empty(response, prompt)

    def test_numeric_prompt_returns_response(self, gemini, validator):
        """TC-026: Numeric-only prompt returns a response"""
        prompt   = "42"
        response = gemini.ask(prompt)
        validator.assert_not_empty(response, prompt)

    def test_very_long_prompt_returns_response(self, gemini, validator):
        """TC-027: Very long prompt still returns a valid response"""
        prompt   = "What is software testing? " * 20  # Repeat 20 times
        response = gemini.ask(prompt)
        validator.assert_not_empty(response, "very long repeated prompt")

    def test_prompt_with_special_characters(self, gemini, validator):
        """TC-028: Prompt with special characters returns a response"""
        prompt   = "What is Python? !@#$%"
        response = gemini.ask(prompt)
        validator.assert_not_empty(response, prompt)

    def test_low_temperature_gives_consistent_response(self, gemini, validator):
        """TC-029: Low temperature (0.1) gives consistent factual response"""
        prompt   = "What is the capital of France? One word."
        response = gemini.ask(prompt, temperature=0.1)
        validator.assert_keyword_in_response(response, "Paris", prompt)

    def test_max_tokens_limits_response_length(self, gemini, validator):
        """TC-030: Setting max_tokens=50 limits response length"""
        prompt   = "Explain the entire history of software testing in detail"
        response = gemini.ask(prompt, max_tokens=50)
        # Should be shorter than an unlimited response
        assert len(response) < 2000, \
            f"Response too long despite max_tokens=50: {len(response)} chars"

    @pytest.mark.parametrize("prompt", [
        "What is Playwright?",
        "What is Selenium?",
        "What is PyTest?",
        "What is API testing?",
        "What is CI/CD?",
    ])
    def test_qa_tool_prompts_all_return_responses(self, gemini, validator, prompt):
        """TC-031: All QA tool-related prompts return valid responses"""
        response = gemini.ask(prompt)
        validator.assert_not_empty(response, prompt)
        validator.assert_is_string(response)
