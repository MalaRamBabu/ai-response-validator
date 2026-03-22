import pytest


class TestFactualResponses:
    """
    Test Suite 2: Factual Response Validation
    Validates that Gemini gives correct answers to
    well-known factual questions — basic hallucination check.
    """

    def test_capital_of_france_is_paris(self, gemini, validator):
        """TC-008: AI correctly identifies Paris as capital of France"""
        prompt   = "What is the capital of France? Answer in one word only."
        response = gemini.ask(prompt)
        validator.assert_keyword_in_response(response, "Paris", prompt)

    def test_python_is_programming_language(self, gemini, validator):
        """TC-009: AI knows Python is a programming language"""
        prompt   = "What is Python? Answer in one sentence."
        response = gemini.ask(prompt)
        validator.assert_keyword_in_response(response, "programming", prompt)

    def test_water_boiling_point_response(self, gemini, validator):
        """TC-010: AI response about boiling point contains 100"""
        prompt   = "What is the boiling point of water in Celsius? Give number only."
        response = gemini.ask(prompt)
        validator.assert_keyword_in_response(response, "100", prompt)

    def test_playwright_is_testing_tool(self, gemini, validator):
        """TC-011: AI knows Playwright is a testing/automation tool"""
        prompt   = "What is Microsoft Playwright? One sentence answer."
        response = gemini.ask(prompt)
        # Should mention testing, automation, or browser
        is_relevant = any(
            word in response.lower()
            for word in ["test", "automation", "browser", "web"]
        )
        assert is_relevant, \
            f"Response doesn't mention testing/automation/browser:\n{response}"

    def test_basic_math_addition(self, gemini, validator):
        """TC-012: AI correctly answers 2 + 2 = 4"""
        prompt   = "What is 2 + 2? Give number only."
        response = gemini.ask(prompt)
        validator.assert_keyword_in_response(response, "4", prompt)

    def test_selenium_is_automation_tool(self, gemini, validator):
        """TC-013: AI knows Selenium is a test automation tool"""
        prompt   = "What is Selenium WebDriver? One sentence."
        response = gemini.ask(prompt)
        is_relevant = any(
            word in response.lower()
            for word in ["test", "automation", "browser", "web"]
        )
        assert is_relevant, \
            f"Response doesn't mention automation/testing:\n{response}"

    def test_pytest_is_python_framework(self, gemini, validator):
        """TC-014: AI knows PyTest is a Python testing framework"""
        prompt   = "What is PyTest? One sentence."
        response = gemini.ask(prompt)
        is_relevant = any(
            word in response.lower()
            for word in ["python", "test", "framework"]
        )
        assert is_relevant, \
            f"Response doesn't mention Python/testing/framework:\n{response}"

    @pytest.mark.parametrize("prompt,expected", [
        ("What is the capital of Japan? One word.", "Tokyo"),
        ("What language does Python code use? One word.", "Python"),
        ("What does API stand for? Short answer.", "Application"),
    ])
    def test_multiple_factual_questions(self, gemini, validator, prompt, expected):
        """TC-015: Multiple factual questions return relevant answers"""
        response = gemini.ask(prompt)
        validator.assert_keyword_in_response(response, expected, prompt)
