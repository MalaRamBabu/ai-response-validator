import pytest


class TestFactualAccuracy:
    """
    Test Suite 2: Factual Accuracy Validation
    Verifies Gemini gives correct answers to known factual questions.
    This is the core of AI testing — checking if the AI knows correct facts.
    """

    def test_math_multiplication(self, gemini, validator):
        """TC-009: Gemini correctly answers 10 x 5 = 50"""
        result = gemini.send_prompt(
            "What is 10 multiplied by 5? Answer with just the number."
        )
        assert validator.is_not_empty(result["text"]), "Empty response"
        assert validator.contains_keyword(result["text"], "50"), \
            f"Expected '50' in response, got: {result['text']}"

    def test_capital_of_japan(self, gemini, validator):
        """TC-010: Gemini correctly identifies Tokyo as capital of Japan"""
        result = gemini.send_prompt(
            "What is the capital city of Japan? Answer in one word."
        )
        assert validator.contains_keyword(result["text"], "tokyo"), \
            f"Expected 'Tokyo' in response, got: {result['text']}"

    def test_closest_planet_to_sun(self, gemini, validator):
        """TC-011: Gemini correctly identifies Mercury as closest to Sun"""
        result = gemini.send_prompt(
            "Which planet is closest to the Sun? Answer in one word."
        )
        assert validator.contains_keyword(result["text"], "mercury"), \
            f"Expected 'Mercury' in response, got: {result['text']}"

    def test_language_spoken_in_brazil(self, gemini, validator):
        """TC-012: Gemini correctly identifies Portuguese as Brazil's language"""
        result = gemini.send_prompt(
            "What is the official language of Brazil? Answer in one word."
        )
        assert validator.contains_keyword(result["text"], "portuguese"), \
            f"Expected 'Portuguese' in response, got: {result['text']}"

    def test_capital_of_india(self, gemini, validator):
        """TC-013: Gemini correctly identifies New Delhi as capital of India"""
        result = gemini.send_prompt(
            "What is the capital of India? Answer in one or two words."
        )
        assert (
            validator.contains_keyword(result["text"], "delhi") or
            validator.contains_keyword(result["text"], "new delhi")
        ), f"Expected 'Delhi' in response, got: {result['text']}"

    def test_python_is_programming_language(self, gemini, validator):
        """TC-014: Gemini correctly identifies Python as a programming language"""
        result = gemini.send_prompt(
            "Is Python a programming language? Answer with just Yes or No."
        )
        assert validator.contains_keyword(result["text"], "yes"), \
            f"Expected 'Yes', got: {result['text']}"

    def test_water_boiling_point(self, gemini, validator):
        """TC-015: Gemini correctly states water boils at 100 degrees Celsius"""
        result = gemini.send_prompt(
            "At what temperature does water boil in Celsius? Answer with just the number."
        )
        assert validator.contains_keyword(result["text"], "100"), \
            f"Expected '100' in response, got: {result['text']}"

    def test_sun_is_a_star(self, gemini, validator):
        """TC-016: Gemini correctly classifies the Sun as a star"""
        result = gemini.send_prompt(
            "Is the Sun a star or a planet? Answer in one word."
        )
        assert validator.contains_keyword(result["text"], "star"), \
            f"Expected 'star', got: {result['text']}"

    @pytest.mark.parametrize("prompt,expected", [
        ("What is 5 + 5? Answer with the number only.", "10"),
        ("How many days are in a week? Answer with the number only.", "7"),
        ("How many months are in a year? Answer with the number only.", "12"),
    ])
    def test_basic_math_and_facts(self, gemini, validator, prompt, expected):
        """TC-017: Gemini correctly answers basic math and general facts"""
        result = gemini.send_prompt(prompt)
        assert validator.contains_keyword(result["text"], expected), \
            f"Prompt: '{prompt}' | Expected: '{expected}' | Got: '{result['text']}'"
