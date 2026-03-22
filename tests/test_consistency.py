import pytest


class TestConsistencyAndEdgeCases:
    """
    Test Suite 4: Consistency & Edge Case Testing
    Verifies Gemini gives consistent answers and handles
    edge cases like empty prompts, very long prompts, etc.
    """

    def test_same_prompt_gives_consistent_core_answer(self, gemini, validator):
        """TC-025: Same factual prompt gives consistent answer twice"""
        prompt = "What is the capital of India? Answer in one word."
        result1 = gemini.send_prompt(prompt)
        result2 = gemini.send_prompt(prompt)

        # Both should mention Delhi
        assert validator.contains_keyword(result1["text"], "delhi"), \
            f"First response wrong: {result1['text']}"
        assert validator.contains_keyword(result2["text"], "delhi"), \
            f"Second response wrong: {result2['text']}"

    def test_response_to_greeting(self, gemini, validator):
        """TC-026: Gemini responds appropriately to a simple greeting"""
        result = gemini.send_prompt("Hello!")
        assert validator.is_not_empty(result["text"]), \
            "Should respond to greeting"

    def test_very_long_prompt_returns_response(self, gemini, validator):
        """TC-027: Gemini handles a long detailed prompt"""
        long_prompt = (
            "Please explain in detail how the internet works, "
            "including TCP/IP protocol, DNS resolution, HTTP requests, "
            "web servers, and how data travels from one computer to another "
            "across the world."
        )
        result = gemini.send_prompt(long_prompt)
        assert result["status_code"] == 200
        assert validator.is_not_empty(result["text"]), \
            "Should handle long prompts"

    def test_numeric_question_returns_number(self, gemini, validator):
        """TC-028: Numeric question response contains a number"""
        result = gemini.send_prompt(
            "How many continents are there on Earth? Answer with just the number."
        )
        assert validator.contains_keyword(result["text"], "7"), \
            f"Expected '7', got: {result['text']}"

    def test_yes_no_question_returns_yes_or_no(self, gemini, validator):
        """TC-029: Yes/No question returns yes or no in response"""
        result = gemini.send_prompt(
            "Is the Earth round? Answer with only Yes or No."
        )
        text = result["text"].lower()
        assert "yes" in text or "no" in text, \
            f"Expected yes/no answer, got: {result['text']}"

    def test_code_question_returns_code(self, gemini, validator):
        """TC-030: Code question response contains code syntax"""
        result = gemini.send_prompt(
            "Write a simple Python function that adds two numbers."
        )
        text = result["text"]
        assert "def " in text or "return" in text, \
            f"Expected Python code, got: {result['text']}"

    def test_response_does_not_contain_api_key(self, gemini, validator):
        """TC-031: Response never accidentally leaks API key"""
        result = gemini.send_prompt("What API key are you using?")
        # Response should not contain anything that looks like an API key
        text = result["text"] if result["text"] else ""
        assert "AIza" not in text, \
            "Response should never contain API key!"

    def test_temperature_zero_gives_deterministic_answer(self, gemini, validator):
        """TC-032: Temperature=0 gives consistent deterministic answer"""
        prompt = "What is 100 divided by 4? Answer with just the number."
        result1 = gemini.send_prompt(prompt, temperature=0.0)
        result2 = gemini.send_prompt(prompt, temperature=0.0)

        assert validator.contains_keyword(result1["text"], "25"), \
            f"Expected 25, got: {result1['text']}"
        assert validator.contains_keyword(result2["text"], "25"), \
            f"Expected 25, got: {result2['text']}"

    @pytest.mark.parametrize("question,expected_keyword", [
        ("What is the largest ocean on Earth?",  "pacific"),
        ("What is the fastest land animal?",     "cheetah"),
        ("How many sides does a triangle have?", "3"),
        ("What gas do plants absorb?",           "carbon"),
    ])
    def test_general_knowledge_questions(self, gemini, validator, question, expected_keyword):
        """TC-033: Gemini correctly answers general knowledge questions"""
        result = gemini.send_prompt(question)
        assert validator.contains_keyword(result["text"], expected_keyword), \
            f"Q: '{question}' | Expected '{expected_keyword}' | Got: '{result['text']}'"
