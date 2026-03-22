import os
import requests


class GeminiClient:
    """
    Client to interact with Google Gemini API.
    API Key is loaded from environment variable — never hardcoded.
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set!\n"
                "Run this command first:\n"
                "  Windows CMD:  set GEMINI_API_KEY=your_key_here\n"
                "  Mac/Linux:    export GEMINI_API_KEY=your_key_here"
            )

    def ask(self, prompt, temperature=0.7, max_tokens=512):
        """
        Send a prompt to Gemini and get response text.
        """
        url      = f"{self.API_URL}?key={self.api_key}"
        payload  = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature":     temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def get_raw_response(self, prompt):
        """
        Send a prompt and return the full raw API response.
        Used for structure and status code validation.
        """
        url     = f"{self.API_URL}?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        return requests.post(url, json=payload, timeout=30)
