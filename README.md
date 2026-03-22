# AI Response Validator

A Python-based test automation framework for validating **Google Gemini AI API** responses using **PyTest**. Tests AI output quality, factual accuracy, safety, response structure, and edge case handling.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Programming language |
| Google Gemini API | AI model being tested |
| Requests | HTTP client for API calls |
| PyTest | Test framework |
| pytest-html | HTML test reports |
| GitHub Actions | CI/CD pipeline |

---

## Project Structure

```
ai_response_validator/
├── utils/
│   ├── gemini_client.py   # Gemini API client
│   └── ai_validator.py    # Reusable assertion helpers
├── tests/
│   ├── test_api_structure.py     # 7 test cases
│   ├── test_factual_responses.py # 8 test cases
│   ├── test_safety_content.py    # 8 test cases
│   └── test_edge_cases.py        # 8 test cases
├── reports/               # HTML reports (auto-generated)
├── .github/workflows/
│   └── ai_tests.yml       # GitHub Actions CI/CD
├── conftest.py            # Fixtures and test data
├── pytest.ini             # PyTest configuration
├── requirements.txt       # Dependencies
├── .env.example           # Environment variable template
└── README.md
```

---

## Test Coverage

| Test Suite | Test Cases | What is Validated |
|---|---|---|
| API Structure | 7 | Status 200, JSON structure, response time |
| Factual Responses | 8 | Correct answers, hallucination check |
| Safety & Content | 8 | No harmful content, English language |
| Edge Cases | 8 | Short/long prompts, special characters |
| **Total** | **31** | |

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/MalaRamBabu/ai-response-validator.git
cd ai-response-validator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key
```bash
# Windows CMD
set GEMINI_API_KEY=your_api_key_here

# Mac / Linux
export GEMINI_API_KEY=your_api_key_here
```
Get a free key at: https://aistudio.google.com

### 4. Run all tests
```bash
pytest
```

### 5. Run specific suite
```bash
pytest tests/test_factual_responses.py
pytest tests/test_safety_content.py
```

---

## CI/CD — GitHub Actions
Tests run automatically on every push.
API key is stored as a **GitHub Secret** — never exposed in code.

---

## Author

**Mala Ram Babu**
Senior QA Automation Engineer | 4+ Years Experience
[LinkedIn](https://www.linkedin.com/in/mala-ram-babu) | [GitHub](https://github.com/MalaRamBabu)
