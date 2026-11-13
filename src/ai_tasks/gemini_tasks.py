
import os
import json
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

CATEGORIES = [
    "Network",
    "Software",
    "Hardware",
    "Access Management",
    "Security"
]

SENTIMENTS = ["Positive", "Neutral", "Negative"]

URGENCY_LEVELS = ["Low", "Medium", "High", "Critical"]

REQUIRED_FIELDS = [
    "category",
    "sentiment",
    "urgency",
    "urgency_reason",
    "troubleshooting_response"
]


def analyze_ticket(ticket_text):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please configure it in your .env file."
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a professional IT Support AI Assistant.

Analyze this IT support ticket:

{ticket_text}

Return exactly one JSON object with these fields:

1. category: choose exactly one:
{CATEGORIES}

2. sentiment: choose exactly one:
{SENTIMENTS}

3. urgency: choose exactly one:
{URGENCY_LEVELS}

4. urgency_reason: briefly explain the urgency.

5. troubleshooting_response: provide clear, practical troubleshooting
steps. Do not claim the issue is fixed. Never ask for passwords,
API keys, or other sensitive information.

Return valid JSON only, without Markdown fences.

Example structure:
{{
  "category": "Network",
  "sentiment": "Negative",
  "urgency": "High",
  "urgency_reason": "The user cannot access a required business service.",
  "troubleshooting_response": "Check your network connection and VPN settings."
}}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    result_text = response.text

    if not result_text:
        raise ValueError("Gemini returned an empty response.")

    result_text = result_text.strip()

    result_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        result_text,
        flags=re.IGNORECASE
    ).strip()

    match = re.search(r"\{.*\}", result_text, re.DOTALL)

    if not match:
        raise ValueError("Gemini did not return valid JSON.")

    result = json.loads(match.group())

    for field in REQUIRED_FIELDS:
        if field not in result:
            raise ValueError(f"Missing field in Gemini response: {field}")

    if result["category"] not in CATEGORIES:
        raise ValueError(f"Invalid category: {result['category']}")

    if result["sentiment"] not in SENTIMENTS:
        raise ValueError(f"Invalid sentiment: {result['sentiment']}")

    if result["urgency"] not in URGENCY_LEVELS:
        raise ValueError(f"Invalid urgency: {result['urgency']}")

    return result
