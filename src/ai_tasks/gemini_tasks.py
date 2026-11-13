
import os
import json
from dotenv import load_dotenv
from google import genai

# Load the API key from the .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.5-flash-lite"


def analyze_ticket(ticket_text):
    """
    Performs three IT support NLP tasks:
    1. Ticket classification
    2. Sentiment and urgency analysis
    3. Troubleshooting response generation
    """

    prompt = f"""
You are an IT Support AI Assistant.

Analyze the following IT support ticket.

Ticket:
{ticket_text}

Perform these three tasks:

1. CLASSIFICATION:
Choose exactly one category:
Network, Software, Hardware, Access Management, Security.

2. SENTIMENT AND URGENCY:
Identify the sentiment as Positive, Neutral, or Negative.
Choose urgency as Low, Medium, High, or Critical.
Give a short reason for the urgency.

3. TROUBLESHOOTING:
Generate a clear, polite and practical IT support response.
Do not claim that an issue has been fixed.
Do not ask for passwords or sensitive information.

Return ONLY valid JSON in this exact structure:

{{
    "category": "Network",
    "sentiment": "Negative",
    "urgency": "High",
    "urgency_reason": "The user cannot access a critical business service.",
    "troubleshooting_response": "Please check your network connection..."
}}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    result_text = response.text.strip()

    # Remove Markdown code fences if the model includes them
    if result_text.startswith("```"):
        result_text = result_text.replace("```json", "", 1)
        result_text = result_text.replace("```", "").strip()

    result = json.loads(result_text)

    required_fields = [
        "category",
        "sentiment",
        "urgency",
        "urgency_reason",
        "troubleshooting_response"
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing field in Gemini response: {field}")

    return result
