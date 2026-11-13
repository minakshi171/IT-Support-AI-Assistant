
import json
import re

import streamlit as st
from transformers import pipeline

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


@st.cache_resource
def load_local_model():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device=-1
    )


def analyze_ticket_local(ticket_description):

    chatbot = load_local_model()

    prompt = f"""
Analyze this IT support ticket:

{ticket_description}

Return exactly one JSON object with these fields:

category: choose exactly one of
{CATEGORIES}

sentiment: choose exactly one of
{SENTIMENTS}

urgency: choose exactly one of
{URGENCY_LEVELS}

urgency_reason: briefly explain the selected urgency.

troubleshooting_response: provide clear, practical troubleshooting
steps. Do not claim the issue is fixed. Never ask for passwords,
API keys, or other sensitive information.

Return valid JSON only. Do not include Markdown or text outside JSON.

Example:
{{
  "category": "Network",
  "sentiment": "Negative",
  "urgency": "High",
  "urgency_reason": "The user cannot access a required business service.",
  "troubleshooting_response": "Check your network connection and VPN settings."
}}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional IT support assistant. "
                "Follow the required JSON format and allowed labels exactly."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = chatbot(
        messages,
        max_new_tokens=350,
        do_sample=False
    )

    generated = response[0]["generated_text"]

    if isinstance(generated, list):
        generated_text = generated[-1]["content"]
    else:
        generated_text = str(generated)

    match = re.search(r"\{.*\}", generated_text, re.DOTALL)

    if not match:
        raise ValueError(
            "Qwen did not return valid JSON. Please try again."
        )

    result = json.loads(match.group())

    for field in REQUIRED_FIELDS:
        if field not in result:
            raise ValueError(f"Missing field in Qwen response: {field}")

    if result["category"] not in CATEGORIES:
        raise ValueError(f"Invalid category: {result['category']}")

    if result["sentiment"] not in SENTIMENTS:
        raise ValueError(f"Invalid sentiment: {result['sentiment']}")

    if result["urgency"] not in URGENCY_LEVELS:
        raise ValueError(f"Invalid urgency: {result['urgency']}")

    return result
