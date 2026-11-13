import json
import re
import streamlit as st
from transformers import pipeline


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
You are a professional IT Support AI Assistant.

Analyze the following IT support ticket:

{ticket_description}

Return your analysis in valid JSON only.
Do not include Markdown or explanations outside JSON.

Use this exact JSON structure:

{{
    "category": "IT ticket category",
    "sentiment": "Positive, Neutral or Negative",
    "urgency": "Low, Medium or High",
    "urgency_reason": "Explain why this urgency was selected",
    "troubleshooting_response": "Give clear step-by-step troubleshooting instructions"
}}

Make the troubleshooting instructions practical and easy to understand.
"""

    messages = [
        {
            "role": "system",
            "content": "You are an accurate IT support ticket analysis assistant. Always return valid JSON."
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

    generated_text = response[0]["generated_text"][-1]["content"]

    # Extract the JSON response
    match = re.search(r"\{.*\}", generated_text, re.DOTALL)

    if not match:
        raise ValueError(
            "The local AI model did not return valid JSON. Please try again."
        )

    result = json.loads(match.group())

    required_fields = [
        "category",
        "sentiment",
        "urgency",
        "urgency_reason",
        "troubleshooting_response"
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing field in AI response: {field}")

    return result
