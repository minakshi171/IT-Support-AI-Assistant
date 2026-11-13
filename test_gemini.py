
import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

models_to_try = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash"
]

prompt = """
Explain in two sentences why an IT helpdesk
uses ticket classification.
"""

for model in models_to_try:
    try:
        print(f"\nTesting model: {model}")

        response = client.models.generate_content(
            model=model,
            contents=prompt
        )

        print("\nSUCCESS!")
        print("Model:", model)
        print("Response:", response.text)
        break

    except Exception as error:
        print("Model unavailable:", error)

        if model == models_to_try[-1]:
            print("\nAll model attempts failed.")
            print("Please try again later.")
