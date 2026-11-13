import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import pipeline

print("===================================")
print(" IT SUPPORT AI - LOCAL MODEL TEST")
print("===================================")

print("PyTorch:", torch.__version__)
print("Transformers model loading...")

chatbot = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=-1
)

print("Model loaded successfully!")

messages = [
    {
        "role": "system",
        "content": "You are a professional IT Support Assistant. Give simple and practical troubleshooting solutions."
    },
    {
        "role": "user",
        "content": "My laptop is connected to Wi-Fi but the internet is not working. How can I fix it?"
    }
]

print("\nGenerating AI response...\n")

response = chatbot(
    messages,
    max_new_tokens=150,
    do_sample=False
)

print("========== AI RESPONSE ==========")
print(response[0]["generated_text"][-1]["content"])
print("===================================")
