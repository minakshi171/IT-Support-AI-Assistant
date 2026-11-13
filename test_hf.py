
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load the secret token from .env
load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    print("ERROR: Hugging Face token not found.")
else:
    try:
        api = HfApi(token=token)
        user = api.whoami()

        print("SUCCESS: Hugging Face API connected!")
        print("Authenticated account:", user["name"])

    except Exception as e:
        print("Connection failed:", e)
