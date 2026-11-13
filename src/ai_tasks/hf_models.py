
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

client = InferenceClient(token=HF_TOKEN)

CATEGORY_MODEL = "facebook/bart-large-mnli"

SENTIMENT_MODEL = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


def classify_it_ticket(text):
    labels = [
        "Network issue",
        "Software issue",
        "Hardware issue",
        "Access management issue",
        "Security issue"
    ]

    results = client.zero_shot_classification(
        text=text,
        candidate_labels=labels,
        model=CATEGORY_MODEL
    )

    best = max(results, key=lambda item: item.score)

    return {
        "label": best.label,
        "score": best.score
    }


def analyse_sentiment(text):
    results = client.text_classification(
        text=text,
        model=SENTIMENT_MODEL
    )

    best = max(results, key=lambda item: item.score)

    return {
        "label": best.label,
        "score": best.score
    }


def classify_urgency(text):
    labels = [
        "Critical priority IT incident affecting business operations",
        "High priority IT issue affecting work",
        "Medium priority IT issue affecting an individual user",
        "Low priority IT request or minor inconvenience"
    ]

    results = client.zero_shot_classification(
        text=text,
        candidate_labels=labels,
        model=CATEGORY_MODEL
    )

    best = max(results, key=lambda item: item.score)

    return {
        "label": best.label,
        "score": best.score
    }
