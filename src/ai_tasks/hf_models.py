
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

CATEGORY_MODEL = "facebook/bart-large-mnli"

SENTIMENT_MODEL = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


def get_client():
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN not found in .env file.")

    return InferenceClient(token=token)


def classify_it_ticket(text):

    labels = [
        "Network",
        "Software",
        "Hardware",
        "Access Management",
        "Security"
    ]

    results = get_client().zero_shot_classification(
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

    results = get_client().text_classification(
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
        "Critical",
        "High",
        "Medium",
        "Low"
    ]

    descriptions = {
        "Critical": "Critical priority IT incident affecting business operations",
        "High": "High priority IT issue affecting work",
        "Medium": "Medium priority IT issue affecting an individual user",
        "Low": "Low priority IT request or minor inconvenience"
    }

    results = get_client().zero_shot_classification(
        text=text,
        candidate_labels=[
            descriptions[label] for label in labels
        ],
        model=CATEGORY_MODEL
    )

    best = max(results, key=lambda item: item.score)

    selected_label = next(
        label
        for label, description in descriptions.items()
        if description == best.label
    )

    return {
        "label": selected_label,
        "score": best.score
    }
