
from src.ai_tasks.hf_models import (
    classify_it_ticket,
    analyse_sentiment
)

sample_ticket = """
I cannot connect to the company VPN.
This is urgent because I cannot access my work applications.
"""

print("\n========== IT TICKET CLASSIFICATION ==========")

try:
    category_result = classify_it_ticket(sample_ticket)

    for item in category_result:
        print(
            f"Category: {item.label} | "
            f"Confidence: {item.score:.4f}"
        )

except Exception as error:
    print("Classification error:", error)


print("\n========== SENTIMENT ANALYSIS ==========")

try:
    sentiment_result = analyse_sentiment(sample_ticket)

    for item in sentiment_result:
        print(
            f"Sentiment: {item.label} | "
            f"Confidence: {item.score:.4f}"
        )

except Exception as error:
    print("Sentiment error:", error)
