
from src.ai_tasks.gemini_tasks import analyze_ticket

sample_ticket = """
I cannot connect to the company VPN since this morning.
I have an important client meeting and cannot access internal applications.
Please resolve this urgently.
"""

print("Testing IT Support AI Assistant...\n")

result = analyze_ticket(sample_ticket)

print("AI ANALYSIS RESULTS")
print("-------------------")
print("Category:", result["category"])
print("Sentiment:", result["sentiment"])
print("Urgency:", result["urgency"])
print("Urgency reason:", result["urgency_reason"])
print("\nAI-generated troubleshooting response:")
print(result["troubleshooting_response"])
