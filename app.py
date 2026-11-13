
import streamlit as st

st.set_page_config(
    page_title="IT Support AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 IT Support AI Assistant")

st.write(
    "AI-powered IT ticket classification, sentiment analysis, "
    "urgency detection and troubleshooting."
)

st.divider()

st.subheader("Select AI Model")

ai_model = st.selectbox(
    "Choose your AI provider",
    [
        "Google Gemini (Cloud AI)",
        "Hugging Face Qwen (Local AI)"
    ]
)

if ai_model == "Google Gemini (Cloud AI)":
    st.info("Using Google Gemini through cloud inference.")
else:
    st.info("Using Hugging Face Qwen locally on your MacBook CPU.")

st.divider()

st.subheader("Submit an IT Support Ticket")

user_name = st.text_input("Enter your name")

ticket_description = st.text_area(
    "Describe your IT problem",
    placeholder="Example: I cannot connect to company VPN.",
    height=150
)

if st.button("Analyze Ticket", type="primary"):

    if not ticket_description.strip():
        st.warning("Please enter your IT problem.")

    else:
        try:
            with st.spinner(f"{ai_model} is analyzing your ticket..."):

                if ai_model == "Google Gemini (Cloud AI)":
                    from src.ai_tasks.gemini_tasks import analyze_ticket
                    result = analyze_ticket(ticket_description)

                else:
                    from src.ai_tasks.local_model_tasks import analyze_ticket_local
                    result = analyze_ticket_local(ticket_description)

            st.success("Ticket analysis completed successfully!")
            st.divider()
            st.subheader("AI Analysis Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Ticket Category", result["category"])

            with col2:
                st.metric("Sentiment", result["sentiment"])

            with col3:
                st.metric("Urgency", result["urgency"])

            st.subheader("Urgency Explanation")
            st.write(result["urgency_reason"])

            st.subheader("AI-Generated Troubleshooting Response")
            st.info(result["troubleshooting_response"])

            st.success(f"Analysis completed using {ai_model}.")

        except Exception as error:
            st.error("Something went wrong while analyzing the ticket.")
            st.exception(error)

st.divider()

st.caption(
    "IT Support AI Assistant | Google Gemini + Hugging Face "
    "Local AI | M.Tech Assignment 2"
)
