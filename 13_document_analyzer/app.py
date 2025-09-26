import streamlit as st
import requests
from PyPDF2 import PdfReader

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "gpt-4o",
    "mistral-small",
    "distilbert-base-uncased",
    "t5-small",
    "x-ai/grok-4-fast:free"
]

def call_openrouter_api(prompt_text, api_key, model=FREE_MODELS, max_tokens=10000):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=json_data)

    if response.status_code != 200:
        st.error(f"API request failed with status {response.status_code}: {response.text}")
        return None

    try:
        result = response.json()
        return result['choices'][0]['message']['content']
    except ValueError:
        st.error("Failed to parse JSON response from API.")
        st.write("Raw response:")
        st.write(response.text)
        return None

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Streamlit App UI
st.title("Multiple Model-Based Document Analyzer with OpenRouter")

api_key = st.text_input("Enter your OpenRouter API Key", type="password")

if api_key:
    model = st.selectbox("Select AI Model", FREE_MODELS)

    uploaded_file = st.file_uploader("Upload PDF or TXT Document", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")

        st.markdown("### Document Text Preview")
        st.write(text[:1000] + ("..." if len(text) > 1000 else ""))

        if st.button("Analyze Document"):
            with st.spinner("Analyzing, please wait..."):
                entity_prompt = f"Extract named entities with types (PERSON, ORG, DATE, LOCATION) from this text:\n\n{text}"
                entities = call_openrouter_api(entity_prompt, api_key, model)

                summary_prompt = f"Summarize the following text briefly:\n\n{text}"
                summary = call_openrouter_api(summary_prompt, api_key, model)

                sentiment_prompt = f"Analyze the sentiment of this text. Is it Positive, Negative, or Neutral?\n\n{text}"
                sentiment = call_openrouter_api(sentiment_prompt, api_key, model)

            if entities:
                st.markdown("### Named Entities")
                st.write(entities)
            if summary:
                st.markdown("### Document Summary")
                st.write(summary)
            if sentiment:
                st.markdown("### Sentiment Analysis")
                st.write(sentiment)

else:
    st.warning("Please enter your OpenRouter API Key to proceed.")
