import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load system prompt from prompt.txt
def load_prompt(filepath="prompt.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Failed to load prompt file: {e}")
        return ""

SYSTEM_PROMPT = load_prompt()

st.set_page_config(page_title="AI Marketing Slogan Generator", layout="wide")
st.title("AI Marketing Slogan Generator")
st.markdown(
    "Generate strategic, creative, and high-impact slogans using open-source LLMs and the OpenRouter API."
)
st.info("Complete all fields below for best results.")

# API Key input (override env)
user_api_key_input = st.text_input(
    "Enter your OpenRouter API Key:",
    type="password",
    placeholder="sk-..."
)

# Determine which API key to use
api_key = user_api_key_input.strip() or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.warning("Please enter an API key above or set OPENROUTER_API_KEY in environment.")
    st.stop()

with st.form("product_form"):
    st.header("Product/Service Information")
    product_name = st.text_input("Name")
    product_details = st.text_area("Core details (features, benefits, use case, specs, price, stage, awards)", height=80)
    unique_selling = st.text_area("Unique Value Proposition / Why Different?", height=40)
    target_market = st.text_input("Target Market Segment / Industry")
    problem_solved = st.text_area("Primary Customer Problem Solved", height=40)
    emotional_benefit = st.text_area("Key Emotional Benefit/Brand Feeling", height=40)
    brand_voice = st.text_input("Brand Voice/Tone (e.g. playful, premium, casual, etc.)")
    mission = st.text_input("Brand Mission (optional)")
    vision = st.text_input("Brand Vision (optional)")
    competitors = st.text_area("Key Competitors & Their Slogans (optional)", height=40)
    specific_goal = st.text_input("Campaign/Business Goal (optional)")
    usecase = st.text_input("Application (website, ad, social media, etc.)")
    additional = st.text_area("Any other relevant info?", height=40)

    model = st.selectbox(
        "Choose an Open Source LLM",
        [
            "openai/gpt-oss-20b:free",
            "x-ai/grok-4-fast:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "google/gemma-3-4b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free"
        ]
    )

    submit = st.form_submit_button("Generate Slogans")

if submit:
    with st.spinner("Generating expert slogans. This may take up to a minute..."):
        user_content = f"""
Product Name: {product_name}
Core Details: {product_details}
Unique Selling Proposition: {unique_selling}
Product Category/Market: {target_market}
Problem Solved: {problem_solved}
Emotional Benefit: {emotional_benefit}
Brand Voice/Tone: {brand_voice}
Mission: {mission}
Vision: {vision}
Competitors: {competitors}
Business/Campaign Goal: {specific_goal}
Primary Use Case: {usecase}
Additional Info: {additional}
        """

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        }

        response = requests.post(API_URL, json=payload, headers=headers)
        if response.ok:
            raw_result = response.json()
            try:
                output = raw_result["choices"][0]["message"]["content"]
                st.success("Slogans generated!")
                st.markdown("#### Output from LLM")
                st.markdown(output)
            except Exception as e:
                st.error("Unexpected response structure.")
                st.write(raw_result)
        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)
else:
    st.caption("The AI will analyze your product and brand details, then generate structured slogan options and guidance.")
