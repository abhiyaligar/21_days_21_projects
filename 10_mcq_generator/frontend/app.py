import streamlit as st
import requests
import json

API_BASE_URL = "http://localhost:8000"  # Change if backend deployed elsewhere

st.title("AI-Powered MCQ Generator")

api_key = st.text_input("Enter your OpenRouter API Key", type="password")
model_name = st.text_input("Enter AI Model Name", value="openai/gpt-4")
wiki_url = st.text_input("Enter Wikipedia URL or Document Link")

headers = {"X-API-Key": api_key} if api_key else {}

def extract_text_from_url(url):
    response = requests.post(
        f"{API_BASE_URL}/extract_text/",
        json={"url": url},
        headers=headers
    )
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        st.error("Failed to extract text from URL")
        return ""

def generate_mcqs(context, count=10):
    json_payload = {
        "context": context,
        "count": count,
        "model": model_name
    }
    response = requests.post(
        f"{API_BASE_URL}/generate_mcqs/",
        json=json_payload,
        headers=headers
    )
    if response.status_code == 200:
        mcqs_json_str = response.json().get("mcqs", "[]")
        try:
            mcqs = json.loads(mcqs_json_str)
            return mcqs
        except json.JSONDecodeError:
            st.error("Failed to parse MCQs from API response")
            return []
    else:
        st.error("Failed to generate MCQs")
        return []

def grade_mcqs(mcqs, answers):
    # Prepare payload combining each MCQ with user answer
    graded_payload = {
        "model": model_name,
        "mcqs": []
    }
    for mcq, user_answer in zip(mcqs, answers):
        mcq_with_answer = {
            "question": mcq["question"],
            "options": mcq["options"],
            "answer": mcq["answer"],
            "user_answer": user_answer if user_answer else ""
        }
        graded_payload["mcqs"].append(mcq_with_answer)

    response = requests.post(
        f"{API_BASE_URL}/grade_mcqs/",
        json=graded_payload,
        headers=headers
    )
    if response.status_code == 200:
        return response.json().get("grading_result", "No result returned")
    else:
        st.error("Failed to grade MCQs")
        return ""

if "mcqs" not in st.session_state:
    st.session_state["mcqs"] = []
    st.session_state["answers"] = []

if st.button("Extract Text and Generate MCQs"):
    if not api_key:
        st.warning("Please enter your OpenRouter API key")
    elif not wiki_url:
        st.warning("Please enter a valid URL")
    else:
        with st.spinner("Extracting text..."):
            context = extract_text_from_url(wiki_url)
        if context:
            st.success("Text extracted successfully!")
            st.write(context[:1000] + " ...")

            mcqs = generate_mcqs(context)
            if mcqs:
                st.session_state["mcqs"] = mcqs
                st.session_state["answers"] = [None] * len(mcqs)
            else:
                st.warning("No MCQs generated.")
        else:
            st.warning("Could not extract text from the URL.")

if st.session_state["mcqs"]:
    st.header("Multiple Choice Questions")
    for i, mcq in enumerate(st.session_state["mcqs"]):
        st.write(f"**Q{i+1}. {mcq['question']}**")
        st.session_state["answers"][i] = st.radio(
            f"Select answer for Q{i+1}",
            mcq["options"],
            key=f"mcq_{i}"
        )

    if st.button("Submit Answers"):
        if None in st.session_state["answers"]:
            st.warning("Please answer all questions before submitting.")
        else:
            with st.spinner("Grading your answers..."):
                result = grade_mcqs(st.session_state["mcqs"], st.session_state["answers"])
            st.subheader("Grading Result")
            st.text(result)

if st.button("Create 10 More MCQs") and st.session_state["mcqs"]:
    new_mcqs = generate_mcqs(" ".join([mcq["question"] for mcq in st.session_state["mcqs"]]))
    if new_mcqs:
        st.session_state["mcqs"].extend(new_mcqs)
        st.session_state["answers"].extend([None] * len(new_mcqs))
