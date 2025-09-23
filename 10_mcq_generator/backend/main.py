import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai
import wikipediaapi
import logging
from typing import List
load_dotenv()

app = FastAPI()

# OpenRouter base URL (constant)
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='New/1.0 (new@example.com) Python wikipediaapi module'
)

# Pydantic models
class WikiURL(BaseModel):
    url: str

class MCQRequest(BaseModel):
    context: str
    count: int = 10
    model: str = "openai/gpt-4"

class MCQItemWithUserAnswer(BaseModel):
    question: str
    options: List[str]
    answer: str       # Correct answer
    user_answer: str  # User submitted answer

class GradeRequest(BaseModel):
    mcqs: List[MCQItemWithUserAnswer]
    model: str = "openai/gpt-4"

def extract_wikipedia_text(wiki_url: str) -> str:
    try:
        # Extract the page title from URL
        page_title = wiki_url.rsplit('/', 1)[-1].replace('_', ' ')
        page = wiki_wiki.page(page_title)
        if not page.exists():
            raise HTTPException(status_code=404, detail="Wikipedia page not found")
        # Combine sections and summaries as plain text
        text = page.summary + "\n"
        for section in page.sections:
            text += section.text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract Wikipedia text: {e}")

@app.post("/extract_text/")
def extract_text(data: WikiURL):
    text = extract_wikipedia_text(data.url)
    return {"text": text[:5000]}  # truncate for LLM input limits


@app.post("/generate_mcqs/")
async def generate_mcqs(data: MCQRequest, request: Request):
    user_api_key = request.headers.get("X-API-Key")
    if not user_api_key:
        raise HTTPException(status_code=400, detail="Missing OpenRouter API key in headers")

    openai.api_key = user_api_key
    openai.api_base = OPENROUTER_API_BASE

    prompt = (
        f"Read the following material and generate {data.count} multiple-choice questions (MCQs). "
        "For each MCQ, provide 4 options and indicate the correct answer. Present output in JSON in the format: "
        "[{{\"question\": ..., \"options\": [...], \"answer\": ...}}, ...].\n\n"
        f"Material:\n{data.context[:3500]}"
    )

    try:
        logging.info(f"Prompt sent:\n{prompt}")
        response = openai.ChatCompletion.create(
            model=data.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500
        )
        mcqs = response['choices'][0]['message']['content']
        logging.info(f"MCQs received:\n{mcqs}")
        return {"mcqs": mcqs}
    except Exception as e:
        logging.error(f"Error during MCQ generation: {e}")
        raise HTTPException(status_code=500, detail=f"MCQ generation failed: {e}")

@app.post("/grade_mcqs/")
async def grade_mcqs(data: GradeRequest, request: Request):
    user_api_key = request.headers.get("X-API-Key")
    if not user_api_key:
        raise HTTPException(status_code=400, detail="Missing OpenRouter API key in headers")

    openai.api_key = user_api_key
    openai.api_base = OPENROUTER_API_BASE

    # Construct a prompt that lets AI grade the submitted answers
    grading_prompt = "Grade the following multiple-choice questions with user's answers. For each question, say Correct or Incorrect. Then give total score:\n\n"
    for i, mcq in enumerate(data.mcqs):
        grading_prompt += (
            f"Q{i+1}: {mcq.question}\n"
            f"Options: {mcq.options}\n"
            f"Correct Answer: {mcq.answer}\n"
            f"User Answer: {mcq.user_answer}\n\n"
        )
    grading_prompt += "Provide a summary: Number of correct answers, total questions, and percentage score."

    try:
        response = openai.ChatCompletion.create(
            model=data.model,
            messages=[{"role": "user", "content": grading_prompt}],
            temperature=0,
            max_tokens=500
        )
        grading_result = response['choices'][0]['message']['content']
        return {"grading_result": grading_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading failed: {e}")