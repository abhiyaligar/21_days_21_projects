from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from newspaper import Article
import os
import requests
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class NewsSummary(BaseModel):
    title: str
    url: str
    publishedAt: str
    summary: Optional[str]

def summarize_text_openrouter(text: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "x-ai/grok-4-fast:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following news article text briefly:\n\n{text}"}
        ],
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "stream": False
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        return "Summary unavailable"

@router.get("/latest/{company_symbol}", response_model=List[NewsSummary])
async def get_latest_news(company_symbol: str):

    n_days = 3
    from_date = (datetime.utcnow() -timedelta(days=n_days)).date()
    all_articles = newsapi.get_everything(
        q=company_symbol,
        from_param=str(from_date),
        language="en",
        sort_by="relevancy",
        page_size=5,
    )
    if not all_articles or all_articles.get("status") != "ok":
        raise HTTPException(status_code=500, detail="Failed to fetch news")

    summaries = []
    for article_data in all_articles["articles"]:
        try:
            article = Article(article_data["url"])
            article.download()
            article.parse()
            summary_text = summarize_text_openrouter(article.text)
        except Exception:
            summary_text = "Summary unavailable"

        summaries.append(NewsSummary(
            title=article_data["title"],
            url=article_data["url"],
            publishedAt=article_data["publishedAt"],
            summary=summary_text
        ))
    return summaries
