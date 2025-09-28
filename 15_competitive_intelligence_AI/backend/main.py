from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth import router as auth_router
from companies import router as companies_router
from news import router as news_router

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router, prefix="/auth")
app.include_router(companies_router, prefix="/companies")
app.include_router(news_router, prefix="/news")

@app.get("/")
def read_root():
    return {"message": "Competitive Intelligence API Running"}
