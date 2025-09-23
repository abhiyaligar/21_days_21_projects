from pydantic import BaseModel, HttpUrl
from typing import List


class WikiURL(BaseModel):
    url: HttpUrl  # Validates input as a proper URL

class MCQItem(BaseModel):
    question: str
    options: List[str]
    answer: str

class MCQRequest(BaseModel):
    context: str
    count: int = 10  # Default number of MCQs to generate

class MCQResponse(BaseModel):
    mcqs: List[MCQItem]

class AnswersSubmission(BaseModel):
    user_answers: List[str]  # User's selected answers in order
    mcqs: List[MCQItem]      # Original MCQs for reference

class MCQItemWithUserAnswer(BaseModel):
    question: str
    options: List[str]
    answer: str       # Correct answer
    user_answer: str  # User submitted answer

class GradeRequest(BaseModel):
    mcqs: List[MCQItemWithUserAnswer]
    model: str = "x-ai/grok-4-fast:free"
