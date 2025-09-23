# AI-Powered MCQ Generator Web App

An interactive web application that allows users to input a Wikipedia URL or document link, automatically extracts the text, generates multiple-choice questions (MCQs) using AI models via OpenRouter API, and grades user-submitted answers. Built with a FastAPI backend and Streamlit frontend.

---

## Features

- **Wikipedia Text Extraction**: Retrieve page summaries and contents via the Wikipedia API.
- **Dynamic MCQ Generation**: Generate 10 (or more) MCQs based on the input content using customizable AI models.
- **User-defined AI Models & API Keys**: Users provide their own OpenRouter API keys and specify which AI model to use.
- **Answer Submission & AI Grading**: Submit answers for AI-based grading with detailed correctness feedback and score summary.
- **Incremental MCQ Generation**: Generate additional questions on demand.
- **Modern, Responsive UI**: Interactive quiz interface built with Streamlit.

---

## Tech Stack

| Component       | Technology         |
|-----------------|--------------------|
| Backend API     | Python, FastAPI    |
| AI Integration  | OpenRouter API (LLM access) |
| Wikipedia Data  | `wikipedia-api` Python library |
| Frontend UI     | Streamlit          |
| Request Handling| `requests` library  |
| Environment    | `python-dotenv`    |


---

## Getting Started

### Prerequisites

- Python 3.8 or above
- An OpenRouter API key ([Sign up here](https://openrouter.ai/))
- Git and GitHub account for deployment (optional)

### Installation (Local Development)

1. **Clone the repository**

```
git clone https://github.com/abhiyaligar/21_days_21_projects.git
cd 10_mcq_generator
```


2. **Setup Environment**

```
cd 10_mcq_generator
python -m venv env
source env/bin/activate # On Windows use env\Scripts\activate
pip install -r requirements.txt
```


3. **Run Backend FastAPI Server**

```
cd backend
uvicorn main:app --reload
```

4. **Run Streamlit App**

```
cd frontend
streamlit run app.py
```
---

## Usage

1. Open the Streamlit app link displayed after running.

2. Enter your **OpenRouter API key** and optionally the AI model name (default: `openai/gpt-4`).

3. Paste a **Wikipedia article URL** (for example, `https://en.wikipedia.org/wiki/FastAPI`).

4. Click **Extract Text and Generate MCQs**.

5. MCQs will display with answer options. Select your answers.

6. Click **Submit Answers** to get AI-based grading and feedback.

7. Use **Create 10 More MCQs** to expand the question set.

---

## Troubleshooting

- **500 Internal Server Errors**: Check backend logs for invalid prompts or API quota issues.
- **CORS Errors**: Ensure backend allows requests from frontend domain.
- **Parsing MCQs fails**: Verify AI model outputs correct JSON format, check prompt correctness.
- **Slow responses**: AI model latency depends on provider and request size.

---

## Contributing

Contributions are welcome! Feel free to fork the repo, create feature branches, and submit pull requests.

---

## Acknowledgments

- [OpenRouter](https://openrouter.ai) for unified AI models API
- [Wikipedia-API](https://pypi.org/project/wikipedia-api/) Python library
- [FastAPI](https://fastapi.tiangolo.com) framework
- [Streamlit](https://streamlit.io) for frontend UI

---
