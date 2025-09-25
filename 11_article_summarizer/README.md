# Article Summarizer Web App

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen)](https://summarizeart.streamlit.app/)


A simple Streamlit web application that fetches and summarizes online articles using the [newspaper3k](https://newspaper.readthedocs.io/en/latest/) library for article extraction and Hugging Face’s T5 transformer model for abstractive summarization.

## Features

- Extracts the full article text from any URL.
- Uses state-of-the-art T5-based text summarization for concise summaries.
- Interactive web interface built with Streamlit for easy usage.
- Lightweight, runs locally with open-source libraries.

## Technologies

- Python 3.8+
- Streamlit - Web app framework
- Newspaper3k - Article scraping and parsing
- Transformers (Hugging Face) - Pretrained NLP models (T5)
- PyTorch - Model backend

## Installation

1. Clone the repository:
```
git clone https://github.com/abhiyaligar/21_days_21_projects.git
cd 11_article_summarizer
```


2. Create and activate a Python virtual environment:
```
python -m venv env
source env/bin/activate # On Windows use env\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```


## Usage

Run the Streamlit app locally:
```
streamlit run app.py
```

- Paste an article URL into the input box.
- Click the "Summarize" button.
- View the generated summary below.
