# Multiple Model-Based Document Analyzer with OpenRouter  

This project is a **Streamlit web application** that allows users to upload PDF or TXT documents and analyze them with multiple AI models available on **OpenRouter**. The application supports **named entity recognition (NER)**, **document summarization**, and **sentiment analysis**, all powered by large language models (LLMs).  

---

### Features
- Upload **PDF** or **TXT** documents.  
- Extract full raw text content from uploaded files.  
- Preview document text before analysis.  
- Choose from multiple free AI models hosted on OpenRouter:
  - gpt-4o  
  - mistral-small  
  - distilbert-base-uncased  
  - t5-small  
  - x-ai/grok-4-fast:free  
- AI-powered tasks:
  - Named Entity Extraction (PERSON, ORG, DATE, LOCATION)  
  - Summarization of the document  
  - Sentiment Analysis (Positive, Negative, or Neutral)  

---

### Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Backend Processing:** OpenRouter API  
- **Libraries Used:**
  - `requests` – API calls  
  - `PyPDF2` – Extracting PDF text  
  - `streamlit` – Web interface  

---

### Installation

1. Clone the repository:
```
git clone https://github.com/abhiyaligar/21_days_21_projects.git
cd 13_document_analyzer
```

2. Create and activate a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate # On Linux/Mac
venv\Scripts\activate # On Windows
```

3. Install dependencies:
```
pip install -r requirements.txt
```

---

### Usage

1. Run the Streamlit app:
```
streamlit run app.py
```

2. Open your browser at `http://localhost:8501` (Streamlit will also display the link).  

3. Steps inside the app:
- Enter your **OpenRouter API Key**  
- Select a model from the dropdown  
- Upload a PDF or TXT file  
- Preview text  
- Click **Analyze Document**  

---

### Example Output
When analyzing a document, the app will display:
- **Named Entities** (e.g., PERSON: John Doe, LOCATION: Paris)  
- **Summary** (brief overview of document contents)  
- **Sentiment** (Positive / Negative / Neutral)  

---

### API Key
You need an **OpenRouter API key** to use AI models. Get your key by creating an account on OpenRouter and copying the API token.  

In the app, paste your API key into the input box to start analysis.  

---
