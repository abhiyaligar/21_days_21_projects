# AI Powered ATS Friendly Resume Builder

This is a Streamlit web application that helps job seekers generate ATS (Applicant Tracking System) optimized resumes using AI. It leverages the OpenRouter API for AI-powered resume content generation and uses fpdf2 for lightweight PDF resume creation.

## Features

- Input candidate details including name, contact info, professional summary, skills, experience, and education.
- Optional upload of existing resume (PDF or DOCX) for enhanced parsing.
- Integration with GitHub to fetch user profile and repository info.
- AI-powered resume optimization for ATS friendliness with keyword focus.
- Generates well-formatted Markdown resumes and converts them to PDF.
- Downloadable PDF resume with clean layout suitable for recruiters.
- Secure data processing with no information stored on servers.

## How It Works

1. User fills candidate and job target info.
2. Optionally uploads an existing resume for parsing details.
3. The app calls the OpenRouter API to generate a keyword-optimized Markdown resume based on user inputs and job requirements.
4. Markdown is converted to PDF using the fpdf2 library.
5. User can preview and download the ATS-optimized resume PDF.

## Installation

Clone The Project from the github 

```
git clone https://github.com/abhiyaligar/21_days_21_projects.git
cd 14_resume_builder
```

Install dependencies with:

```
pip install -r requirements.txt
```


## Usage

Start the app by running:
```
streamlit run app.py
```

Fill in the form fields and click "Generate ATS-Optimized Resume" to create your resume.

## Notes

- Requires a valid OpenRouter API key.
- Resume parsing supports PDF and DOCX formats.
- Outputs clean, simple PDFs optimized for ATS systems.

