import streamlit as st
from pyresparser import ResumeParser
import requests
import tempfile
import os
import base64
import markdown
from fpdf import FPDF
from bs4 import BeautifulSoup
import re

class ResumePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def set_resume_font(self, style='', size=11):
        """Set font - using basic Arial for maximum compatibility"""
        self.set_font('Arial', style, size)
        
    def header(self):
        pass  # No header for resume
        
    def footer(self):
        pass  # No footer for resume

def get_bullet_char():
    """Get ASCII-compatible bullet character"""
    return '-'  # Simple dash that works with all fonts

def clean_text(text):
    """Clean text for PDF generation with ASCII compatibility"""
    if not text:
        return ""
    
    # Remove extra whitespace and clean up text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Replace problematic Unicode characters with ASCII equivalents
    replacements = {
        ''': "'",    # Smart quote to regular quote
        ''': "'",    # Smart quote to regular quote
        '"': '"',    # Smart double quote to regular quote
        '"': '"',    # Smart double quote to regular quote
        '—': '-',    # Em dash to regular dash
        '–': '-',    # En dash to regular dash
        '•': '-',    # Unicode bullet to dash
        '…': '...',  # Ellipsis to three dots
        '©': '(c)',  # Copyright symbol
        '®': '(R)',  # Registered trademark
        '™': '(TM)', # Trademark symbol
    }
    
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Final cleanup - keep only ASCII characters
    try:
        # Encode to ASCII, replacing problematic characters
        text = text.encode('ascii', 'replace').decode('ascii')
        # Clean up any ? characters that resulted from replacement
        text = re.sub(r'\?+', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    except Exception:
        # Fallback: remove all non-ASCII characters
        text = ''.join(char for char in text if ord(char) < 128)
        return text

def md_to_pdf_fpdf2(md_text, output_pdf_path):
    """Convert markdown to PDF using fpdf2 with improved formatting and Unicode support"""
    try:
        # Convert markdown to HTML
        html = markdown.markdown(md_text, extensions=['extra'])
        soup = BeautifulSoup(html, 'html.parser')
        
        # Create PDF document
        pdf = ResumePDF()
        pdf.add_page()
        
        # Track current position
        line_height = 6
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol']):
            # Check if we need a new page
            if pdf.get_y() > 250:
                pdf.add_page()
            
            if element.name == 'h1':
                # Main title (candidate name)
                pdf.set_resume_font('B', 18)
                pdf.ln(5)
                text = clean_text(element.get_text())
                pdf.cell(0, 12, text, ln=True, align='C')
                pdf.ln(5)
                # Add underline
                pdf.line(20, pdf.get_y(), 190, pdf.get_y())
                pdf.ln(8)
                
            elif element.name == 'h2':
                # Section headers
                pdf.set_resume_font('B', 14)
                pdf.ln(8)
                text = clean_text(element.get_text())
                pdf.cell(0, 8, text, ln=True)
                pdf.ln(3)
                # Add subtle line under section header
                pdf.line(20, pdf.get_y(), 100, pdf.get_y())
                pdf.ln(5)
                
            elif element.name == 'h3':
                # Sub-headers
                pdf.set_resume_font('B', 12)
                pdf.ln(5)
                text = clean_text(element.get_text())
                pdf.cell(0, 7, text, ln=True)
                pdf.ln(2)
                
            elif element.name == 'p':
                # Regular paragraphs
                pdf.set_resume_font('', 11)
                text = clean_text(element.get_text())
                if text.strip():  # Only add non-empty paragraphs
                    pdf.multi_cell(0, line_height, text)
                    pdf.ln(3)
                    
            elif element.name in ['ul', 'ol']:
                # Lists
                pdf.set_resume_font('', 11)
                for i, li in enumerate(element.find_all('li')):
                    text = clean_text(li.get_text())
                    if text.strip():
                        # Use dash for bullets, numbers for ordered lists
                        if element.name == 'ul':
                            bullet = get_bullet_char()
                        else:
                            bullet = f'{i+1}.'
                        
                        # Set position for bullet
                        pdf.cell(8, line_height, bullet, ln=False)
                        
                        # Calculate remaining width for text
                        remaining_width = 190 - 8
                        
                        # Split long text if needed
                        if len(text) > 80:
                            # Use multi_cell for long text
                            current_y = pdf.get_y()
                            pdf.multi_cell(remaining_width, line_height, text)
                            pdf.ln(2)
                        else:
                            pdf.cell(remaining_width, line_height, text, ln=True)
                            pdf.ln(1)
                pdf.ln(5)
        
        # Save the PDF
        pdf.output(output_pdf_path)
        return output_pdf_path
        
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return None

def escape_curly_braces(text):
    if not text:
        return ""
    return text.replace("{", "{{").replace("}", "}}")

def parse_resume(file_path):
    try:
        return ResumeParser(file_path).get_extracted_data()
    except Exception:
        return {}

def fetch_github_info(github_url):
    if not github_url or 'github.com/' not in github_url:
        return ""
    try:
        username = github_url.split('github.com/')[-1].split('/')[0]
        r = requests.get(f"https://api.github.com/users/{username}", timeout=10)
        repos = requests.get(f"https://api.github.com/users/{username}/repos?per_page=100", timeout=10)
        
        if r.status_code != 200:
            return ""
            
        user = r.json()
        repo_titles = [repo['name'] for repo in repos.json()] if repos.status_code == 200 else []
        info = f'GitHub User: {user.get("login", "")}\nPublic Repos: {user.get("public_repos", 0)}\nTop Repos: {", ".join(repo_titles[:5])}'
        return info
    except Exception as e:
        st.warning(f"Could not fetch GitHub info: {str(e)}")
        return ""

def call_openrouter(prompt, api_key):
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        body = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1200,
        }
        r = requests.post(url, headers=headers, json=body, timeout=30)
        res = r.json()
        
        if "choices" in res and len(res["choices"]) > 0:
            return res["choices"][0]["message"]["content"]
        else:
            return f"AI response error: {res.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Exception calling AI API: {e}"

def download_button(file_path, label):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
            dl_link = f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(file_path)}">{label}</a>'
            st.markdown(dl_link, unsafe_allow_html=True)

# Streamlit app
st.set_page_config(page_title="ATS Resume Builder - fpdf2", layout="centered")
st.title("🚀 AI Powered ATS Friendly Resume Builder")
st.markdown("*Using fpdf2 for lightweight PDF generation*")

# Input fields
st.header("👤 Candidate Information")
col1, col2 = st.columns(2)

with col1:
    candidate_name = st.text_input("Candidate Name *", placeholder="John Doe")
    phone = st.text_input("Phone Number *", placeholder="+1 (555) 123-4567")
    email = st.text_input("Email Address *", placeholder="john.doe@email.com")

with col2:
    portfolio_url = st.text_input("Portfolio URL", placeholder="https://johndoe.com")
    github_url = st.text_input("GitHub URL", placeholder="https://github.com/johndoe")
    linkedin_url = st.text_input("LinkedIn URL", placeholder="https://linkedin.com/in/johndoe")

professional_summary = st.text_area("Professional Summary * (50-80 words)", 
                                  placeholder="Brief professional summary highlighting your key skills and experience...")

technical_skills = st.text_area("Technical Skills * (comma separated)", 
                               placeholder="Python, JavaScript, React, SQL, AWS...")

experience_data = st.text_area("Experience Data (Employment / Internships / Projects)", 
                             placeholder="Describe your work experience, internships, and projects...")

education_data = st.text_area("Education Data", 
                            placeholder="Your educational background...")

st.header("🎯 Target Position")
job_title = st.text_input("Target Job Title *", placeholder="Software Engineer")
company_name = st.text_input("Target Company Name", placeholder="Tech Company Inc.")
job_requirements = st.text_area("Job Requirements *", 
                               placeholder="Copy and paste the job description requirements...")
preferred_keywords = st.text_area("Preferred Keywords (comma separated)", 
                                 placeholder="Python, Machine Learning, API Development...")

st.header("🔑 API Configuration")
api_key = st.text_input("Enter your OpenRouter API Key *", type="password", 
                       help="Get your API key from https://openrouter.ai/")

st.header("📄 Resume Upload (Optional)")
uploaded_file = st.file_uploader("Upload Existing Resume", type=['pdf','docx'],
                                help="Upload your current resume to extract additional information")

# Generate button
st.markdown("---")
if st.button("🎨 Generate ATS-Optimized Resume", type="primary"):
    # Validation
    required_fields = [
        (candidate_name, "Candidate Name"),
        (phone, "Phone Number"),
        (email, "Email Address"),
        (professional_summary, "Professional Summary"),
        (technical_skills, "Technical Skills"),
        (job_title, "Job Title"),
        (job_requirements, "Job Requirements"),
        (api_key, "API Key")
    ]
    
    missing_fields = [field_name for field_value, field_name in required_fields if not field_value]
    
    if missing_fields:
        st.error(f"Please fill in the following required fields: {', '.join(missing_fields)}")
    else:
        with st.spinner("🔄 Processing your resume..."):
            # Parse uploaded resume if provided
            parsed_resume = {}
            if uploaded_file:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix="."+uploaded_file.name.split('.')[-1]) as tf:
                        tf.write(uploaded_file.read())
                        tf.flush()
                        parsed_resume = parse_resume(tf.name)
                        os.unlink(tf.name)  # Clean up temp file
                except Exception as e:
                    st.warning(f"Could not parse uploaded resume: {str(e)}")

            # Fetch GitHub info
            github_info = fetch_github_info(github_url) if github_url else ""

            # Prepare data for AI
            escaped_parsed_resume = escape_curly_braces(str(parsed_resume))
            escaped_github_info = escape_curly_braces(github_info)
            escaped_job_req = escape_curly_braces(job_requirements)

            # AI prompt
            prompt = f"""
You are an elite resume optimization specialist with expertise in ATS systems, 
hiring manager preferences, and modern recruitment practices. Your sole objective 
is to create a perfectly formatted, keyword-optimized **Markdown** resume that maximizes interview potential.

## CRITICAL OUTPUT REQUIREMENTS
**PRODUCE ONLY THE FINAL RESUME IN MARKDOWN FORMAT - NO EXPLANATIONS, NO COMMENTARY, NO ANALYSIS**

## INPUT DATA
**Candidate Information:**
- NAME: {candidate_name}
- PHONE: {phone}  
- EMAIL: {email}
- PORTFOLIO: {portfolio_url}
- GITHUB: {github_url}
- LINKEDIN: {linkedin_url}
- PROFESSIONAL_SUMMARY: {professional_summary}
- TECHNICAL_SKILLS: {technical_skills}
- EXPERIENCE_DATA: {experience_data}
- EDUCATION_DATA: {education_data}
- PARSED_RESUME: {escaped_parsed_resume}
- GITHUB_INFO: {escaped_github_info}

**Target Position:**
- JOB_TITLE: {job_title}
- COMPANY_NAME: {company_name}
- JOB_REQUIREMENTS: {escaped_job_req}
- PREFERRED_KEYWORDS: {preferred_keywords}

## FORMATTING REQUIREMENTS
Use this exact structure:
# {candidate_name}
## Contact Information
## Professional Summary
## Core Competencies
## Professional Experience
## Education

For experience entries, use 4 bullet points per role:
- Achievement/Project with quantified results
- Technical stack and tools used
- Business impact and value delivered
- Key skills demonstrated

Keep content ATS-friendly with clear keywords and simple formatting.
Generate the markdown resume now.
"""

            # Generate resume with AI
            st.write("🤖 Generating optimized resume content...")
            output_markdown_resume = call_openrouter(prompt, api_key)
            
            if "Exception" in output_markdown_resume or "error" in output_markdown_resume.lower():
                st.error("Failed to generate resume content. Please check your API key and try again.")
                st.code(output_markdown_resume)
            else:
                # Generate PDF
                st.write("📄 Converting to PDF...")
                pdf_file = "generated_resume.pdf"
                
                result = md_to_pdf_fpdf2(output_markdown_resume, pdf_file)
                
                if result:
                    st.success("✅ Resume generated successfully!")
                    
                    # Download button
                    download_button(pdf_file, "📥 Download Resume PDF")
                    
                    # Preview
                    st.markdown("---")
                    st.subheader("📋 Markdown Preview")
                    st.code(output_markdown_resume, language='markdown')
                    
                    # Clean up
                    try:
                        os.remove(pdf_file)
                    except:
                        pass
                else:
                    st.error("Failed to generate PDF. Please try again.")
st.markdown("🔒 Your data is processed securely. No information is stored on our servers.")