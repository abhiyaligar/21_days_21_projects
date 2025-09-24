import streamlit as st
from newspaper import Article
from transformers import pipeline

# Initialize the summarization pipeline (using T5 or BART)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-base")

summarizer = load_summarizer()

# UI: Enter article URL
st.title("Online Article Summarizer")
url = st.text_input("Paste the Article URL:")

if st.button("Summarize"):
    if not url:
        st.warning("Please enter an article URL.")
    else:
        try:
            # Download and parse the article
            article = Article(url)
            article.download()
            article.parse()
            st.info("Article successfully fetched.")

            # Run summarization on the article content
            text = article.text
            if len(text) < 80:
                st.warning("Extracted text is very short; try a different article.")
            else:
                summary = summarizer(text, max_length=150, min_length=60, do_sample=False)
                st.subheader("Summary")
                st.success(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"Failed to summarize the article: {e}")
