# DistilBERT Fine-Tuning for Movie Review Sentiment Analysis

## Project Overview
This project fine-tunes the DistilBERT transformer model on the IMDb movie reviews dataset to perform sentiment analysis. DistilBERT, a smaller and faster version of BERT, is adapted for binary classification to identify positive or negative movie reviews.

The entire pipeline, implemented in Python using Hugging Face Transformers and Datasets libraries, runs on Google Colab with GPU acceleration for efficient training.

---

## Features
- Preprocessing with DistilBERT tokenizer for input text normalization
- Fine-tuning on 50,000 labeled IMDb movie reviews
- Evaluation after each epoch for accuracy monitoring
- Saving the fine-tuned model and tokenizer for deployment
- Demonstration of inference for sentiment prediction on new text

---

## Getting Started

### Dependencies
- Python 3.7+
- torch
- transformers
- datasets

1. Open Google Colab and create a new notebook.
2. Enable GPU by selecting `Runtime` -> `Change runtime type` -> `GPU`.
3. Copy the provided code blocks sequentially for:
   - Package installation
   - Dataset loading
   - Tokenization
   - Model fine-tuning
   - Saving the model
   - Inference testing

## Acknowledgments
- Hugging Face for providing the Transformers and Datasets libraries.
- The IMDb dataset contributors.
- Google Colab for free access to GPU acceleration.
