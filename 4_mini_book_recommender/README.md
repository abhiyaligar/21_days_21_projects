# Mini Book Recommendation System

## Overview
This project is a simple Book Recommendation System built using Python.  
It demonstrates how to use similarity metrics on book data to recommend books to users based on their preferences.  
The system takes a given book title as input and suggests similar books by analyzing features such as book titles, authors, and genres.

## Features
- Load and preprocess book dataset
- Vectorize book features for similarity comparison
- Recommend top N similar books based on cosine similarity
- Easy to extend with additional metadata or datasets

## Tech Stack
- **Python 3.8+**
- **Jupyter Notebook**

### Libraries:
- pandas
- numpy
- scikit-learn

## How It Works
1. Dataset is cleaned and preprocessed.
2. Feature extraction is applied using TF-IDF / Count Vectorizer.
3. Cosine similarity is calculated between books.
4. Top recommendations are returned for the given input book.

## Installation
Clone this repository and install dependencies:

```
git clone https://github.com/21_days_21_projects/tree/main/4_mini_book_recommender
cd 4_mini_book_recommender
```


## Usage
- Open the Jupyter Notebook:
```
Mini_Book_Recommendation_system.ipynb
```
## Acknowledgments
- Dataset : https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data