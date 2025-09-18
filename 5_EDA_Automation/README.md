# Exploratory Data Analysis Automation with Feature Transformation and Creation

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen)](https://edaauto.streamlit.app/)

## Project Overview
This project automates and enhances **Exploratory Data Analysis (EDA)** by providing an interactive Streamlit web application.  
It enables users to upload datasets and perform comprehensive data profiling, including:
- Descriptive statistics
- Missing data analysis
- Distribution visualizations  

In addition, it supports **feature transformations** (standardization, logarithm) and **custom feature creation** (via mathematical operations).  
The app dynamically generates correlation heatmaps and automated insights to help users quickly understand their data.



## Motivation
Data exploration is one of the most **critical yet time-consuming** steps in the data science pipeline.  
This tool was created to **speed up EDA workflows** by:
- Automating repetitive tasks
- Providing flexibility to experiment with feature engineering
- Enabling intuitive, interactive exploration  

The goal is to help **data scientists and analysts make faster, data-driven decisions**.



## Key Features
- Upload CSV datasets for instant analysis  
- Comprehensive descriptive statistics (numeric + categorical)  
- Missing data analysis with detailed reports  
- Interactive plots: histograms, boxplots, bar charts  
- Data transformations: standardization & log transformations  
- Custom feature creation (add, subtract, multiply, divide existing columns)  
- Correlation heatmap generation with selectable feature subsets  
- Automated insights highlighting anomalies and data quality issues  
- Exportable **HTML reports** summarizing the analysis  



## Technologies Used
- **Python** – Core programming language  
- **Streamlit** – Web app framework  
- **Pandas** – Data manipulation & preprocessing  
- **Seaborn & Matplotlib** – Data visualization  
- **Scikit-learn** – Feature standardization  
- **NumPy** – Numerical computations  



## Getting Started

### Installation
Clone the repository:
```
git clone https://github.com/abhiyaligar/21_days_21_projects.git/5_EDA_Automation
cd 5_EDA_Automation
```

Create and activate your virtual environment:

```
python -m venv env
# On macOS/Linux:
source env/bin/activate
# On Windows:
env\Scripts\activate
```
Install dependencies:

```
pip install -r requirements.txt
```
Running the App
```
streamlit run app.py
```
Open your browser at: http://localhost:8501


## Usage
- Upload your CSV dataset  
- Apply feature transformations (optional)  
- Create new features by combining columns  
- Choose columns and generate interactive plots  
- Explore the correlation heatmap  
- Review automated insights  
- Export your EDA report as HTML  



## Future Work
- Add more transformation methods & custom formulas  
- Support async processing for large datasets  
- Integrate ML model prototyping with EDA  
- Better error handling & user feedback  
- Add unit tests + contributor guide  


## Contributing
Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo, open issues, and submit PRs.


