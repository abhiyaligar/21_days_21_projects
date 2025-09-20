# Outlier Detection Web App

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen)](https://abhiyaligar-21-days-21-pr-6-outlier-detection-web-appapp-s9pbby.streamlit.app/)

A simple and interactive web application built with Streamlit for detecting outliers in your datasets using machine learning.

## Overview

This web application uses the Isolation Forest algorithm to automatically detect anomalies and outliers in your CSV data. It provides an intuitive interface for data upload, column selection, visualization, and result export.

## Features

- **Easy CSV Upload**: Simple drag-and-drop file upload interface
- **Data Preview**: View your dataset before processing
- **Interactive Column Selection**: Choose which numeric columns to analyze
- **Automatic Outlier Detection**: Uses Isolation Forest algorithm with 5% contamination rate
- **Visual Results**: Scatter plot visualization showing inliers vs outliers
- **Export Results**: Download processed data with outlier labels as CSV

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Steps

1. **Clone or download the application file**
   ```bash
   git clone https://github.com/abhiyaligar/21_days_21_projects.git/6_Outlier_Detection_Web_App
   cd 6_Outlier_Detection_Web_App
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run outlier_app.py
   ```

4. **Access the app**
   - Open your web browser
   - Navigate to `http://localhost:8501`

## How to Use

### Step 1: Upload Your Data
- Click "Browse files" or drag and drop your CSV file
- Supported format: CSV files only
- The app will display a preview of your data

### Step 2: Select Columns
- The app automatically identifies numeric columns
- Use the multiselect dropdown to choose at least 2 numeric columns
- These columns will be used for outlier detection and visualization

### Step 3: View Results
- The Isolation Forest algorithm processes your selected columns
- Results are displayed in a data table with a new "Outlier" column
- Points are classified as either "Inlier" or "Outlier"

### Step 4: Analyze Visualization
- A scatter plot shows your data points
- Blue points = Normal data (Inliers)
- Red points = Detected outliers
- The plot uses the first two selected columns for X and Y axes

### Step 5: Download Results
- Click "Download Results CSV" to save the processed data
- The downloaded file includes all original data plus outlier labels

## Technical Details

### Isolation Forest Algorithm
- **Method**: Unsupervised anomaly detection
- **Contamination Rate**: 5% (assumes 5% of data points are outliers)
- **Random State**: 42 (for reproducible results)
- **Principle**: Isolates anomalies by randomly selecting features and split values

### Data Requirements
- **File Format**: CSV
- **Minimum Columns**: At least 2 numeric columns required
- **Supported Data Types**: Integer and float columns
- **File Size**: Limited by Streamlit's default upload size (200MB)

## Example Use Cases

- **Quality Control**: Identify defective products in manufacturing data
- **Financial Analysis**: Detect unusual transactions or spending patterns
- **IoT Monitoring**: Find anomalous sensor readings
- **Customer Analytics**: Identify unusual customer behavior patterns
- **Scientific Research**: Detect outliers in experimental data

## Limitations

- Only works with numeric data columns
- Requires at least 2 numeric columns for visualization
- Uses fixed contamination rate (5%)
- Visualization limited to 2D scatter plots
- No support for categorical outlier detection

## Customization Options

You can modify the code to:
- Adjust contamination rate in `IsolationForest(contamination=0.05)`
- Change color scheme in the seaborn palette
- Add more visualization types
- Implement other outlier detection algorithms
- Add data preprocessing steps

