import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

def descriptive_stats(df: pd.DataFrame):
    stats = {}

    # Numeric columns statistics
    num_cols = df.select_dtypes(include=["number"]).columns
    num_stats = df[num_cols].describe().transpose()
    num_stats['median'] = df[num_cols].median()
    num_stats['mode'] = df[num_cols].mode().iloc[0]
    stats['numeric'] = num_stats

    # Categorical columns statistics
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cat_stats = {}
    for col in cat_cols:
        cat_stats[col] = {
            "unique_values": df[col].nunique(),
            "top": df[col].mode()[0] if not df[col].mode().empty else None,
            "freq": df[col].value_counts().iloc[0] if not df[col].value_counts().empty else None
        }
    stats['categorical'] = cat_stats

    return stats


def missing_data_analysis(df):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent
    })
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(by="missing_percent", ascending=False)
    return missing_df


def plot_histogram(df, column):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f'Histogram of {column}')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def plot_boxplot(df, column):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[column].dropna())
    plt.title(f'Boxplot of {column}')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def plot_barplot(df, column):
    plt.figure(figsize=(6, 4))
    sns.countplot(y=column, data=df, order=df[column].value_counts().index)
    plt.title(f'Barplot of {column}')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def get_image_base64(buf):
    return base64.b64encode(buf.read()).decode()

def plot_correlation_heatmap(df, selected_columns):
    if len(selected_columns) < 2:
        return None  # Need at least 2 columns for correlation

    corr = df[selected_columns].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def extract_insights(df):
    insights = []

    # Missing data insights
    missing_df = df.isnull().mean() * 100
    high_missing = missing_df[missing_df > 30]
    if not high_missing.empty:
        for col, pct in high_missing.items():
            insights.append(f"Column '{col}' has high missing rate: {pct:.2f}%")

    # Outlier detection using IQR for numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        if not outliers.empty:
            insights.append(f"Column '{col}' has potential outliers detected by IQR method")

    # Skewness check
    for col in num_cols:
        skewness = df[col].skew()
        if skewness > 1 or skewness < -1:
            insights.append(f"Column '{col}' is highly skewed (skewness = {skewness:.2f})")

    # Correlation check - flag pairs with high correlation
    corr = df[num_cols].corr()
    strong_corrs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.75:
                strong_corrs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    if strong_corrs:
        for c1, c2, val in strong_corrs:
            insights.append(f"Columns '{c1}' and '{c2}' are highly correlated (corr = {val:.2f})")

    if not insights:
        insights.append("No critical insights or warnings detected.")

    return insights

def generate_html_report(df):
    """Generate basic HTML report with descriptive stats and missing data summaries."""
    numeric_desc = df.describe().to_html()
    missing_data = df.isnull().sum().to_frame(name='missing_count').to_html()

    report_html = f"""
    <html>
        <head><title>EDA Report</title></head>
        <body>
            <h1>Exploratory Data Analysis Report</h1>
            <h2>Descriptive Statistics (Numeric Columns)</h2>
            {numeric_desc}
            <h2>Missing Data Summary</h2>
            {missing_data}
        </body>
    </html>
    """
    return report_html

def standardize_columns(df, columns):
    scaler = StandardScaler()
    df_std = df.copy()
    df_std[columns] = scaler.fit_transform(df_std[columns])
    return df_std

def log_transform_columns(df, columns):
    df_log = df.copy()
    for col in columns:
        # Add small constant to avoid log(0)
        df_log[col] = np.log1p(df_log[col].clip(lower=0))
    return df_log

def create_new_feature(df, col1, col2, operation, new_col_name):
    df_new = df.copy()
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Invalid columns selected.")
    
    if operation == 'Add':
        df_new[new_col_name] = df_new[col1] + df_new[col2]
    elif operation == 'Subtract':
        df_new[new_col_name] = df_new[col1] - df_new[col2]
    elif operation == 'Multiply':
        df_new[new_col_name] = df_new[col1] * df_new[col2]
    elif operation == 'Divide':
        # Avoid division by zero
        df_new[new_col_name] = df_new[col1] / df_new[col2].replace(0, pd.NA)
    else:
        raise ValueError("Unsupported operation")
    return df_new
