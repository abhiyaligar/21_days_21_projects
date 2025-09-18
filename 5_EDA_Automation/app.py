import streamlit as st
import pandas as pd
import base64
from utils.eda_util import (
    descriptive_stats, 
    missing_data_analysis, 
    plot_histogram, 
    plot_boxplot, 
    plot_barplot, 
    plot_correlation_heatmap, 
    extract_insights,
    generate_html_report,
    standardize_columns,
    log_transform_columns,
    create_new_feature
)

st.title("Automated EDA with Transformation & Feature Creation")

uploaded_file = st.file_uploader("Upload your CSV file (max 10MB)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head(10))
        st.write(f"Data shape: {df.shape[0]} rows and {df.shape[1]} columns")

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Transformation selection
        transformation = st.selectbox("Select Transformation", ["None", "Standardize", "Logarithm"])
        selected_transform_cols = st.multiselect("Select Numeric Columns for Transformation", num_cols, default=[])

        # Apply transformations
        if transformation == "Standardize" and selected_transform_cols:
            df_transformed = standardize_columns(df, selected_transform_cols)
        elif transformation == "Logarithm" and selected_transform_cols:
            df_transformed = log_transform_columns(df, selected_transform_cols)
        else:
            df_transformed = df.copy()

        # Custom Feature Creation UI
        st.write("### Custom Feature Creation")

        if len(num_cols) >= 2:
            new_feature_col1 = st.selectbox("Select First Column", num_cols)
            new_feature_col2 = st.selectbox("Select Second Column", [col for col in num_cols if col != new_feature_col1])
            operation = st.selectbox("Select Operation", ['Add', 'Subtract', 'Multiply', 'Divide'])
            new_feature_name = st.text_input("New Feature Column Name", value=f"{new_feature_col1}_{operation}_{new_feature_col2}")

            if st.button("Create New Feature"):
                if new_feature_name in df_transformed.columns:
                    st.warning("Column name already exists. Choose a different name.")
                else:
                    try:
                        df_transformed = create_new_feature(df_transformed, new_feature_col1, new_feature_col2, operation, new_feature_name)
                        st.success(f"New feature '{new_feature_name}' created successfully!")
                        # Update num_cols to include new feature
                        num_cols.append(new_feature_name)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Need at least 2 numeric columns to create new feature.")

        st.write("### Descriptive Statistics")
        stats = descriptive_stats(df_transformed)

        st.write("#### Numeric Columns")
        st.dataframe(stats['numeric'])
        st.write("#### Categorical Columns")
        for col, desc in stats['categorical'].items():
            st.write(f"{col}: Unique = {desc['unique_values']}, Top = {desc['top']}, Frequency = {desc['freq']}")

        st.write("### Missing Data Analysis")
        missing_df = missing_data_analysis(df_transformed)
        if not missing_df.empty:
            st.dataframe(missing_df)
        else:
            st.write("No missing data found!")

        # Column selection for visualization
        selected_num_cols = st.multiselect("Select Numeric Columns for Visualization", num_cols, default=num_cols[:3])
        selected_cat_cols = st.multiselect("Select Categorical Columns for Visualization", cat_cols, default=cat_cols[:3])

        st.write("### Distribution Visualizations")
        st.write("#### Numeric Columns")
        for col in selected_num_cols:
            st.write(f"**{col}**")
            hist = plot_histogram(df_transformed, col)
            st.image(hist, use_container_width=True)
            box = plot_boxplot(df_transformed, col)
            st.image(box, use_container_width=True)

        st.write("#### Categorical Columns")
        for col in selected_cat_cols:
            st.write(f"**{col}**")
            bar = plot_barplot(df_transformed, col)
            st.image(bar, use_container_width=True)

        st.write("### Correlation Heatmap")
        correlation_columns = st.multiselect("Select Numeric Columns for Correlation Heatmap", num_cols, default=num_cols[:5])
        if len(correlation_columns) < 2:
            st.info("Please select at least 2 numeric columns to plot correlation heatmap.")
        else:
            heatmap = plot_correlation_heatmap(df_transformed, correlation_columns)
            if heatmap:
                st.image(heatmap, use_container_width=True)

        st.write("### Automated Insights")
        insights = extract_insights(df_transformed)
        for insight in insights:
            st.write(f"- {insight}")

        st.write("### Export Report")
        if st.button("Generate & Download HTML Report"):
            html_report = generate_html_report(df_transformed)
            b64 = base64.b64encode(html_report.encode()).decode()
            href = f'<a href="data:file/html;base64,{b64}" download="eda_report.html">Download EDA Report</a>'
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Upload a CSV file to get started")
