import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Outlier Detection Web App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # Select numeric columns for outlier detection
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns for visualization.")
    else:
        selected_cols = st.multiselect(
            "Select at least two numeric columns for outlier detection and visualization",
            numeric_cols, default=numeric_cols[:2]
        )

        if len(selected_cols) >= 2:
            # Isolation Forest model
            clf = IsolationForest(contamination=0.05, random_state=42)
            preds = clf.fit_predict(data[selected_cols])
            data["Outlier"] = preds
            data["Outlier"] = data["Outlier"].map({1: "Inlier", -1: "Outlier"})

            st.write("Outlier Detection Results:")
            st.write(data)

            # Plotting outliers on scatterplot of first two selected columns
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=data[selected_cols[0]], 
                y=data[selected_cols[1]], 
                hue=data["Outlier"],
                palette={"Inlier": "blue", "Outlier": "red"},
                ax=ax
            )
            ax.set_title("Outlier Detection by Isolation Forest")
            st.pyplot(fig)

            # Download option for results
            csv_exp = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results CSV",
                data=csv_exp,
                file_name='outlier_detection_results.csv',
                mime='text/csv'
            )
        else:
            st.warning("Please select at least two numeric columns.")
else:
    st.info("Please upload a CSV file to begin.")
