# Wine Quality Prediction Web Application

## Project Overview
This project builds and deploys a machine learning model to predict the quality of wine based on 11 physicochemical properties. Using a Random Forest regressor trained on the Wine Quality dataset from Kaggle, the app allows users to input key wine characteristics and receive an estimated quality score.

## Dataset
- **Source:** [Wine Quality Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)  
- **Features (11 physicochemical properties):**  
  - Fixed Acidity (range: 4.0 - 16.0)  
  - Volatile Acidity (range: 0.0 - 2.0)  
  - Citric Acid (range: 0.0 - 1.0)  
  - Residual Sugar (range: 0.0 - 15.0)  
  - Chlorides (range: 0.0 - 0.2)  
  - Free Sulfur Dioxide (range: 0 - 100)  
  - Total Sulfur Dioxide (range: 0 - 300)  
  - Density (range: 0.9900 - 1.0050)  
  - pH (range: 2.5 - 4.5)  
  - Sulphates (range: 0.0 - 2.0)  
  - Alcohol (range: 8.0 - 15.0)  
- **Target:** Wine quality score (continuous value)

## Methodology
1. **Data Preprocessing**  
   - Removed irrelevant columns such as "Id" (if present).  
   - Split data into features (X) and target variable (y).  
   - Scaled features using StandardScaler for consistent input range.

2. **Model Training**  
   - Used Random Forest Regressor trained on 80% of the dataset.  
   - Evaluated performance on the remaining 20% using MAE, RMSE, and R² metrics.

3. **Model Saving**  
   - Serialized the trained model and scaler using joblib for reuse.

4. **Web Application**  
   - Built a user-friendly interface using Streamlit.  
   - Allows users to input 11 physicochemical properties via sliders.  
   - Preprocesses inputs, applies the trained model, and outputs predicted wine quality.

## How to Use
1. Clone/download the repository.  
2. Place the saved files `random_forest_wine_quality_model.pkl` and `scaler.pkl` in the project directory.  
3. Install dependencies:
```
pip install streamlit scikit-learn joblib numpy pandas
```
4. Run the Streamlit app:
```
streamlit run wine_quality_app.py
```
5. Use the sliders to input wine features and click **Predict Quality** to see the estimated wine score.


## Acknowledgments
- Wine Quality dataset from Kaggle (original UCI Machine Learning Repository).  
- Libraries: pandas, scikit-learn, Streamlit, joblib, numpy.
