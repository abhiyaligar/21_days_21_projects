import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('random_forest_wine_quality_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title("Wine Quality Prediction")

# Input features: create sliders/input fields for each feature
fixed_acidity = st.slider('Fixed Acidity', 4.0, 16.0, 7.4)
volatile_acidity = st.slider('Volatile Acidity', 0.0, 2.0, 0.7)
citric_acid = st.slider('Citric Acid', 0.0, 1.0, 0.0)
residual_sugar = st.slider('Residual Sugar', 0.0, 15.0, 1.9)
chlorides = st.slider('Chlorides', 0.0, 0.2, 0.076)
free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 0, 100, 11)
total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 0, 300, 34)
density = st.slider('Density', 0.9900, 1.0050, 0.9978)
pH = st.slider('pH', 2.5, 4.5, 3.51)
sulphates = st.slider('Sulphates', 0.0, 2.0, 0.56)
alcohol = st.slider('Alcohol', 8.0, 15.0, 9.4)

# Collect input into a numpy array and reshape for prediction
features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                      chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                      density, pH, sulphates, alcohol]])

# Scale features
features_scaled = scaler.transform(features)

# Predict button
if st.button('Predict Quality'):
    prediction = model.predict(features_scaled)
    st.success(f'Predicted Wine Quality Score: {prediction[0]:.2f}')
