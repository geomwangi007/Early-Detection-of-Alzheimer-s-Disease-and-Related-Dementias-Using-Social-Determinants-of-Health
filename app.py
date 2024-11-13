import streamlit as st
import pandas as pd
import pickle
from modules.data_preprocessing import load_and_merge_data, apply_mappings, transform_data, preprocess_data
from modules.training import create_pipeline, evaluate_model

# Load trained model
model_path = 'models/best_ridge_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# User input via file upload
uploaded_file = st.file_uploader("Upload your data file (CSV)")

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)

    # Preprocess user input
    preprocessor = preprocess_data(user_data)
    pipeline = create_pipeline(preprocessor)

    # Transform data and make predictions
    user_input_transformed = preprocessor.transform(user_data)
    predictions = model.predict(user_input_transformed)

    st.write("Predictions:", predictions)
