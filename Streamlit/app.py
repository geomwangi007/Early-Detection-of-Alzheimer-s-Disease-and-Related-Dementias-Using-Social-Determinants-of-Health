import streamlit as st
import pandas as pd
from model.model_training import train_model
from utils.data_handling import load_data
import pickle

st.markdown(
    """
    <style>
        .reportview-container .main .block-container{
            max-width: 90%;
            padding-top: 5rem;
            padding-right: 5rem;
            padding-left: 5rem;
            padding-bottom: 5rem;
        }
        img{
            max-width:40%;
            margin-bottom:40px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("DementiaShield – Predicting Alzheimer’s with Social Insights")

# File upload
uploaded_file = st.file_uploader("Upload a .csv file")
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)

    # Load and prepare training data (used for initial model training)
    data = load_data('data/train_features.csv', 'data/train_labels.csv')
    X = data.drop(['composite_score', 'year'], axis=1)
    y = data['composite_score']

    # Train model and evaluate
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    model, metrics = train_model(X, y, num_cols, cat_cols)
    
    st.write("Model Evaluation Metrics:")
    st.json(metrics)
