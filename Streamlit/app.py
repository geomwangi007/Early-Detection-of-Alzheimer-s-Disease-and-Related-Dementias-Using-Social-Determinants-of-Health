import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

header_container = st.container()
processing_container = st.container()

with header_container:
   
    # Different levels of text you can include in your app
    st.title("DementiaShield – Predicting Alzheimer’s with Social Insights")
    st.header("Welcome!")
    st.subheader("This is a great app")
    st.write("The AlzAware project aims to develop a predictive model for early identification of Alzheimer’s Disease (AD) and Alzheimer’s Disease-Related Dementias (AD/ADRD) based on social determinants of health. Utilizing data from the Mexican Health and Aging Study (MHAS), this project focuses on uncovering associations between social, economic, and environmental factors and cognitive decline risk. The ultimate goal is to enable early intervention and improved access to care, especially for underrepresented populations.")

file = st.file_uploader('Upload a file (.txt)')

# Another container
with processing_container:
    # Import dataset
    train_data = pd.read_csv('Original data/train_features.csv')
    train_labels = pd.read_csv('Original data/train_labels.csv')
