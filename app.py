import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.title('DementiaShield – Predicting Alzheimer’s with Social Insights')
st.markdown('* The AlzAware project aims to develop a predictive model for early identification of Alzheimer’s Disease (AD) and Alzheimer’s Disease-Related Dementias (AD/ADRD) based on social determinants of health. Utilizing data from the Mexican Health and Aging Study (MHAS), this project focuses on uncovering associations between social, economic, and environmental factors and cognitive decline risk. The ultimate goal is to enable early intervention and improved access to care, especially for underrepresented populations.')

file = st.file_uploader('Upload a file(.txt)')