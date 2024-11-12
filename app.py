import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the best Ridge regression model
with open('best_ridge_model.pkl', 'rb') as f:
    best_ridge_model = pickle.load(f)

# Extract the preprocessor and Ridge model from the saved pipeline
preprocessor = best_ridge_model['preprocessor']
ridge = best_ridge_model['ridge']

# Create the Streamlit app
st.title("Ridge Regression Prediction")

# Create input fields for user to enter data
edu_gru_12 = st.selectbox("Select your education level (12):", ['0. No education', '1. 1–5 years', '2. 6 years', '3. 7–9 years', '4. 10+ years'])
j11_12 = st.selectbox("Select the material of your floor (12):", ['Wood, mosaic, or other covering 1', 'Concrete 2', 'Mud 3'])
rameduc_m = st.selectbox("Select your mother's education level:", ['1.None', '2.Some primary', '3.Primary', '4.More than primary'])
age_12 = st.selectbox("Select your age (12):", ['0. 49 or younger', '1. 50–59', '2. 60–69', '3. 70–79', '4. 80+'])
economic_strain_index = st.number_input("Enter your economic strain index:", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
table_games_12 = st.number_input("Enter your table games score (12):", min_value=0, max_value=100, value=50, step=1)
reads_12 = st.number_input("Enter your reads score (12):", min_value=0, max_value=100, value=50, step=1)
rrfcntx_m_12 = st.selectbox("Select your religious attendance frequency (12):", ['1.Almost every day', '2.4 or more times a week', '3.2 or 3 times a week', '4.Once a week', '5.4 or more times a month', '6.2 or 3 times a month', '7.Once a month', '8.Almost Never, sporadic', '9.Never'])
healthy_lifestyle_score = st.number_input("Enter your healthy lifestyle score:", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
n_living_child_12 = st.selectbox("Select the number of your living children (12):", ['0. No children', '1. 1 or 2', '2. 3 or 4', '3. 5 or 6', '4. 7+'])
rearnings_12 = st.number_input("Enter your earnings (12):", min_value=0, max_value=1000000, value=50000, step=1000)
hincome_03 = st.number_input("Enter your household income (03):", min_value=0, max_value=1000000, value=50000, step=1000)
rsocact_m_12 = st.selectbox("Select your social activity frequency (12):", ['1.Almost every day', '2.4 or more times a week', '3.2 or 3 times a week', '4.Once a week', '5.2 or 3 times a month', '6.Once a month', '7.4 or more times a month', '8.Almost Never, sporadic', '9.Never'])
care_child_12 = st.number_input("Enter your child care score (12):", min_value=0, max_value=100, value=50, step=1)
age_03 = st.selectbox("Select your age (03):", ['0. 49 or younger', '1. 50–59', '2. 60–69', '3. 70–79', '4. 80+'])

# Make a prediction when the user clicks a button
if st.button("Predict"):
    # Create a DataFrame from user input
    user_data = pd.DataFrame({
        'edu_gru_12': [edu_gru_12],
        'j11_12': [j11_12],
        'rameduc_m': [rameduc_m],
        'age_12': [age_12],
        'economic_strain_index': [economic_strain_index],
        'table_games_12': [table_games_12],
        'reads_12': [reads_12],
        'rrfcntx_m_12': [rrfcntx_m_12],
        'healthy_lifestyle_score': [healthy_lifestyle_score],
        'n_living_child_12': [n_living_child_12],
        'rearnings_12': [rearnings_12],
        'hincome_03': [hincome_03],
        'rsocact_m_12': [rsocact_m_12],
        'care_child_12': [care_child_12],
        'age_03': [age_03]
    })

    # Make a prediction using the loaded model
    prediction = best_ridge_model.predict(user_data)

    # Display the prediction
    st.write(f"The predicted value is: {prediction[0]}")