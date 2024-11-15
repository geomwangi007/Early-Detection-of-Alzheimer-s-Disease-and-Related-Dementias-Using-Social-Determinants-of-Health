import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page title and description
st.title('DementiaShield – Predicting Alzheimer’s with Social Insights')
st.markdown("""
* The AlzAware project aims to develop a predictive model for early identification of Alzheimer’s Disease (AD) and Alzheimer’s Disease-Related Dementias (AD/ADRD) based on social determinants of health. Utilizing data from the Mexican Health and Aging Study (MHAS), this project focuses on uncovering associations between social, economic, and environmental factors and cognitive decline risk. The ultimate goal is to enable early intervention and improved access to care, especially for underrepresented populations.
""")

# Define ordinal mappings
EDUCATION_LEVELS = {
    'No Education': 0,
    'Some Primary': 1,
    'Completed Primary': 2,
    'Some Secondary': 3,
    'Completed Secondary': 4,
    'Some Higher Education': 5,
    'Completed Higher Education': 6
}

def user_input_features():
    st.sidebar.header('Education and Background')
    
    # Education level (dropdown)
    edu_gru_12 = st.sidebar.selectbox(
        'Education Level (edu_gru_12)',
        options=list(EDUCATION_LEVELS.keys()),
        index=list(EDUCATION_LEVELS.keys()).index('Completed Primary'),
        help='Your highest level of education completed'
    )
     # Other social and demographic inputs
    age_03 = st.sidebar.number_input(
        'Age (age_03)',
        min_value=0,
        max_value=120,
        value=60,
        help='Age in years'
    )
    edu_gru_03 = st.sidebar.selectbox(
        "Father's Education Level (edu_gru_03)",
        options=list(EDUCATION_LEVELS.keys()),
        index=list(EDUCATION_LEVELS.keys()).index('Some Primary'),
        help="Father's highest level of education completed"
    )
    age_12 = st.sidebar.number_input(
        'Age (age_12)',
        min_value=0,
        max_value=120,
        value=65,
        help='Age in years'
    )
    
    n_living_child_12 = st.sidebar.number_input(
        'Number of Living Children (n_living_child_12)',
        min_value=0,
        max_value=15,
        value=2,
        help='Number of living children'
    )
    # Mother's education (dropdown)
    rameduc_m = st.sidebar.selectbox(
        "Mother's Education Level (rameduc_m)",
        options=list(EDUCATION_LEVELS.keys()),
        index=list(EDUCATION_LEVELS.keys()).index('Some Primary'),
        help="Mother's highest level of education completed"
    )
    
   
    adl_walk_03 = st.sidebar.selectbox(
        'ADL Walking (adl_walk_03)',
        options=['No Difficulty', 'Some Difficulty', 'A Lot of Difficulty', 'Cannot Do'],
        index=0,
        help='Difficulty in walking in the past year'
    )

    iadl_money_03 = st.sidebar.selectbox(
        'IADL Money Management (iadl_money_03)',
        options=['No Difficulty', 'Some Difficulty', 'A Lot of Difficulty', 'Cannot Do'],
        index=0,
        help='Difficulty managing money in the past year'
    )

    adl_walk_12 = st.sidebar.selectbox(
        'ADL Walking (adl_walk_12)',
        options=['No Difficulty', 'Some Difficulty', 'A Lot of Difficulty', 'Cannot Do'],
        index=0,
        help='Difficulty in walking in the past year'
    )

    iadl_money_12 = st.sidebar.selectbox(
        'IADL Money Management (iadl_money_12)',
        options=['No Difficulty', 'Some Difficulty', 'A Lot of Difficulty', 'Cannot Do'],
        index=0,
        help='Difficulty managing money in the past year'
    )

    # Create the data dictionary with proper ordinal encoding
    data = {
        'edu_gru_12': EDUCATION_LEVELS[edu_gru_12],
        'age_03': age_03,
        'edu_gru_03': EDUCATION_LEVELS[edu_gru_03],
        'age_12': age_12,
        'n_living_child_12': n_living_child_12,
        'rameduc_m': EDUCATION_LEVELS[rameduc_m],
        'adl_walk_03': adl_walk_03,
        'iadl_money_03': iadl_money_03,
        'adl_walk_12': adl_walk_12,
        'iadl_money_12': iadl_money_12
    }

    return pd.DataFrame(data, index=[0])

# Get user input
df = user_input_features()

# Show user input
st.subheader('User Input Features')
st.write("Input Values:", df)

# Load the saved model
@st.cache_resource
def load_model():
    try:
        with open('Models/final_ridge_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'final_ridge_model.pkl' is present.")
        return None

model = load_model()

if model is not None:
    # Make prediction
    prediction = model.predict(df)
    
    # Display prediction
    st.subheader('Prediction')
    st.write(f'Predicted Value: {prediction[0]:.2f}')
    
    # Add performance visualization section
    st.subheader('Model Performance Visualization')
    test_file = st.file_uploader("Upload test data (CSV) to visualize model performance", type=['csv'])
    
    if test_file is not None:
        try:
            # Read test data
            test_data = pd.read_csv(test_file)
            
            # Check if required columns exist
            required_features = df.columns
            target_col = 'actual_values'  # Replace with your actual target column name
            
            if all(feat in test_data.columns for feat in required_features) and target_col in test_data.columns:
                # Make predictions on test data
                X_test = test_data[required_features]
                y_test = test_data[target_col]
                y_pred = model.predict(X_test)
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                
                # Display RMSE
                st.metric("Root Mean Square Error (RMSE)", f"{rmse:.4f}")
                
                # Optional: Display actual vs predicted plot
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', size=8)
                ))
                
                # Add perfect prediction line
                min_val = min(min(y_test), min(y_pred))
                max_val = max(max(y_test), max(y_pred))
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title='Actual vs Predicted Values',
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values',
                    showlegend=True
                )
                
                st.plotly_chart(fig)
                
            else:
                st.error("Test data must contain all required features and actual values column")
                
        except Exception as e:
            st.error(f"Error processing test data: {str(e)}")
            
    else:
        st.info("""
        To visualize model performance, upload a CSV file containing:
        - All the input features used in the model
        - A column named 'actual_values' containing the true values
        """)
