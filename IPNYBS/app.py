import streamlit as st
import pandas as pd
import numpy as np
import joblib
from custom_transformers import TemporalFeatureEngineer, EducationProgressionTransformer, MaritalTransitionTransformer, ChronicIllnessTransformer, ADLIADLTransformer, HealthAssessmentChangeTransformer, MoodScoreTransformer, ConsistentExerciseTransformer, LifestyleHealthIndexTransformer, SocioeconomicFeaturesTransformer, SocialEngagementTransformer, HealthServicesTransformer, CustomFeatureEngineer
from sklearn.pipeline import Pipeline


def map_ordinal_variables(X, ordinal_cols, ordinal_mappings):
    """Function to map ordinal variables to numerical values."""
    X = X.copy()
    for col in ordinal_cols:
        if col in X.columns:
            mapping = ordinal_mappings.get(col, {})
            X[col] = X[col].map(mapping)
    return X

# Load the model
model = joblib.load('stacked_model.pkl')

# Define your preprocess_data function
def preprocess_data(input_df, expected_features):
    # Assuming that 'uid' is not necessary for predictions, drop it if exists
    features_to_drop = ['uid']
    X_test = input_df.drop(columns=features_to_drop, errors='ignore')

    # Reindex X_test to match the expected features used during training
    X_test = X_test.reindex(columns=expected_features)

    # Handle missing values by filling with 0 as a placeholder (adjust as needed)
    X_test = X_test.fillna(0)
    
    return X_test

# Streamlit app layout
st.title("Cognitive Decline Prediction App")
st.write("""
Upload a CSV file containing the features required for prediction. The app will output composite cognitive scores for each `uid` and `year`.
""")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load and display uploaded file
    input_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(input_data.head())

    # Preprocess data with the preprocessing pipeline
    expected_features = ['year', 'age_03', 'urban_03', 'married_03', 'n_mar_03', 'edu_gru_03', 'n_living_child_03', 'migration_03', 'glob_hlth_03', 'adl_dress_03', 'adl_walk_03', 'adl_bath_03', 'adl_eat_03', 'adl_bed_03', 'adl_toilet_03', 'n_adl_03', 'iadl_money_03', 'iadl_meds_03', 'iadl_shop_03', 'iadl_meals_03', 'n_iadl_03', 'depressed_03', 'hard_03', 'restless_03', 'happy_03', 'lonely_03', 'enjoy_03', 'sad_03', 'tired_03', 'energetic_03', 'n_depr_03', 'cesd_depressed_03', 'hypertension_03', 'diabetes_03', 'resp_ill_03', 'arthritis_03', 'hrt_attack_03', 'stroke_03', 'cancer_03', 'n_illnesses_03', 'exer_3xwk_03', 'alcohol_03', 'tobacco_03', 'test_chol_03', 'test_tuber_03', 'test_diab_03', 'test_pres_03', 'hosp_03', 'visit_med_03', 'out_proc_03', 'visit_dental_03', 'imss_03', 'issste_03', 'pem_def_mar_03', 'insur_private_03', 'insur_other_03', 'insured_03', 'decis_personal_03', 'employment_03', 'age_12', 'urban_12', 'married_12', 'n_mar_12', 'edu_gru_12', 'n_living_child_12', 'migration_12', 'glob_hlth_12', 'adl_dress_12', 'adl_walk_12', 'adl_bath_12', 'adl_eat_12', 'adl_bed_12', 'adl_toilet_12', 'n_adl_12', 'iadl_money_12', 'iadl_meds_12', 'iadl_shop_12', 'iadl_meals_12', 'n_iadl_12', 'depressed_12', 'hard_12', 'restless_12', 'happy_12', 'lonely_12', 'enjoy_12', 'sad_12', 'tired_12', 'energetic_12', 'n_depr_12', 'cesd_depressed_12', 'hypertension_12', 'diabetes_12', 'resp_ill_12', 'arthritis_12', 'hrt_attack_12', 'stroke_12', 'cancer_12', 'n_illnesses_12', 'bmi_12', 'exer_3xwk_12', 'alcohol_12', 'tobacco_12', 'test_chol_12', 'test_tuber_12', 'test_diab_12', 'test_pres_12', 'hosp_12', 'visit_med_12', 'out_proc_12', 'visit_dental_12', 'imss_12', 'issste_12', 'pem_def_mar_12', 'insur_private_12', 'insur_other_12', 'insured_12', 'decis_famil_12', 'decis_personal_12', 'employment_12', 'vax_flu_12', 'vax_pneu_12', 'seg_pop_12', 'care_adult_12', 'care_child_12', 'volunteer_12', 'attends_class_12', 'attends_club_12', 'reads_12', 'games_12', 'table_games_12', 'comms_tel_comp_12', 'act_mant_12', 'tv_12', 'sewing_12', 'satis_ideal_12', 'satis_excel_12', 'satis_fine_12', 'cosas_imp_12', 'wouldnt_change_12', 'memory_12', 'ragender', 'rameduc_m', 'rafeduc_m', 'sgender_03', 'rearnings_03', 'searnings_03', 'hincome_03', 'hinc_business_03', 'hinc_rent_03', 'hinc_assets_03', 'hinc_cap_03', 'rinc_pension_03', 'sinc_pension_03', 'rrelgimp_03', 'sgender_12', 'rjlocc_m_12', 'rearnings_12', 'searnings_12', 'hincome_12', 'hinc_business_12', 'hinc_rent_12', 'hinc_assets_12', 'hinc_cap_12', 'rinc_pension_12', 'sinc_pension_12', 'rrelgimp_12', 'rrfcntx_m_12', 'rsocact_m_12', 'rrelgwk_12', 'a34_12', 'j11_12']
    preprocessed_data = preprocess_data(input_data, expected_features)

    # Run predictions
    predictions = model.predict(preprocessed_data)

    # Round predictions to nearest integer and convert to integer type
    y_pred_int = np.round(predictions).astype(int)

    # Prepare results for download
    result_df = input_data[['uid', 'year']].copy()
    result_df['composite_score'] = y_pred_int

    st.write("Prediction Results:")
    st.write(result_df)

    # Download button for results
    st.download_button(
        label="Download Predictions as CSV",
        data=result_df.to_csv(index=False),
        file_name="predicted_cognitive_scores.csv",
        mime="text/csv"
    )
