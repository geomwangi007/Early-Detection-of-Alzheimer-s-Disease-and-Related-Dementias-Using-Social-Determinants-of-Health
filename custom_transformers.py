from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Mappings for ordinal variables
age_mapping = {
    '0. 49 or younger': 0,
    '1. 50–59': 1,
    '2. 60–69': 2,
    '3. 70–79': 3,
    '4. 80+': 4,
}

education_mapping = {
    '0. No education': 0,
    '1. 1–5 years': 1,
    '2. 6 years': 2,
    '3. 7–9 years': 3,
    '4. 10+ years': 4,
}

n_living_child_mapping = {
    '0. No children': 0,
    '1. 1 or 2': 1,
    '2. 3 or 4': 2,
    '3. 5 or 6': 3,
    '4. 7+': 4,
}

glob_health_mapping = {
    '1. Excellent': 5,
    '2. Very good': 4,
    '3. Good': 3,
    '4. Fair': 2,
    '5. Poor': 1,
}

bmi_mapping = {
    '1. Underweight': 1,
    '2. Normal weight': 2,
    '3. Overweight': 3,
    '4. Obese': 4,
    '5. Morbidly obese': 5,
}

decis_famil_mapping = {
    '1. Respondent': 1,
    '2. Approximately equal weight': 2,
    '3. Spouse': 3,
}

decis_personal_mapping = {
    '1. A lot': 3,
    '2. A little': 2,
    '3. None': 1
}

agreement_mapping = {
    '1. Agrees': 3,
    '2. Neither agrees nor disagrees': 2,
    '3. Disagrees': 1,
}

memory_mapping = {
    '1. Excellent': 5,
    '2. Very good': 4,
    '3. Good': 3,
    '4. Fair': 2,
    '5. Poor': 1,
}

parent_education_mapping = {
    '1.None': 1,
    '2.Some primary': 2,
    '3.Primary': 3,
    '4.More than primary': 4,
}

religion_importance_mapping = {
    '1.very important': 3,
    '2.somewhat important': 2,
    '3.not important': 1,
}

frequency_mapping = {
    '1.Almost every day': 9,
    '2.4 or more times a week': 8,
    '3.2 or 3 times a week': 7,
    '4.Once a week': 6,
    '5.4 or more times a month': 5,
    '6.2 or 3 times a month': 4,
    '7.Once a month': 3,
    '8.Almost Never, sporadic': 2,
    '9.Never': 1,
}

religious_services_mapping = {
    '1.Yes': 1,
    '0.No': 0,
}

english_proficiency_mapping = {
    'Yes 1': 1,
    'No 2': 0,
}

# Compile all mappings into a dictionary for easy access
ordinal_mappings = {
    'age_03': age_mapping,
    'age_12': age_mapping,
    'edu_gru_03': education_mapping,
    'edu_gru_12': education_mapping,
    'n_living_child_03': n_living_child_mapping,
    'n_living_child_12': n_living_child_mapping,
    'glob_hlth_03': glob_health_mapping,
    'glob_hlth_12': glob_health_mapping,
    'bmi_12': bmi_mapping,
    'decis_famil_12': decis_famil_mapping,
    'decis_personal_12': decis_personal_mapping,
    'satis_ideal_12': agreement_mapping,
    'satis_excel_12': agreement_mapping,
    'satis_fine_12': agreement_mapping,
    'cosas_imp_12': agreement_mapping,
    'wouldnt_change_12': agreement_mapping,
    'memory_12': memory_mapping,
    'rameduc_m': parent_education_mapping,
    'rafeduc_m': parent_education_mapping,
    'rrelgimp_03': religion_importance_mapping,
    'rrelgimp_12': religion_importance_mapping,
    'rrfcntx_m_12': frequency_mapping,
    'rsocact_m_12': frequency_mapping,
    'rrelgwk_12': religious_services_mapping,
    'a34_12': english_proficiency_mapping,
}

# Function to map ordinal variables
def map_ordinal_variables(X, ordinal_cols, ordinal_mappings):
    X = X.copy()
    for col in ordinal_cols:
        if col in X.columns:
            mapping = ordinal_mappings.get(col, {})
            X[col] = X[col].map(mapping)
    return X

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer that creates temporal features such as change over time
    and time gap since last measurement.

    Parameters:
    - numerical_common_features: List of features present in both 2003 and 2012 for temporal analysis.
    - ordinal_mappings: Dictionary mapping ordinal variables to numerical values.

    Methods:
    - fit: Returns self.
    - transform: Applies temporal feature engineering to the data.
    """
    def __init__(self, numerical_common_features, ordinal_mappings):
        self.numerical_common_features = numerical_common_features
        self.ordinal_mappings = ordinal_mappings

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        X = X.copy()
        
        # Determine last feature year using vectorized operations
        X['last_feature_year'] = self.get_last_feature_year(X)

        # Calculate time gap between last measurement and target year
        if 'year' in X.columns and 'last_feature_year' in X.columns:
            X['time_gap'] = X['year'] - X['last_feature_year']
        else:
            X['time_gap'] = np.nan  # Handle missing 'year' or 'last_feature_year'
        
        # Handle ordinal variables and map them
        for feature in self.numerical_common_features:
            base_feature = feature  # e.g., 'age', 'edu_gru'
            col_03 = feature + '_03'
            col_12 = feature + '_12'
            change_col = feature + '_change'

            # Map ordinal variables if necessary
            if base_feature in self.ordinal_mappings:
                mapping = self.ordinal_mappings[base_feature]
                if col_03 in X.columns:
                    X[col_03] = X[col_03].map(mapping)
                if col_12 in X.columns:
                    X[col_12] = X[col_12].map(mapping)

            # Convert columns to numeric (if not already)
            if col_03 in X.columns:
                X[col_03] = pd.to_numeric(X[col_03], errors='coerce')
            if col_12 in X.columns:
                X[col_12] = pd.to_numeric(X[col_12], errors='coerce')

            # Calculate change: 2012 value - 2003 value
            if col_03 in X.columns and col_12 in X.columns:
                # Both columns exist
                valid_mask = X[col_03].notnull() & X[col_12].notnull()
                X.loc[valid_mask, change_col] = X.loc[valid_mask, col_12] - X.loc[valid_mask, col_03]
                X.loc[~valid_mask, change_col] = np.nan
            else:
                X[change_col] = np.nan  # Set change to NaN if both columns do not exist

        # Drop 'last_feature_year' if not needed
        # X.drop(columns=['last_feature_year'], inplace=True)
        # Depending on whether we want to keep it

        return X

    @staticmethod
    def get_last_feature_year(X):
        """
        Determine the last year when data is available for each individual using vectorized operations.
        """
        data_2012_cols = [col for col in X.columns if col.endswith('_12')]
        data_2003_cols = [col for col in X.columns if col.endswith('_03')]

        has_2012_data = X[data_2012_cols].notnull().any(axis=1)
        has_2003_data = X[data_2003_cols].notnull().any(axis=1)

        last_year = pd.Series(np.nan, index=X.index)
        last_year[has_2012_data] = 2012
        last_year[~has_2012_data & has_2003_data] = 2003

        return last_year

class EducationProgressionTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates the change in education level between 2003 and 2012.
    
    Methods:
    - fit: Returns self.
    - transform: Computes 'education_transition' as the difference between 'edu_gru_12' and 'edu_gru_03'.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X = X.copy()
        if 'edu_gru_03' in X.columns and 'edu_gru_12' in X.columns:
            # Ensure ordinal mappings have been applied
            X['edu_gru_03'] = pd.to_numeric(X['edu_gru_03'], errors='coerce')
            X['edu_gru_12'] = pd.to_numeric(X['edu_gru_12'], errors='coerce')
            
            valid_mask = X['edu_gru_03'].notnull() & X['edu_gru_12'].notnull()
            X.loc[valid_mask, 'education_transition'] = X.loc[valid_mask, 'edu_gru_12'] - X.loc[valid_mask, 'edu_gru_03']
            X.loc[~valid_mask, 'education_transition'] = np.nan
        else:
            X['education_transition'] = np.nan
        return X
class MaritalTransitionTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates the marital transition between two time points.
    
    Parameters:
    - married_cols_03: List of marital status columns at time point 2003.
    - married_cols_12: List of marital status columns at time point 2012.
    """
    def __init__(self, married_cols_03, married_cols_12):
        self.married_cols_03 = married_cols_03
        self.married_cols_12 = married_cols_12

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure columns exist
        if self.married_cols_03[0] in X.columns and self.married_cols_12[0] in X.columns:
            # Convert to numeric if necessary
            X[self.married_cols_03[0]] = pd.to_numeric(X[self.married_cols_03[0]], errors='coerce')
            X[self.married_cols_12[0]] = pd.to_numeric(X[self.married_cols_12[0]], errors='coerce')
            # Calculate marital transition
            X['marital_transition'] = X[self.married_cols_12[0]] - X[self.married_cols_03[0]]
        else:
            X['marital_transition'] = np.nan
        return X

# Count of chronic illnesses, with change over time
class ChronicIllnessTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates the total number of chronic illnesses for each individual
    in 2003 and 2012 by summing binary indicators of specific chronic illnesses.
    
    Parameters:
    - chronic_illness_cols_03: List of chronic illness columns for 2003.
    - chronic_illness_cols_12: List of chronic illness columns for 2012.
    
    Methods:
    - fit: Returns self.
    - transform: Adds 'chronic_illness_count_03' and 'chronic_illness_count_12' to the dataset.
    """
    def __init__(self, chronic_illness_cols_03, chronic_illness_cols_12):
        self.chronic_illness_cols_03 = chronic_illness_cols_03
        self.chronic_illness_cols_12 = chronic_illness_cols_12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Adjust columns based on those present in X
        illness_cols_03 = [col for col in self.chronic_illness_cols_03 if col in X.columns]
        illness_cols_12 = [col for col in self.chronic_illness_cols_12 if col in X.columns]
        
        # Convert to numeric and fill missing values
        if illness_cols_03:
            X[illness_cols_03] = X[illness_cols_03].apply(pd.to_numeric, errors='coerce').fillna(0)
            X['chronic_illness_count_03'] = X[illness_cols_03].sum(axis=1)
        else:
            X['chronic_illness_count_03'] = np.nan
            
        if illness_cols_12:
            X[illness_cols_12] = X[illness_cols_12].apply(pd.to_numeric, errors='coerce').fillna(0)
            X['chronic_illness_count_12'] = X[illness_cols_12].sum(axis=1)
        else:
            X['chronic_illness_count_12'] = np.nan

        # Calculate change over time if both counts are available
        if 'chronic_illness_count_03' in X.columns and 'chronic_illness_count_12' in X.columns:
            valid_mask = X['chronic_illness_count_03'].notnull() & X['chronic_illness_count_12'].notnull()
            X.loc[valid_mask, 'chronic_illness_count_change'] = X.loc[valid_mask, 'chronic_illness_count_12'] - X.loc[valid_mask, 'chronic_illness_count_03']
            X.loc[~valid_mask, 'chronic_illness_count_change'] = np.nan
        else:
            X['chronic_illness_count_change'] = np.nan
        return X

# Limitations of Activities of daily living count and progression
class ADLIADLTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates the total number of ADL and IADL limitations
    for each individual in 2003 and 2012, and computes the progression over time.

    Parameters:
    - adl_cols_03: List of ADL columns for 2003.
    - adl_cols_12: List of ADL columns for 2012.
    - iadl_cols_03: List of IADL columns for 2003.
    - iadl_cols_12: List of IADL columns for 2012.

    Methods:
    - fit: Returns self.
    - transform: Adds 'total_adl_limitations_03', 'total_adl_limitations_12', 'total_iadl_limitations_03', 'total_iadl_limitations_12', and 'adl_iadl_progression' to the dataset.
    """
    def __init__(self, adl_cols_03, adl_cols_12, iadl_cols_03, iadl_cols_12):
        self.adl_cols_03 = adl_cols_03
        self.adl_cols_12 = adl_cols_12
        self.iadl_cols_03 = iadl_cols_03
        self.iadl_cols_12 = iadl_cols_12

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Check if required columns are present
        adl_cols_present = all(col in X.columns for col in self.adl_cols_03 + self.adl_cols_12)
        iadl_cols_present = all(col in X.columns for col in self.iadl_cols_03 + self.iadl_cols_12)
        if adl_cols_present and iadl_cols_present:
            # Calculate total ADL limitations
            X['total_adl_limitations_03'] = X[self.adl_cols_03].sum(axis=1)
            X['total_adl_limitations_12'] = X[self.adl_cols_12].sum(axis=1)
            # Calculate total IADL limitations
            X['total_iadl_limitations_03'] = X[self.iadl_cols_03].sum(axis=1)
            X['total_iadl_limitations_12'] = X[self.iadl_cols_12].sum(axis=1)
            # Calculate progression
            X['adl_iadl_progression'] = (
                (X['total_adl_limitations_12'] + X['total_iadl_limitations_12']) -
                (X['total_adl_limitations_03'] + X['total_iadl_limitations_03'])
            )
        else:
            X['adl_iadl_progression'] = np.nan
        return X

# Self Reported Health Change
class HealthAssessmentChangeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates the change in self-reported global health status
    between 2003 and 2012.
    
    Methods:
    - fit: Returns self.
    - transform: Computes 'health_self_assessment_change' as the difference between 'glob_hlth_12' and 'glob_hlth_03'.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X = X.copy()
        if 'glob_hlth_03' in X.columns and 'glob_hlth_12' in X.columns:
            # Ensure that both columns are numeric (assuming mapping is done)
            X['glob_hlth_03'] = pd.to_numeric(X['glob_hlth_03'], errors='coerce')
            X['glob_hlth_12'] = pd.to_numeric(X['glob_hlth_12'], errors='coerce')
            
            # Identify valid entries where both health assessments are not null
            valid_mask = X['glob_hlth_03'].notnull() & X['glob_hlth_12'].notnull()
            X.loc[valid_mask, 'health_self_assessment_change'] = X.loc[valid_mask, 'glob_hlth_12'] - X.loc[valid_mask, 'glob_hlth_03']
            X.loc[~valid_mask, 'health_self_assessment_change'] = np.nan
        else:
            X['health_self_assessment_change'] = np.nan
        return X

# Custom transformer to engineer positive and negative mood scores
class MoodScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes aggregate positive and negative mood scores for each individual
    in 2003 and 2012, and calculates changes over time.

    Parameters:
    - positive_mood_cols_03: List of positive mood indicator columns for 2003.
    - positive_mood_cols_12: List of positive mood indicator columns for 2012.
    - negative_mood_cols_03: List of negative mood indicator columns for 2003.
    - negative_mood_cols_12: List of negative mood indicator columns for 2012.

    Methods:
    - fit: Returns self.
    - transform: Adds mood scores and changes to the dataset.
    """
    def __init__(self, positive_mood_cols_03, positive_mood_cols_12, negative_mood_cols_03, negative_mood_cols_12):
        self.positive_mood_cols_03 = positive_mood_cols_03
        self.positive_mood_cols_12 = positive_mood_cols_12
        self.negative_mood_cols_03 = negative_mood_cols_03
        self.negative_mood_cols_12 = negative_mood_cols_12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Adjust columns based on those present in X
        pos_mood_cols_03 = [col for col in self.positive_mood_cols_03 if col in X.columns]
        neg_mood_cols_03 = [col for col in self.negative_mood_cols_03 if col in X.columns]
        pos_mood_cols_12 = [col for col in self.positive_mood_cols_12 if col in X.columns]
        neg_mood_cols_12 = [col for col in self.negative_mood_cols_12 if col in X.columns]
        
        # Ensure columns are numeric
        for cols in [pos_mood_cols_03, neg_mood_cols_03, pos_mood_cols_12, neg_mood_cols_12]:
            if cols:
                X[cols] = X[cols].apply(pd.to_numeric, errors='coerce')
        
        # Fill missing values with 0
        for cols in [pos_mood_cols_03, neg_mood_cols_03, pos_mood_cols_12, neg_mood_cols_12]:
            if cols:
                X[cols] = X[cols].fillna(0)
        
        # Create aggregate scores for positive and negative moods in 2003
        if pos_mood_cols_03:
            X['positive_mood_score_03'] = X[pos_mood_cols_03].sum(axis=1)
        else:
            X['positive_mood_score_03'] = np.nan
        
        if neg_mood_cols_03:
            X['negative_mood_score_03'] = X[neg_mood_cols_03].sum(axis=1)
        else:
            X['negative_mood_score_03'] = np.nan
        
        # Create aggregate scores for positive and negative moods in 2012
        if pos_mood_cols_12:
            X['positive_mood_score_12'] = X[pos_mood_cols_12].sum(axis=1)
        else:
            X['positive_mood_score_12'] = np.nan
        
        if neg_mood_cols_12:
            X['negative_mood_score_12'] = X[neg_mood_cols_12].sum(axis=1)
        else:
            X['negative_mood_score_12'] = np.nan
        
        # Calculate mood changes over time
        valid_pos = X['positive_mood_score_03'].notnull() & X['positive_mood_score_12'].notnull()
        X['positive_mood_change'] = np.nan
        X.loc[valid_pos, 'positive_mood_change'] = X.loc[valid_pos, 'positive_mood_score_12'] - X.loc[valid_pos, 'positive_mood_score_03']
        
        valid_neg = X['negative_mood_score_03'].notnull() & X['negative_mood_score_12'].notnull()
        X['negative_mood_change'] = np.nan
        X.loc[valid_neg, 'negative_mood_change'] = X.loc[valid_neg, 'negative_mood_score_12'] - X.loc[valid_neg, 'negative_mood_score_03']
        
        return X

# Consistency of exercise tracking
class ConsistentExerciseTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that creates a feature indicating whether an individual consistently exercised
    three times per week in both 2003 and 2012.

    Methods:
    - fit: Returns self.
    - transform: Adds 'consistent_exercise' to the dataset.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X = X.copy()
        if 'exer_3xwk_03' in X.columns and 'exer_3xwk_12' in X.columns:
            # Convert to numeric
            X['exer_3xwk_03'] = pd.to_numeric(X['exer_3xwk_03'], errors='coerce')
            X['exer_3xwk_12'] = pd.to_numeric(X['exer_3xwk_12'], errors='coerce')
            
            # Identify valid entries
            valid_mask = X['exer_3xwk_03'].notnull() & X['exer_3xwk_12'].notnull()
            
            # Initialize the feature with NaN
            X['consistent_exercise'] = np.nan
            
            # Compute consistent_exercise
            X.loc[valid_mask, 'consistent_exercise'] = (
                (X.loc[valid_mask, 'exer_3xwk_03'] == 1) & (X.loc[valid_mask, 'exer_3xwk_12'] == 1)
            ).astype(int)
        else:
            X['consistent_exercise'] = np.nan
        return X


# Alcohol and smoking tracking
class LifestyleHealthIndexTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes a lifestyle health index for each individual in 2003 and 2012
    by summing up binary indicators for alcohol consumption and tobacco use.

    Parameters:
    - lifestyle_cols_03: List of lifestyle columns for 2003.
    - lifestyle_cols_12: List of lifestyle columns for 2012.

    Methods:
    - fit: Returns self.
    - transform: Adds 'lifestyle_health_index_03', 'lifestyle_health_index_12', and optionally 'lifestyle_health_index_change' to the dataset.
    """
    def __init__(self, lifestyle_cols_03, lifestyle_cols_12):
        self.lifestyle_cols_03 = lifestyle_cols_03
        self.lifestyle_cols_12 = lifestyle_cols_12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Adjust columns based on those present in X
        lifestyle_cols_03 = [col for col in self.lifestyle_cols_03 if col in X.columns]
        lifestyle_cols_12 = [col for col in self.lifestyle_cols_12 if col in X.columns]

        # Ensure columns are numeric and fill missing values
        for cols in [lifestyle_cols_03, lifestyle_cols_12]:
            if cols:
                X[cols] = X[cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Sum lifestyle health indices
        if lifestyle_cols_03:
            X['lifestyle_health_index_03'] = X[lifestyle_cols_03].sum(axis=1)
        else:
            X['lifestyle_health_index_03'] = np.nan

        if lifestyle_cols_12:
            X['lifestyle_health_index_12'] = X[lifestyle_cols_12].sum(axis=1)
        else:
            X['lifestyle_health_index_12'] = np.nan

        # Calculate change over time
        valid_mask = X['lifestyle_health_index_03'].notnull() & X['lifestyle_health_index_12'].notnull()
        X['lifestyle_health_index_change'] = np.nan
        X.loc[valid_mask, 'lifestyle_health_index_change'] = X.loc[valid_mask, 'lifestyle_health_index_12'] - X.loc[valid_mask, 'lifestyle_health_index_03']

        return X

# Income and insurance coverage
class SocioeconomicFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes aggregate income and insurance coverage depth for each individual
    in 2003 and 2012, and determines insurance continuity over time.

    Parameters:
    - income_cols_03: List of income columns for 2003.
    - income_cols_12: List of income columns for 2012.
    - insurance_cols_03: List of insurance columns for 2003.
    - insurance_cols_12: List of insurance columns for 2012.

    Methods:
    - fit: Returns self.
    - transform: Adds socioeconomic features to the dataset.
    """
    def __init__(self, income_cols_03, income_cols_12, insurance_cols_03, insurance_cols_12):
        self.income_cols_03 = income_cols_03
        self.income_cols_12 = income_cols_12
        self.insurance_cols_03 = insurance_cols_03
        self.insurance_cols_12 = insurance_cols_12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Adjust columns based on those present in X
        income_cols_03 = [col for col in self.income_cols_03 if col in X.columns]
        income_cols_12 = [col for col in self.income_cols_12 if col in X.columns]
        insurance_cols_03 = [col for col in self.insurance_cols_03 if col in X.columns]
        insurance_cols_12 = [col for col in self.insurance_cols_12 if col in X.columns]
        
        # Convert to numeric and fill missing values
        for cols in [income_cols_03, income_cols_12]:
            if cols:
                X[cols] = X[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        for cols in [insurance_cols_03, insurance_cols_12]:
            if cols:
                X[cols] = X[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Aggregate income
        if income_cols_03:
            X['aggregate_income_03'] = X[income_cols_03].sum(axis=1)
        else:
            X['aggregate_income_03'] = np.nan
        if income_cols_12:
            X['aggregate_income_12'] = X[income_cols_12].sum(axis=1)
        else:
            X['aggregate_income_12'] = np.nan
        
        # Insurance coverage depth
        if insurance_cols_03:
            X['insurance_coverage_depth_03'] = X[insurance_cols_03].sum(axis=1)
        else:
            X['insurance_coverage_depth_03'] = np.nan
        if insurance_cols_12:
            X['insurance_coverage_depth_12'] = X[insurance_cols_12].sum(axis=1)
        else:
            X['insurance_coverage_depth_12'] = np.nan
        
        # Insurance continuity
        valid_insurance = X['insurance_coverage_depth_03'].notnull() & X['insurance_coverage_depth_12'].notnull()
        X['insurance_continuity'] = np.nan
        X.loc[valid_insurance, 'insurance_continuity'] = ((X.loc[valid_insurance, 'insurance_coverage_depth_03'] > 0) & (X.loc[valid_insurance, 'insurance_coverage_depth_12'] > 0)).astype(int)
        
        # Optionally, calculate changes over time
        # (Include inflation adjustment if applicable)
        
        return X

# Social Engagement score
class SocialEngagementTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes a social engagement score for each individual in 2012 by summing up various indicators of social activities.

    Parameters:
    - social_engagement_cols: List of social engagement columns.

    Methods:
    - fit: Returns self.
    - transform: Adds 'social_engagement_12' to the dataset.
    """
    def __init__(self, social_engagement_cols):
        self.social_engagement_cols = social_engagement_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Adjust columns based on those present in X
        social_engagement_cols = [col for col in self.social_engagement_cols if col in X.columns]
        
        # Ensure columns are numeric and fill missing values
        if social_engagement_cols:
            X[social_engagement_cols] = X[social_engagement_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            # Sum social engagement score
            X['social_engagement_12'] = X[social_engagement_cols].sum(axis=1)
        else:
            X['social_engagement_12'] = np.nan
        return X

# Custom transformer to create preventive care index and health services usage
class HealthServicesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that creates indices for preventive care and health service usage for each individual in 2003 and 2012, and calculates changes over time.

    Parameters:
    - preventive_care_cols_03: List of preventive care columns for 2003.
    - preventive_care_cols_12: List of preventive care columns for 2012.
    - health_service_usage_cols_03: List of health service usage columns for 2003.
    - health_service_usage_cols_12: List of health service usage columns for 2012.

    Methods:
    - fit: Returns self.
    - transform: Adds indices and changes to the dataset.
    """
    def __init__(self, preventive_care_cols_03, preventive_care_cols_12, health_service_usage_cols_03, health_service_usage_cols_12):
        self.preventive_care_cols_03 = preventive_care_cols_03
        self.preventive_care_cols_12 = preventive_care_cols_12
        self.health_service_usage_cols_03 = health_service_usage_cols_03
        self.health_service_usage_cols_12 = health_service_usage_cols_12

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Adjust columns based on those present in X
        preventive_care_cols_03 = [col for col in self.preventive_care_cols_03 if col in X.columns]
        preventive_care_cols_12 = [col for col in self.preventive_care_cols_12 if col in X.columns]
        health_service_usage_cols_03 = [col for col in self.health_service_usage_cols_03 if col in X.columns]
        health_service_usage_cols_12 = [col for col in self.health_service_usage_cols_12 if col in X.columns]

        # Convert to numeric and fill missing values
        for cols in [preventive_care_cols_03, preventive_care_cols_12, health_service_usage_cols_03, health_service_usage_cols_12]:
            if cols:
                X[cols] = X[cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Create preventive care index for 2003 and 2012
        if preventive_care_cols_03:
            X['preventive_care_index_03'] = X[preventive_care_cols_03].sum(axis=1)
        else:
            X['preventive_care_index_03'] = np.nan

        if preventive_care_cols_12:
            X['preventive_care_index_12'] = X[preventive_care_cols_12].sum(axis=1)
        else:
            X['preventive_care_index_12'] = np.nan

        # Create health service usage score for 2003 and 2012
        if health_service_usage_cols_03:
            X['health_service_usage_03'] = X[health_service_usage_cols_03].sum(axis=1)
        else:
            X['health_service_usage_03'] = np.nan

        if health_service_usage_cols_12:
            X['health_service_usage_12'] = X[health_service_usage_cols_12].sum(axis=1)
        else:
            X['health_service_usage_12'] = np.nan

        # Calculate changes between 2003 and 2012
        valid_preventive = X['preventive_care_index_03'].notnull() & X['preventive_care_index_12'].notnull()
        X['preventive_care_change'] = np.nan
        X.loc[valid_preventive, 'preventive_care_change'] = X.loc[valid_preventive, 'preventive_care_index_12'] - X.loc[valid_preventive, 'preventive_care_index_03']

        valid_usage = X['health_service_usage_03'].notnull() & X['health_service_usage_12'].notnull()
        X['health_service_usage_change'] = np.nan
        X.loc[valid_usage, 'health_service_usage_change'] = X.loc[valid_usage, 'health_service_usage_12'] - X.loc[valid_usage, 'health_service_usage_03']

        return X

# Create a custom transformer that applies all custom transformations
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_common_features, ordinal_mappings, married_cols_03, married_cols_12,
                 chronic_illness_cols_03, chronic_illness_cols_12, adl_cols_03, adl_cols_12,
                 iadl_cols_03, iadl_cols_12, positive_mood_cols_03, positive_mood_cols_12,
                 negative_mood_cols_03, negative_mood_cols_12, lifestyle_cols_03, lifestyle_cols_12,
                 income_cols_03, income_cols_12, insurance_cols_03, insurance_cols_12, social_engagement_cols,
                 preventive_care_cols_03, preventive_care_cols_12, health_service_usage_cols_03, health_service_usage_cols_12):
        
        # Assign parameters to instance variables
        self.numerical_common_features = numerical_common_features
        self.ordinal_mappings = ordinal_mappings
        self.married_cols_03 = married_cols_03
        self.married_cols_12 = married_cols_12
        self.chronic_illness_cols_03 = chronic_illness_cols_03
        self.chronic_illness_cols_12 = chronic_illness_cols_12
        self.adl_cols_03 = adl_cols_03
        self.adl_cols_12 = adl_cols_12
        self.iadl_cols_03 = iadl_cols_03
        self.iadl_cols_12 = iadl_cols_12
        self.positive_mood_cols_03 = positive_mood_cols_03
        self.positive_mood_cols_12 = positive_mood_cols_12
        self.negative_mood_cols_03 = negative_mood_cols_03
        self.negative_mood_cols_12 = negative_mood_cols_12
        self.lifestyle_cols_03 = lifestyle_cols_03
        self.lifestyle_cols_12 = lifestyle_cols_12
        self.income_cols_03 = income_cols_03
        self.income_cols_12 = income_cols_12
        self.insurance_cols_03 = insurance_cols_03
        self.insurance_cols_12 = insurance_cols_12
        self.social_engagement_cols = social_engagement_cols
        self.preventive_care_cols_03 = preventive_care_cols_03
        self.preventive_care_cols_12 = preventive_care_cols_12
        self.health_service_usage_cols_03 = health_service_usage_cols_03
        self.health_service_usage_cols_12 = health_service_usage_cols_12
        
        # Initialize all custom transformers
        self.temporal_features = TemporalFeatureEngineer(numerical_common_features, ordinal_mappings)
        self.education_progression = EducationProgressionTransformer()
        self.marital_transition = MaritalTransitionTransformer(married_cols_03, married_cols_12)
        self.chronic_illness = ChronicIllnessTransformer(chronic_illness_cols_03, chronic_illness_cols_12)
        self.adl_iadl = ADLIADLTransformer(adl_cols_03, adl_cols_12, iadl_cols_03, iadl_cols_12)
        self.health_assessment_change = HealthAssessmentChangeTransformer()
        self.mood_score = MoodScoreTransformer(positive_mood_cols_03, positive_mood_cols_12, negative_mood_cols_03, negative_mood_cols_12)
        self.consistent_exercise = ConsistentExerciseTransformer()
        self.lifestyle_health_index = LifestyleHealthIndexTransformer(lifestyle_cols_03, lifestyle_cols_12)
        self.socioeconomic_features = SocioeconomicFeaturesTransformer(income_cols_03, income_cols_12, insurance_cols_03, insurance_cols_12)
        self.social_engagement = SocialEngagementTransformer(social_engagement_cols)
        self.health_services = HealthServicesTransformer(preventive_care_cols_03, preventive_care_cols_12, health_service_usage_cols_03, health_service_usage_cols_12)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = self.temporal_features.transform(X)
        X = self.education_progression.transform(X)
        X = self.marital_transition.transform(X)
        X = self.chronic_illness.transform(X)
        X = self.adl_iadl.transform(X)
        X = self.health_assessment_change.transform(X)
        X = self.mood_score.transform(X)
        X = self.consistent_exercise.transform(X)
        X = self.lifestyle_health_index.transform(X)
        X = self.socioeconomic_features.transform(X)
        X = self.social_engagement.transform(X)
        X = self.health_services.transform(X)
        return X
    

class InteractionTermsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Interaction: Health service usage change * Lifestyle health index (2012)
        if 'health_service_usage_change' in X.columns and 'lifestyle_health_index_12' in X.columns:
            X['health_lifestyle_interaction'] = (
                X['health_service_usage_change'] * X['lifestyle_health_index_12']
            )
        
        # Interaction: Education transition * Aggregate income (2012)
        if 'education_transition' in X.columns and 'aggregate_income_12' in X.columns:
            X['education_income_interaction'] = (
                X['education_transition'] * X['aggregate_income_12']
            )
        
        # Interaction: Social engagement (2012) * Positive mood change
        if 'social_engagement_12' in X.columns and 'positive_mood_change' in X.columns:
            X['social_mood_interaction'] = (
                X['social_engagement_12'] * X['positive_mood_change']
            )
        
        # Interaction: Preventive care change * Chronic illness count (2012)
        if 'preventive_care_change' in X.columns and 'chronic_illness_count_12' in X.columns:
            X['preventive_chronic_interaction'] = (
                X['preventive_care_change'] * X['chronic_illness_count_12']
            )
        
        return X
    
# SHAP-based feature selection as a custom transformer
class SHAPFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, base_model, num_features):
        self.base_model = base_model
        self.num_features = num_features
        self.selected_features = None

    def fit(self, X, y):
        # Train the base model for SHAP analysis
        self.base_model.fit(X, y)
        explainer = shap.TreeExplainer(self.base_model)
        shap_values = explainer.shap_values(X)
        # Calculate mean absolute SHAP values for feature importance
        feature_importances = np.abs(shap_values).mean(axis=0)
        # Select top features
        self.selected_features = np.argsort(feature_importances)[-self.num_features:]
        return self

    def transform(self, X):
    #    Return only the selected features
        return X[:, self.selected_features]



