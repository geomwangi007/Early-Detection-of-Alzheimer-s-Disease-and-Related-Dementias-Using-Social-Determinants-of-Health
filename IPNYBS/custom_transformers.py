from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Custom Transformers and mappings as defined in the notebook
# ================================================

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_common_features, ordinal_mappings):
        self.numerical_common_features = numerical_common_features
        self.ordinal_mappings = ordinal_mappings
        
    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        X = X.copy()
        
        # Handle ordinal variables
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
                X[change_col] = X[col_12] - X[col_03]
            else:
                X[change_col] = np.nan  # Handle missing columns
        
        # Determine last feature year
        X['last_feature_year'] = X.apply(self.get_last_feature_year, axis=1)

        # Calculate time gap
        X['time_gap'] = X['year'] - X['last_feature_year']
        
        # Drop 'last_feature_year' if not needed
        X.drop(columns=['last_feature_year'], inplace=True)
        
        return X

    @staticmethod
    def get_last_feature_year(row):
        if not pd.isnull(row.get('age_12')):
            return 2012
        elif not pd.isnull(row.get('age_03')):
            return 2003
        else:
            return np.nan  # No data available

# Education Progression
class EducationProgressionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X = X.copy()
        if 'edu_gru_03' in X.columns and 'edu_gru_12' in X.columns:
            X['education_transition'] = X['edu_gru_12'] - X['edu_gru_03']
        else:
            X['education_transition'] = np.nan
        return X
    
# Marital status stability
class MaritalTransitionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, married_cols_03, married_cols_12):
        self.married_cols_03 = married_cols_03
        self.married_cols_12 = married_cols_12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.married_cols_03 and self.married_cols_12:
            X['marital_transition'] = (
                X[self.married_cols_03].sum(axis=1) != X[self.married_cols_12].sum(axis=1)
            ).astype(int)
        else:
            X['marital_transition'] = 0
        return X

# Count of chronic illnesses
class ChronicIllnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, chronic_illness_cols_03, chronic_illness_cols_12):
        self.chronic_illness_cols_03 = chronic_illness_cols_03
        self.chronic_illness_cols_12 = chronic_illness_cols_12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Sum chronic illnesses
        X['chronic_illness_count_03'] = X[self.chronic_illness_cols_03].sum(axis=1)
        X['chronic_illness_count_12'] = X[self.chronic_illness_cols_12].sum(axis=1)
        return X

# Limitations of Activities of daily living count and progression
class ADLIADLTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, adl_cols_03, adl_cols_12, iadl_cols_03, iadl_cols_12):
        self.adl_cols_03 = adl_cols_03
        self.adl_cols_12 = adl_cols_12
        self.iadl_cols_03 = iadl_cols_03
        self.iadl_cols_12 = iadl_cols_12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Sum ADL limitations
        X['total_adl_limitations_03'] = X[self.adl_cols_03].sum(axis=1)
        X['total_adl_limitations_12'] = X[self.adl_cols_12].sum(axis=1)
        # Sum IADL limitations
        X['total_iadl_limitations_03'] = X[self.iadl_cols_03].sum(axis=1)
        X['total_iadl_limitations_12'] = X[self.iadl_cols_12].sum(axis=1)
        # Calculate progression
        X['adl_iadl_progression'] = (
            (X['total_adl_limitations_12'] + X['total_iadl_limitations_12']) -
            (X['total_adl_limitations_03'] + X['total_iadl_limitations_03'])
        )
        return X

# Self Reported Health Change
class HealthAssessmentChangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'glob_hlth_03' in X.columns and 'glob_hlth_12' in X.columns:
            X['health_self_assessment_change'] = X['glob_hlth_12'] - X['glob_hlth_03']
        else:
            X['health_self_assessment_change'] = np.nan
        return X

# Custom transformer to engineer positive and negative mood scores
class MoodScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, positive_mood_cols_03, positive_mood_cols_12, negative_mood_cols_03, negative_mood_cols_12):
        self.positive_mood_cols_03 = positive_mood_cols_03
        self.positive_mood_cols_12 = positive_mood_cols_12
        self.negative_mood_cols_03 = negative_mood_cols_03
        self.negative_mood_cols_12 = negative_mood_cols_12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Create aggregate scores for positive and negative moods in 2003
        X['positive_mood_score_03'] = X[self.positive_mood_cols_03].sum(axis=1)
        X['negative_mood_score_03'] = X[self.negative_mood_cols_03].sum(axis=1)
        
        # Create aggregate scores for positive and negative moods in 2012
        X['positive_mood_score_12'] = X[self.positive_mood_cols_12].sum(axis=1)
        X['negative_mood_score_12'] = X[self.negative_mood_cols_12].sum(axis=1)
        
        # Calculate mood changes over time
        X['positive_mood_change'] = X['positive_mood_score_12'] - X['positive_mood_score_03']
        X['negative_mood_change'] = X['negative_mood_score_12'] - X['negative_mood_score_03']
        
        return X

# Consistency of exercise tracking
class ConsistentExerciseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'exer_3xwk_03' in X.columns and 'exer_3xwk_12' in X.columns:
            X['consistent_exercise'] = ((X['exer_3xwk_03'] == 1) & (X['exer_3xwk_12'] == 1)).astype(int)
        else:
            X['consistent_exercise'] = np.nan
        return X

# Alcohol and smoking tracking
class LifestyleHealthIndexTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lifestyle_cols_03, lifestyle_cols_12):
        self.lifestyle_cols_03 = lifestyle_cols_03
        self.lifestyle_cols_12 = lifestyle_cols_12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['lifestyle_health_index_03'] = X[self.lifestyle_cols_03].sum(axis=1)
        X['lifestyle_health_index_12'] = X[self.lifestyle_cols_12].sum(axis=1)
        return X

# Income and insurance coverage
class SocioeconomicFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, income_cols_03, income_cols_12, insurance_cols_03, insurance_cols_12):
        self.income_cols_03 = income_cols_03
        self.income_cols_12 = income_cols_12
        self.insurance_cols_03 = insurance_cols_03
        self.insurance_cols_12 = insurance_cols_12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Aggregate income
        X['aggregate_income_03'] = X[self.income_cols_03].sum(axis=1)
        X['aggregate_income_12'] = X[self.income_cols_12].sum(axis=1)
        # Insurance coverage depth
        X['insurance_coverage_depth_03'] = X[self.insurance_cols_03].sum(axis=1)
        X['insurance_coverage_depth_12'] = X[self.insurance_cols_12].sum(axis=1)
        # Insurance continuity
        X['insurance_continuity'] = ((X['insurance_coverage_depth_03'] > 0) & (X['insurance_coverage_depth_12'] > 0)).astype(int)
        return X

# Social Engagement score
class SocialEngagementTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, social_engagement_cols):
        self.social_engagement_cols = social_engagement_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Convert columns to numeric, coercing errors to NaN
        X[self.social_engagement_cols] = X[self.social_engagement_cols].apply(pd.to_numeric, errors='coerce')
        # Fill NaN values with 0
        X[self.social_engagement_cols] = X[self.social_engagement_cols].fillna(0)
        # Sum the engagement activities
        X['social_engagement_12'] = X[self.social_engagement_cols].sum(axis=1)
        return X

# Custom transformer to create preventive care index and health services usage
class HealthServicesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, preventive_care_cols_03, preventive_care_cols_12, health_service_usage_cols_03, health_service_usage_cols_12):
        self.preventive_care_cols_03 = preventive_care_cols_03
        self.preventive_care_cols_12 = preventive_care_cols_12
        self.health_service_usage_cols_03 = health_service_usage_cols_03
        self.health_service_usage_cols_12 = health_service_usage_cols_12

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Create preventive care index for 2003 and 2012
        X['preventive_care_index_03'] = X[self.preventive_care_cols_03].sum(axis=1)
        X['preventive_care_index_12'] = X[self.preventive_care_cols_12].sum(axis=1)
        
        # Create health service usage score for 2003 and 2012
        X['health_service_usage_03'] = X[self.health_service_usage_cols_03].sum(axis=1)
        X['health_service_usage_12'] = X[self.health_service_usage_cols_12].sum(axis=1)
        
        # Calculate changes between 2003 and 2012
        X['preventive_care_change'] = X['preventive_care_index_12'] - X['preventive_care_index_03']
        X['health_service_usage_change'] = X['health_service_usage_12'] - X['health_service_usage_03']
        
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
    
# ================================================

# Define mappings for ordinal columns
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
