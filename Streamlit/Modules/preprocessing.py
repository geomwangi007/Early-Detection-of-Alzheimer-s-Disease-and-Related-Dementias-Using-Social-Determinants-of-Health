import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .utils import ConvertToString

def load_and_merge_data(features_path, labels_path):
    # Load datasets
    train_features = pd.read_csv(features_path)
    train_labels = pd.read_csv(labels_path)
    # Merge datasets
    df = pd.merge(train_features, train_labels, on='uid', how='left')
    return df

def apply_mappings(df, mappings):
    # Apply categorical mappings
    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping)
    return df

def transform_data(df):
    # Transformation pipeline
    stubnames = list({col.rsplit('_', 1)[0] for col in df.columns if col.endswith('_03') or col.endswith('_12')})
    data_long = pd.wide_to_long(df, stubnames=stubnames, i=['uid', 'year', 'composite_score'], j='time', sep='_', suffix='\\d+').reset_index()
    return data_long

def preprocess_data(X):
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('to_string', ConvertToString()), ('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])
    return preprocessor


