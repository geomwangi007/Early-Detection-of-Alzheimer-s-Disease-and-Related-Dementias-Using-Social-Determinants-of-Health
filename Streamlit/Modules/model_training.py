from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from custom_transformers import ConvertToString
from sklearn.model_selection import train_test_split

def create_preprocessing_pipeline(num_cols, cat_cols):
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('to_string', ConvertToString()),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    return preprocessor

def train_model(X, y, preprocessor):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test
