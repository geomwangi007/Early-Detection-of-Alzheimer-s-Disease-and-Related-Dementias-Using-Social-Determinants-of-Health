# model_training.py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .preprocessing import create_preprocessor

def train_model(data, target, num_cols, cat_cols):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ridge', Ridge())
    ])

    # Hyperparameter tuning
    param_grid = {'ridge__alpha': [0.01, 0.1, 1, 10, 100], 'ridge__solver': ['auto', 'lsqr', 'sparse_cg', 'sag', 'lbfgs']}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Best model and evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return best_model, {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

