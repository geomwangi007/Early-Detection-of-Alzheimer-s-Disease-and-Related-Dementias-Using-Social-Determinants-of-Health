# data_handling.py
import pandas as pd
from .mappings import apply_mappings

def load_data(features_path, labels_path):
    train_features = pd.read_csv(features_path)
    train_labels = pd.read_csv(labels_path)
    df = pd.merge(train_features, train_labels, on='uid', how='left')
    return apply_mappings(df)
