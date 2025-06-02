# ml_api/utils.py
import joblib
import os
import shap
import numpy as np
import pandas as pd
from typing import Tuple, List, Any

MODEL_DIR = "artifacts"
MODEL_FILE = os.path.join(MODEL_DIR, "model.joblib")
FEATURE_NAMES_FILE = os.path.join(MODEL_DIR, "feature_names.joblib")
SHAP_BACKGROUND_FILE = os.path.join(MODEL_DIR, "shap_background.joblib")

def save_artifacts(model, selected_feature_names, X_train_processed_sample):
    """
    Saves the fitted preprocessor, trained model, selected feature names,
    and a sample of processed training data for SHAP background.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(selected_feature_names, FEATURE_NAMES_FILE)
    joblib.dump(X_train_processed_sample, SHAP_BACKGROUND_FILE)
    return MODEL_FILE, FEATURE_NAMES_FILE, SHAP_BACKGROUND_FILE

def load_artifacts() -> Tuple[Any, Any, List[str], Any]:
    """
    Loads the fitted preprocessor, trained model, selected feature names,
    and SHAP background data.
    Raises FileNotFoundError if artifacts are not found.
    """
    if not os.path.exists(MODEL_FILE) or \
       not os.path.exists(FEATURE_NAMES_FILE) or \
       not os.path.exists(SHAP_BACKGROUND_FILE):
        raise FileNotFoundError("Model artifacts not found. Please train the model first.")

    model = joblib.load(MODEL_FILE)
    selected_feature_names = joblib.load(FEATURE_NAMES_FILE)
    shap_background_data = joblib.load(SHAP_BACKGROUND_FILE)
    return model, selected_feature_names, shap_background_data

def get_shap_explainer(model, X_background_df: pd.DataFrame):
    """
    Initializes and returns a SHAP TreeExplainer.
    """
    # For TreeExplainer, passing a background dataset (e.g., a sample of training data)
    # is recommended for accurate expected_value calculation.
    explainer = shap.TreeExplainer(model, X_background_df)
    return explainer

def calculate_shap_values(explainer, X_data_df: pd.DataFrame):
    """
    Calculates SHAP values for the positive class (index 1) for given data.
    """
    # explainer.shap_values returns a list of arrays for multi-output models (e.g., multi-class classification)
    # For binary classification, it returns [shap_values_for_class_0, shap_values_for_class_1]
    shap_values = explainer.shap_values(X_data_df)
    # We typically want explanations for the positive class (assuming binary classification)
    shap_values_positive_class = shap_values[1] # For class 1
    expected_value_positive_class = explainer.expected_value[1] # Expected value for class 1

    return shap_values_positive_class.tolist(), expected_value_positive_class