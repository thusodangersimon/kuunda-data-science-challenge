import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from category_encoders import TargetEncoder
from pylab import plt
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeClassifierCV
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.utils.model_io import save_artifacts
from utils.get_data import get_all_data, load_training_data

bucket_name = 'kuunda-datascience-challenge'
data_path = '/app/data'


def download_and_load_data(bucket_name: str, data_path: str) -> pd.DataFrame:
    """
    Downloads data from S3 bucket and loads it into a DataFrame.
    """
    get_all_data(bucket_name, data_path)
    df = load_training_data(data_path)
    return df


def get_columns_splits(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits the DataFrame into categorical and numerical columns.
    """
    # split out features into numeric and categorica/ordinal base on data type
    features = [col for col in df.columns if col.startswith('feature')]
    target_col = ['default_ind']
    numeric_cols = []
    categorical_cols = []

    for col in features:
        if pd.api.types.is_float_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or \
                pd.api.types.is_integer_dtype(df[col]) or \
                pd.api.types.is_bool_dtype(df[col]):
            categorical_cols.append(col)
        else:
            print(f"Warning: Column '{col}' has an unexpected dtype: {df[col].dtype}.")
    return categorical_cols, target_col, numeric_cols


def split_data(df: pd.DataFrame, dev_pct: float, val_pct: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into train, dev, and validation sets.
    """
    # split into sets
    train_val_df, dev_df = train_test_split(df, test_size=dev_pct)
    train_df, val_df = train_test_split(train_val_df, test_size=val_pct / (1 - dev_pct))
    og_size = df.shape[0]
    logging.debug('shapes of data are training={}, val={}, dev={}'.format(
        train_df.shape[0] / og_size,
        val_df.shape[0] / og_size,
        dev_df.shape[0] / og_size)
    )
    return train_df, val_df, dev_df


def get_model(model: ClassifierMixin, categorical_cols: List[str], numeric_cols: List[str]) -> Pipeline:
    """
    gets a model pipeline with preprocessing and feature selection.
    :param model: ClassifierMixin, e.g., RandomForestClassifier, HistGradientBoostingClassifier, etc.
    :param categorical_cols: List[str], list of categorical columns
    :param numeric_cols: List[str], list of numeric columns
    :return: Pipeline
    """

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    ordinal_categorical_transformer = Pipeline(steps=[
        ('target_encoder', TargetEncoder(cols=categorical_cols))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_cols),
            ('ordinal', ordinal_categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # Keep columns not specified (e.g., target variable)
    )

    feature_selection_pipeline = Pipeline(steps=[
        ('pre-proceesor', preprocessor),
        ('feature_selection', SelectFromModel(model)),
        ('model', model)
    ])
    return feature_selection_pipeline


def get_explainer(model: ClassifierMixin, X_background_df: pd.DataFrame) -> shap.TreeExplainer:
    """
    Initializes and returns a SHAP TreeExplainer.
    """
    # For TreeExplainer, passing a background dataset (e.g., a sample of training data)
    # is recommended for accurate expected_value calculation.
    explainer = shap.TreeExplainer(model, X_background_df)
    return explainer


def train_model(model: ClassifierMixin, train_df: pd.DataFrame) -> Tuple[ClassifierMixin, List[str], pd.DataFrame]:
    """
    Trains the model on the training DataFrame.
    """
    categorical_cols, target_col, numeric_cols = get_columns_splits(train_df)
    X_train = train_df[categorical_cols + numeric_cols]
    y_train = train_df[target_col].values.ravel()
    # Get the model pipeline
    model_pipeline = get_model(model, categorical_cols, numeric_cols)

    # Fit the model
    model_pipeline.fit(X_train, y_train)

    # Get selected feature names
    selected_feature_names = X_train.columns[model_pipeline.named_steps['feature_selection'].get_support()].tolist()

    # Get a sample of processed training data for SHAP background
    X_train_processed_sample = pd.DataFrame(model_pipeline.named_steps['pre-proceesor'].transform(X_train),
                                            columns=model_pipeline.named_steps['pre-proceesor'].get_feature_names_out())

    return model_pipeline, selected_feature_names, X_train_processed_sample


def main():
    logging.basicConfig(level=logging.INFO)
    # Download and load data
    df = download_and_load_data(bucket_name, data_path)

    # Split data into train, validation, and dev sets
    train_df, val_df, dev_df = split_data(df, dev_pct=0.2, val_pct=0.2)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    trained_model, selected_feature_names, X_train_processed_sample = train_model(model, train_df)

    # Get SHAP explainer
    explainer = get_explainer(trained_model.named_steps['model'], X_train_processed_sample)

    logging.info("Model training completed successfully.")
    save_artifacts(trained_model, selected_feature_names)
    return trained_model, selected_feature_names, explainer


if __name__ == "__main__":
    trained_model, selected_feature_names, explainer = main()
    logging.info(f"Trained model: {trained_model}")
    logging.info(f"Selected feature names: {selected_feature_names}")
    logging.info(f"SHAP Explainer: {explainer}")
