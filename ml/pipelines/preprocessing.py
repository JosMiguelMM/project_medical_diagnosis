from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

def create_preprocessing_pipeline(numerical_features, categorical_features, data):
    """
    Crea un pipeline de preprocesamiento para datos numéricos y categóricos.

    Args:
        numerical_features (list): Lista de nombres de columnas numéricas.
        categorical_features (list): Lista de nombres de columnas categóricas.
        data: El dataframe.

    Returns:
        sklearn.pipeline.Pipeline: El pipeline de preprocesamiento.
    """

    # Pipeline para características numéricas
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Imputación de valores faltantes
        ('scaler', StandardScaler())  # Escalado de características
    ])

    # Pipeline para características categóricas
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputación de valores faltantes
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Codificación One-Hot
    ])

    # Identifica las columnas que realmente existen en los datos.

    present_numerical = [col for col in numerical_features if col in data.columns]
    present_categorical = [col for col in categorical_features if col in data.columns]

    # Combina los pipelines usando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, present_numerical),
            ('cat', categorical_pipeline, present_categorical)
        ],
        remainder='passthrough'  # Deja pasar las columnas que no se especificaron
    )

    return preprocessor