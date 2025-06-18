from typing import List
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def make_regression_pipeline(
    num_cols: List[str],
    cat_cols: List[str],
    model: RegressorMixin,
) -> Pipeline:
    """Return preprocessing + model pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="passthrough",
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def evaluate_model(
    name: str,
    model: TransformedTargetRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Compute MAE & MAPE for *model* on test data."""
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE (%)": mean_absolute_percentage_error(y_test, y_pred) * 100,
    }

