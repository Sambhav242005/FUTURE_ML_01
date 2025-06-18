from __future__ import annotations

import matplotlib

from helper import evaluate_model, make_regression_pipeline

"""
Gradio UI: compare regression models on the classic-cars sales dataset.

Run this file directly:

    $ python sales_model_comparison.py

It will open a browser window with a **“Run comparison”** button that:
1. loads and cleans the dataset,
2. fits Linear Regression, Gradient Boosting, Random Forest and a Stacking ensemble,
3. displays an interactive table of MAE & MAPE on the test split, plus a bar-chart.

The heavy work happens only when you click the button, so the UI starts quickly.
"""

from typing import Dict, List, Tuple
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split

from data import assign_numbers, load_sales_dataset

###############################################################################
# Core training & comparison (cached to avoid re-training per click)
###############################################################################

@functools.lru_cache(maxsize=1)
def train_and_compare() -> Tuple[pd.DataFrame, "matplotlib.figure.Figure"]:
    """Return (results_df, bar_chart_figure). Heavy work is cached."""

    # ------------------------------------------------------------------
    # 1. Load & clean data
    # ------------------------------------------------------------------
    df = load_sales_dataset()

    cols_to_drop = [
        "CUSTOMERNAME",
        "ADDRESSLINE1",
        "ADDRESSLINE2",
        "PHONE",
        "CONTACTFIRSTNAME",
        "CONTACTLASTNAME",
        "STATE",
        "CITY",
        "POSTALCODE",
        "ORDERDATE",
        "PRODUCTCODE",
        "TERRITORY",
        "COUNTRY",
        "ORDERNUMBER",
        "MONTH_ID",
    ]
    df = df.drop(cols_to_drop, axis=1)

    for col in ["STATUS", "PRODUCTLINE", "YEAR_ID", "DEALSIZE"]:
        df[col], _ = assign_numbers(df[col])

    X = df.drop("SALES", axis=1)
    y = df["SALES"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )

    num_cols = [
        "QUANTITYORDERED",
        "PRICEEACH",
        "ORDERLINENUMBER",
        "QTR_ID",
        "MSRP",
        "YEAR_ID",
    ]
    cat_cols = ["STATUS", "PRODUCTLINE", "DEALSIZE"]

    # ------------------------------------------------------------------
    # 2. Define models
    # ------------------------------------------------------------------
    models: Dict[str, RegressorMixin] = {
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, random_state=0, max_depth=3
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=850, max_depth=15, n_jobs=-1, random_state=0
        ),
        "Stacking": StackingRegressor(
            estimators=[
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=300, random_state=0, n_jobs=-1
                    ),
                ),
                (
                    "gbr",
                    GradientBoostingRegressor(n_estimators=300, random_state=0),
                ),
            ],
            final_estimator=RidgeCV(),
            n_jobs=-1,
        ),
    }

    results: List[dict] = []

    for name, base_model in models.items():
        pipe = make_regression_pipeline(num_cols, cat_cols, base_model)
        ttr = TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1
        )
        ttr.fit(X_train, y_train)
        results.append(evaluate_model(name, ttr, X_test, y_test))

    results_df = (
        pd.DataFrame(results)
        .sort_values("MAE", ascending=True)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 3. Build bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["MAE"])
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("Model Comparison - Mean Absolute Error")
    ax.tick_params(axis="x")

    return results_df, fig

###############################################################################
# Gradio UI
###############################################################################

def run_comparison() -> Tuple[pd.DataFrame, "matplotlib.figure.Figure"]:
    """Gradio wrapper - returns DataFrame & plot."""
    return train_and_compare()


def launch_ui():
    with gr.Blocks(title="Sales Model Comparison") as demo:
        gr.Markdown("# 🏎️ Classic-Cars Sales - Model Comparison")
        gr.Markdown(
            "Click **Run comparison** to train four models on the dataset and "
            "see how they stack up. Results are cached after the first run.")

        run_btn = gr.Button("🚀 Run comparison", variant="primary")
        df_out = gr.DataFrame(headers=["Model", "MAE", "MAPE (%)"], interactive=False)
        plot_out = gr.Plot(label="MAE bar-chart")

        run_btn.click(run_comparison, inputs=None, outputs=[df_out, plot_out])

    demo.launch()

if __name__ == "__main__":
    launch_ui()
