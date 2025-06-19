from __future__ import annotations

import functools
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
import gradio as gr

# local modules  ------------------------------------------------------------
from data import assign_numbers, load_sales_dataset
from helper import evaluate_model, make_regression_pipeline
# --------------------------------------------------------------------------- 

#  Core training & comparison  -----------------------------------------------

@functools.lru_cache(maxsize=1)
def train_and_compare() -> Tuple[
    pd.DataFrame,                 # results_df
    matplotlib.figure.Figure,     # mae_bar_fig
    matplotlib.figure.Figure,     # accuracy_fig
    matplotlib.figure.Figure,     # seasonality_fig
    pd.DataFrame,                 # forecast_df
    matplotlib.figure.Figure      # forecast_fig
]:
    """Train/compare models & prepare all visuals. Heavy work is cached."""

    # ------------------------------------------------------------------ 1. load + clean data
    df = load_sales_dataset()

    cols_to_drop = [
        "CUSTOMERNAME", "ADDRESSLINE1", "ADDRESSLINE2", "PHONE",
        "CONTACTFIRSTNAME", "CONTACTLASTNAME", "STATE", "CITY", "POSTALCODE",
        "ORDERDATE", "PRODUCTCODE", "TERRITORY", "COUNTRY", "ORDERNUMBER",
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

    num_cols = ["QUANTITYORDERED", "PRICEEACH", "ORDERLINENUMBER",
                "QTR_ID", "MSRP", "YEAR_ID"]
    cat_cols = ["STATUS", "PRODUCTLINE", "DEALSIZE"]

    # ------------------------------------------------------------------ 2. define & train models
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
                ("rf", RandomForestRegressor(
                    n_estimators=300, random_state=0, n_jobs=-1)),
                ("gbr", GradientBoostingRegressor(
                    n_estimators=300, random_state=0)),
            ],
            final_estimator=RidgeCV(),
            n_jobs=-1,
        ),
    }

    results: List[dict] = []
    trained_models: Dict[str, TransformedTargetRegressor] = {}

    for name, base_model in models.items():
        pipe = make_regression_pipeline(num_cols, cat_cols, base_model)
        ttr = TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1
        )
        ttr.fit(X_train, y_train)
        results.append(evaluate_model(name, ttr, X_test, y_test))
        trained_models[name] = ttr

    results_df = (
        pd.DataFrame(results)
        .sort_values("MAE", ascending=True)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------ 3. visuals
    # 3-a MAE bar chart
    mae_bar_fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["MAE"])
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("Model Comparison - Mean Absolute Error")
    ax.tick_params(axis="x",rotation=15)

    # 3-b choose best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # 3-c accuracy scatter
    y_test_pred = best_model.predict(X_test)
    accuracy_fig = plot_test_accuracy(y_test, y_test_pred)

    # 3-d seasonality
    seasonality_fig = plot_seasonality(df)

    # 3-e forecast two years ahead
    forecast_df, forecast_fig = forecast_future_sales(best_model, X_train)

    return (
        results_df,
        mae_bar_fig,
        accuracy_fig,
        seasonality_fig,
        forecast_df,
        forecast_fig,
    )

# --------------------------------------------------------------------------- helper plots
def plot_test_accuracy(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--")       # 45° reference
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Predicted sales")
    ax.set_title(
        f"Test-set accuracy\n"
        f"MAE={mean_absolute_error(y_true, y_pred):.0f} | "
        f"R²={r2_score(y_true, y_pred):.2f}"
        f"MAPE={mean_absolute_percentage_error(y_true, y_pred):.2f}"
    )
    return fig


def plot_seasonality(df: pd.DataFrame):
    seasonal = (
        df.groupby("QTR_ID")["SALES"]
        .mean()
        .reindex([1, 2, 3, 4])
    )
    fig, ax = plt.subplots()
    ax.bar(seasonal.index, seasonal.values)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Average historical sales")
    ax.set_title("Seasonality - quarterly average")
    return fig


def forecast_future_sales(
    model,
    X_train: pd.DataFrame,
    years_ahead: int = 2
) -> Tuple[pd.DataFrame, matplotlib.figure.Figure]:
    last_year = int(X_train["YEAR_ID"].max())
    future_rows = []

    for year in range(last_year + 1, last_year + 1 + years_ahead):
        for qtr in range(1, 5):  # Q1..Q4
            sample = X_train.iloc[-1].copy()
            sample["YEAR_ID"] = year
            sample["QTR_ID"] = qtr
            future_rows.append(sample)

    X_future = pd.DataFrame(future_rows)
    y_pred = model.predict(X_future)

    forecast_df = X_future[["YEAR_ID", "QTR_ID"]].copy()
    forecast_df["PREDICTED_SALES"] = y_pred
    forecast_df["Period"] = (
        forecast_df["YEAR_ID"].astype(int).astype(str) +
        " Q" +
        forecast_df["QTR_ID"].astype(int).astype(str)
    )

    fig, ax = plt.subplots()
    ax.plot(forecast_df["Period"], forecast_df["PREDICTED_SALES"], marker="o")
    ax.set_title("📈 Forecasted Sales Trend")
    ax.set_ylabel("Predicted sales")
    ax.set_xlabel("Period")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    return forecast_df, fig

# --------------------------------------------------------------------------- Gradio UI

def run_comparison():
    """Thin wrapper for Gradio."""
    return train_and_compare()


def launch_ui():
    with gr.Blocks(title="Sales Model Comparison and Forecasting") as demo:
        gr.Markdown("# 🏎️ Classic-Cars Sales - Model Comparison & Forecasting")
        gr.Markdown(
            "Click **Run comparison** to train models, evaluate them, "
            "and see a two-year sales forecast."
        )

        run_btn = gr.Button("🚀 Run comparison", variant="primary")

        # ----- six output components in the same order as the tuple -----
        with gr.Row():
            leaderboard_df = gr.DataFrame(
                label="📊 Model Comparison (MAE)",
                interactive=False,
            )
            mae_bar_plot = gr.Plot(label="MAE Bar Chart")

        with gr.Row():
            accuracy_plot = gr.Plot(label="Test-set accuracy")
            seasonality_plot = gr.Plot(label="Seasonality (Quarterly)")

        with gr.Row():
            forecast_df = gr.DataFrame(
                label="🔮 Forecasted Sales",
                interactive=False,
            )
            forecast_plot = gr.Plot(label="Forecast Trend")

        run_btn.click(
            run_comparison,
            inputs=None,
            outputs=[
                leaderboard_df,   # results_df
                mae_bar_plot,     # mae_bar_fig
                accuracy_plot,    # accuracy_fig
                seasonality_plot, # seasonality_fig
                forecast_df,      # forecast_df
                forecast_plot,    # forecast_fig
            ],
        )

    demo.launch()


if __name__ == "__main__":
    launch_ui()
