# 🏎️ Classic‑Cars Sales — Model Comparison

A small, self‑contained project that trains four regression models on the Classic Cars sales dataset and lets you explore their performance in an interactive **Gradio** app.

---

## 📋 Features

| Component            | Description                                                                   |
| -------------------- | ----------------------------------------------------------------------------- |
| **Data cleaning**    | Drops PII / sparse columns, label‑encodes categorical features.               |
| **Models**           | Linear Regression, Gradient Boosting, Random Forest, and a Stacking ensemble. |
| **Target transform** | Uses `log1p` / `expm1` via `TransformedTargetRegressor` for stability.        |
| **Metrics**          | Mean Absolute Error (MAE) & Mean Absolute Percentage Error (MAPE).            |
| **UI**               | One‑click “Run comparison” button, interactive table, and MAE bar‑chart.      |

---

## 🚀 Quick start

```bash
# 1. Clone / download the repo
git clone https://github.com/Sambhav242005/FUTURE_ML_01.git
# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Gradio app
python main.py

# 4. Go to Browser enter this url to access it
http://127.0.0.1:7860

```

Your browser will open automatically. Click **Run comparison** to trigger training (≈10 s on a modern laptop). Results are cached, so subsequent clicks are instant.

---

## 🗂️ Project structure

```
.
├── main.py                     # Main script model run & UI
├── data.py                     # load_sales_dataset(), assign_numbers()
├── helper.py                   # Helper modules
├── requirements.txt            # Python dependencies
├── README.md                   # You are here
```

---

## 🛠️ Customisation

### Add or remove models

Edit the `models` dict in **sales\_model\_comparison.py**:

```python
models = {
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(tree_method="hist"),  # example
}
```

### Hyper‑parameter tuning

Swap `GradientBoostingRegressor` / `RandomForestRegressor` for the HistGradientBoosting or plug in `GridSearchCV` inside `make_regression_pipeline()`.

### Additional metrics

Extend `evaluate_model()` with RMSE, R², etc.

---

## 📑 License

Released under the MIT License — see `LICENSE` for details.
