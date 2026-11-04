from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "yield" / "processed" / "haryana_wheat_yield_weather.csv"
OUTPUT_DIR = ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "district",
    "year_start",
    "yield_lag_1",
    "yield_lag_2",
    "historical_avg_yield",
    "rainfall_mm",
    "temp_max_c",
    "temp_min_c",
    "humidity_pct",
]
TARGET = "yield_tonnes_per_hectare"

# Chronological holdout: train through 2018, evaluate on 2019–2022.
TRAIN_END_YEAR = 2018
TEST_START_YEAR = 2019

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Keep only rows where all model inputs and target are available.
    data = df.dropna(subset=required).copy()
    data["year_start"] = pd.to_numeric(data["year_start"], errors="raise")

    train = data[data["year_start"] <= TRAIN_END_YEAR].copy()
    test = data[data["year_start"] >= TEST_START_YEAR].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Empty chronological split. Train <= {TRAIN_END_YEAR}: "
            f"{len(train)}, test >= {TEST_START_YEAR}: {len(test)}"
        )

    # Refit a fresh model using the saved artifact's observed XGBoost settings.
    # This avoids evaluating the original artifact that was trained on all 388 rows.
    preprocessor = ColumnTransformer(
        transformers=[
            ("district_encoder",
             OneHotEncoder(handle_unknown="ignore"),
             ["district"]),
        ],
        remainder="passthrough",
    )

    regressor = XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", regressor),
    ])

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results = test[["district", "year", TARGET]].copy()
    results["predicted_yield_tonnes_per_hectare"] = predictions
    results["absolute_error"] = (
        results[TARGET] - results["predicted_yield_tonnes_per_hectare"]
    ).abs()
    results["error"] = (
        results["predicted_yield_tonnes_per_hectare"] - results[TARGET]
    )

    results_path = OUTPUT_DIR / "yield_predictions.csv"
    results.to_csv(results_path, index=False)

    metrics = {
        "evaluation_method": "chronological holdout with fresh retraining",
        "training_years": f"2004-{TRAIN_END_YEAR}",
        "testing_years": f"{TEST_START_YEAR}-2022",
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "mae_tonnes_per_hectare": float(mae),
        "rmse_tonnes_per_hectare": float(rmse),
        "r2": float(r2),
        "warning": (
            "This is a new chronological holdout evaluation. The saved production "
            "artifact was trained on all usable rows, so its in-sample score must "
            "not be reported as independent test performance. This evaluation "
            "re-trains a fresh model with the saved artifact's observed XGBoost "
            "hyperparameters and equivalent district one-hot preprocessing."
        ),
    }

    metrics_path = OUTPUT_DIR / "yield_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Actual vs predicted plot.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual yield (tonnes/hectare)")
    ax.set_ylabel("Predicted yield (tonnes/hectare)")
    ax.set_title("Wheat Yield: Actual vs Predicted (2019–2022)")
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "yield_actual_vs_predicted.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print("\n===== CHRONOLOGICAL YIELD EVALUATION =====")
    print(f"Training rows: {len(train)} (years <= {TRAIN_END_YEAR})")
    print(f"Testing rows:  {len(test)} (years >= {TEST_START_YEAR})")
    print(f"MAE:  {mae:.4f} tonnes/hectare")
    print(f"RMSE: {rmse:.4f} tonnes/hectare")
    print(f"R²:   {r2:.4f}")
    print("\nSaved:")
    print(results_path)
    print(metrics_path)
    print(plot_path)
    print("\nNote: Fresh model retrained on earlier years; not the all-data saved artifact.")

if __name__ == "__main__":
    main()
