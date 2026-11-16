
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "LLM Inference Logs & Performance Metrics"
    / "llm_inference_logs.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_latency_model.joblib"


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

FEATURES = [
    "prompt_tokens",
    "completion_tokens",
    "cache_hit_tokens",
    "ttft_ms",
    "tpot_ms",
    "estimated_cost_usd",
    "status_code",
]

TARGET = "total_latency_ms"

RANDOM_STATE = 42


# --------------------------------------------------
# 3. Load and prepare the dataset
# --------------------------------------------------

def prepare_data():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Same column normalization as the notebook
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_")
    )

    # Same missing-value treatment as the notebook
    df["estimated_cost_usd"] = (
        df["estimated_cost_usd"].fillna(0)
    )

    # Select the same features and target
    df_subset = df[FEATURES + [TARGET]].copy()

    # Check remaining missing values
    if df_subset.isnull().any().any():
        missing = df_subset.isnull().sum()
        raise ValueError(
            f"Missing values remain:\n{missing[missing > 0]}"
        )

    X = df_subset[FEATURES]
    y = df_subset[TARGET]

    return X, y


# --------------------------------------------------
# 4. Reproduce the original train/validation/test split
# --------------------------------------------------

def split_data(X, y):

    df_subset = X.copy()
    df_subset[TARGET] = y

    df_full_train, df_test = train_test_split(
        df_subset,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]

    return X_train, y_train


# --------------------------------------------------
# 5. Build and tune Random Forest
# --------------------------------------------------

def train_model(X_train, y_train):

    print("Training Random Forest...")

    rf_params = {
        "n_estimators": [50, 100, 150],
        "max_depth": [10, 20, None],
    }

    rf_grid = GridSearchCV(
        RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        rf_params,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    rf_grid.fit(X_train, y_train)

    best_model = rf_grid.best_estimator_

    print("Best parameters:", rf_grid.best_params_)

    return best_model


# --------------------------------------------------
# 6. Save the trained model
# --------------------------------------------------

def save_model(model):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved successfully: {MODEL_PATH}")


# --------------------------------------------------
# 7. Main function
# --------------------------------------------------

def main():

    X, y = prepare_data()

    X_train, y_train = split_data(X, y)

    model = train_model(X_train, y_train)

    save_model(model)


if __name__ == "__main__":
    main()
