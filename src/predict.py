
from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# 1. Locate the saved model
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "best_latency_model.joblib"
)


# --------------------------------------------------
# 2. Define the model features
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


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run train.py first."
        )

    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# 4. Prepare incoming data
# --------------------------------------------------

def prepare_input(
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit_tokens: int,
    ttft_ms: float,
    tpot_ms: float,
    estimated_cost_usd: float,
    status_code: int,
):

    input_data = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "estimated_cost_usd": estimated_cost_usd,
        "status_code": status_code,
    }

    return pd.DataFrame(
        [input_data],
        columns=FEATURES,
    )


# --------------------------------------------------
# 5. Generate prediction
# --------------------------------------------------

def predict_latency(
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit_tokens: int,
    ttft_ms: float,
    tpot_ms: float,
    estimated_cost_usd: float,
    status_code: int,
) -> float:

    input_data = prepare_input(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_hit_tokens=cache_hit_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        estimated_cost_usd=estimated_cost_usd,
        status_code=status_code,
    )

    prediction = model.predict(input_data)

    return float(prediction[0])
