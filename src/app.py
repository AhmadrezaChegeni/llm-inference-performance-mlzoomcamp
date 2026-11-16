
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_latency


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="LLM Inference Latency Prediction API",
    description=(
        "An ML API that predicts LLM inference "
        "latency using a trained Random Forest model."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# 2. Define incoming request structure
# --------------------------------------------------

class PredictionRequest(BaseModel):

    prompt_tokens: int = Field(ge=0)

    completion_tokens: int = Field(ge=0)

    cache_hit_tokens: int = Field(ge=0)

    ttft_ms: float = Field(ge=0, allow_inf_nan=False)

    tpot_ms: float = Field(ge=0, allow_inf_nan=False)

    estimated_cost_usd: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    status_code: int = Field(ge=100, le=599)


# --------------------------------------------------
# 3. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "LLM Latency Prediction API is running",
        "model": "Random Forest",
    }


# --------------------------------------------------
# 4. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "loaded",
    }


# --------------------------------------------------
# 5. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        prediction = predict_latency(
            prompt_tokens=request.prompt_tokens,
            completion_tokens=request.completion_tokens,
            cache_hit_tokens=request.cache_hit_tokens,
            ttft_ms=request.ttft_ms,
            tpot_ms=request.tpot_ms,
            estimated_cost_usd=request.estimated_cost_usd,
            status_code=request.status_code,
        )

        return {
            "predicted_total_latency_ms": prediction,
            "unit": "milliseconds",
            "model": "Random Forest",
        }

    except (ValueError, TypeError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
