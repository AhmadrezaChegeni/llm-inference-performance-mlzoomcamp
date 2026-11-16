# llm-inference-performance-mlzoomcamp
Machine Learning Zoomcamp Midterm Project - LLM Inference Latency Prediction
# LLM Inference Performance & Latency Prediction API

An end-to-end Machine Learning web service that predicts LLM inference latency using a trained Random Forest regressor, served via **FastAPI** and containerized with **Docker**.

---

## 📋 Table of Contents
1. [Project Overview & Problem Statement](#-project-overview--problem-statement)
2. [Solution Architecture](#-solution-architecture)
3. [Project Structure](#-project-structure)
4. [Getting Started & Local Installation](#-getting-started--local-installation)
5. [Running the API Locally](#-running-the-api-locally)
6. [Docker Containerization](#-docker-containerization)
7. [API Endpoints & Usage Example](#-api-endpoints--usage-example)

---

## 🎯 Project Overview & Problem Statement

### The Problem
Large Language Model (LLM) serving costs and user experience heavily depend on inference performance metrics like Time to First Token (TTFT) and Time per Output Token (TPOT). Predicting inference latency accurately *before* or *during* request dispatch helps optimize resource routing, budget estimation, and user SLA management. However, raw inference times fluctuate based on token counts, cache hits, and underlying hardware states.

### The Solution
We built a supervised machine learning regression pipeline utilizing a **Random Forest Regressor** trained on historical inference runs. The pipeline takes request metadata (such as prompt tokens, completion tokens, cache hit tokens, TTFT, TPOT, and cost) and predicts the total latency (`predicted_total_latency_ms`). The model is exposed via a production-ready **FastAPI** application and fully containerized using **Docker** for reliable, reproducible deployment anywhere.

---

## 🏗️ Solution Architecture

📁 Project Structure
```text
[ Client Request ] ---> [ FastAPI /predict Endpoint ] ---> [ Pydantic Validation ] 
                                                                    │
                                                                    ▼
[ JSON Response ] <--- [ Random Forest Model (.joblib) ] <--- [ Feature Processing ]
llm-inference-performance-mlzoomcamp/
│
├── data/                    # Raw and processed datasets
├── models/                  # Serialized model artifact (.joblib)
├── src/
│   ├── app.py               # FastAPI application and routing logic
│   └── train.py             # Model training script
│
├── .dockerignore            # Files to exclude from Docker build context
├── .gitignore               # Git ignored files (virtualenvs, etc.)
├── Dockerfile               # Container definition for the API
├── requirements.txt         # Pinned python dependencies
└── README.md                # Project documentation

⚙️ Getting Started & Local Installation
Follow these steps to clone and set up the repository on your local machine:

1. Clone the Repository

git clone [https://github.com/AhmadrezaChegeni/llm-inference-performance-mlzoomcamp.git](https://github.com/AhmadrezaChegeni/llm-inference-performance-mlzoomcamp.git)
cd llm-inference-performance-mlzoomcamp

2. Create and Activate a Virtual Environment

python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate
# On macOS/Linux:
source .venv/bin/activate

3. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

🚀 Running the API Locally
You can launch the FastAPI server directly on your host machine using Uvicorn:

uvicorn src.app:app --reload --host 127.0.0.1 --port 8000

Interactive Swagger UI Docs: Open http://127.0.0.1:8000/docs in your browser to test endpoints interactively.

Alternative ReDoc UI: Visit http://127.0.0.1:8000/redoc.
🐳 Docker Deployment & Containerization

Containerizing the application ensures it runs with the exact same dependencies and environment regardless of the host machine.

1. Build the Docker Image

Make sure Docker Desktop is running, then run the following command in your project root:

docker build -t llm-latency-api .

2. Run the Docker Container

Map port 8000 from your container to your host machine:

docker run -p 8000:8000 llm-latency-api

3. Verify the Containerized Service

Once running, open http://127.0.0.1:8000/docs in your browser. The API is now fully operational inside the container environment.


💡 API Usage & Example
Endpoint

    POST /predict: Accepts inference parameters and returns the predicted total latency in milliseconds.

Example Request (cURL)

curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "cache_hit_tokens": 10,
    "ttft_ms": 120.0,
    "tpot_ms": 25.0,
    "estimated_cost_usd": 0.002,
    "status_code": 200
  }'

Example Response

{
  "predicted_total_latency_ms": 1331.9866666666667,
  "unit": "milliseconds",
  "model": "Random Forest"
}


