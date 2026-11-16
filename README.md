⚙️ Getting Started & Local Installation

Follow these steps to clone and set up the repository on your local machine:

1. Clone the Repository

git clone https://github.com/AhmadrezaChegeni/llm-inference-performance-mlzoomcamp.git
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

🐳 Docker Containerization & Deployment

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
  'http://127.0.0.1:8000/predict' \
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
