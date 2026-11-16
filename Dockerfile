FROM python:3.12-slim

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and trained model artifact
COPY src/ ./src/
COPY models/ ./models/

# Expose port 8000
EXPOSE 8000

# Start FastAPI (supports dynamic $PORT for cloud platforms like Render/Railway)
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]