# ML Model Serving Pipeline

A production-grade ML inference system with FastAPI, Docker, CI/CD, and Prometheus/Grafana monitoring.

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Client    │────▶│    FastAPI      │────▶│   Model     │
│  (Request)  │     │  + Prometheus   │     │  (PyTorch)  │
└─────────────┘     └────────┬────────┘     └─────────────┘
                             │
                             ▼ /metrics
                    ┌─────────────────┐
                    │   Prometheus    │
                    │   (Scraper)     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Grafana      │
                    │   (Dashboard)   │
                    └─────────────────┘
```

## Features

- **FastAPI** — High-performance async API for model inference
- **Docker** — Containerized deployment
- **CI/CD** — Automated testing and builds with GitHub Actions
- **Prometheus** — Metrics collection (request count, latency, predictions)
- **Grafana** — Pre-configured monitoring dashboard
- **Input Validation** — Pydantic schemas with proper error handling

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose

### Option 1: Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ml-serving-pipeline.git
cd ml-serving-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload
```

Visit **http://localhost:8000/docs** to test the API.

### Option 2: Run with Docker

```bash
# Build and run
docker build -t ml-serving-api .
docker run -p 8000:8000 ml-serving-api
```

### Option 3: Run Full Stack (API + Monitoring)

```bash
# Start all services
docker-compose up --build
```

This starts:

- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (login: admin/admin)

## API Endpoints

| Endpoint       | Method | Description         |
| -------------- | ------ | ------------------- |
| `/`            | GET    | Health check        |
| `/health`      | GET    | Health status       |
| `/api/predict` | POST   | Run model inference |
| `/metrics`     | GET    | Prometheus metrics  |
| `/docs`        | GET    | Swagger UI          |

## Usage

### Make a Prediction

```bash
# Using curl
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [0.0, 0.0, ... 784 floats ...]}'
```

```python
# Using Python
import requests

# 784 values for 28x28 MNIST image
data = [0.0] * 784
response = requests.post(
    "http://localhost:8000/api/predict",
    json={"data": data}
)
print(response.json())
# {"prediction": [0.1, -0.2, 0.5, ...]}  # 10 logits for digits 0-9
```

### Generate Test Traffic

```bash
python scripts/generate_traffic.py
```

## Development

### Run Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

### Run Linter

```bash
ruff check app/ tests/
ruff format app/ tests/
```

### Project Structure

```
ml-serving-pipeline/
├── app/
│   ├── api/
│   │   ├── routes.py          # API endpoints
│   │   └── schemas.py         # Pydantic models
│   ├── models/
│   │   ├── loader.py          # Model loading logic
│   │   └── model.py           # PyTorch model architecture
│   ├── monitoring/
│   │   └── metrics.py         # Prometheus metrics
│   └── main.py                # FastAPI app
├── tests/
│   └── test_api.py            # API tests
├── grafana/
│   ├── dashboards/            # Pre-configured dashboards
│   └── provisioning/          # Auto-setup configs
├── scripts/
│   └── generate_traffic.py    # Load testing script
├── .github/workflows/
│   └── ci_cd_pipeline.yml     # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
└── README.md
```

## Monitoring

### Metrics Collected

| Metric                           | Type      | Description              |
| -------------------------------- | --------- | ------------------------ |
| `ml_api_requests_total`          | Counter   | Total requests by status |
| `ml_api_request_latency_seconds` | Histogram | Request latency          |
| `ml_api_predictions_total`       | Counter   | Predictions by digit     |
| `ml_api_inference_time_seconds`  | Histogram | Model inference time     |

### Grafana Dashboard

The pre-configured dashboard includes:

- Request rate (per second)
- Latency percentiles (p50, p95, p99)
- Success rate gauge
- Prediction distribution by digit

## CI/CD Pipeline

On every push to `main`:

1. **Lint** — Code style check with ruff
2. **Test** — Run pytest suite
3. **Build** — Build and verify Docker image

## Tech Stack

- **Framework**: FastAPI
- **ML**: PyTorch
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **Monitoring**: Prometheus + Grafana
- **CI/CD**: GitHub Actions
- **Testing**: pytest
- **Linting**: ruff
