import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthCheck:
    """Tests for basic API health."""

    def test_root_returns_ok(self, client):
        """Root endpoint should return status ok."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestPredictEndpoint:
    """Tests for /api/predict endpoint."""

    def test_predict_valid_input(self, client):
        """Valid 784-element input should return 10 predictions."""
        payload = {"data": [0.0] * 784}
        response = client.post("/api/predict", json=payload)

        assert response.status_code == 200
        assert "prediction" in response.json()
        assert len(response.json()["prediction"]) == 10

    def test_predict_invalid_input_wrong_size(self, client):
        """Input with wrong size should fail gracefully."""
        payload = {"data": [0.0] * 100}  # Wrong size
        response = client.post("/api/predict", json=payload)

        # Should return 500 (model reshape fails) or 422 (validation)
        assert response.status_code in [422, 500]

    def test_predict_empty_input(self, client):
        """Empty input should fail."""
        payload = {"data": []}
        response = client.post("/api/predict", json=payload)

        assert response.status_code in [422, 500]

    def test_predict_missing_data_field(self, client):
        """Missing 'data' field should return 422."""
        payload = {}
        response = client.post("/api/predict", json=payload)

        assert response.status_code == 422

    def test_predict_returns_floats(self, client):
        """Predictions should be floats."""
        payload = {"data": [0.0] * 784}
        response = client.post("/api/predict", json=payload)

        predictions = response.json()["prediction"]
        assert all(isinstance(p, float) for p in predictions)
