import pytest
from fastapi.testclient import TestClient
from src.backend import app

client = TestClient(app)

def test_status_endpoint():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

def test_hand_requirements_endpoint():
    response = client.get("/hand_requirements")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

def test_collect_stats_endpoint():
    response = client.get("/collect_stats")
    assert response.status_code == 200
    data = response.json()
    assert "alphabet" in data

def test_predict_endpoint_zero():
    payload = {"landmarks": [0.0] * 126, "mode": "alphabet"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "---"
