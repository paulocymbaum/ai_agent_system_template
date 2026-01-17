import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze():
    response = client.post(
        "/api/v1/analyze",
        json={
            "query": "Test query",
            "max_iterations": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data
