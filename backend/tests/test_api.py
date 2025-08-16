from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Chat With Docs API"
    assert response.json()["version"] == "1.0.0"


def test_health_check():
    """Test that the API is responding."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
