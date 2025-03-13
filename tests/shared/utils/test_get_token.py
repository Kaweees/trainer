from utils import get_token


def test_get_token(monkeypatch):
    # Set up mock environment variables for the test
    monkeypatch.setenv("DASHBOARD_PORT", "8000")
    monkeypatch.setenv("DASHBOARD_HOST", "0.0.0.0")

    # Run assertions with mocked environment
    assert get_token("DASHBOARD_PORT") == "8000"
    assert get_token("DASHBOARD_HOST") == "0.0.0.0"
