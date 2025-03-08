from utils import get_token


def test_get_token():
    assert get_token("DASHBOARD_PORT") == "8000"
    assert get_token("DASHBOARD_HOST") == "0.0.0.0"
