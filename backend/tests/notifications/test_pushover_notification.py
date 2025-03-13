import requests
import requests_mock
import pytest
from urllib.parse import parse_qs
from senti_ai.interface.notification import send_pushover_notification
from senti_ai.params import *

def test_send_pushover_notification():
    with requests_mock.Mocker() as m:
        m.post("https://api.pushover.net/1/messages.json", json={"status": 1})

        response = send_pushover_notification("Test message", "Test title", priority=1)

        assert response["status"] == 1
        assert m.called
        assert m.call_count == 1

        # Properly parse form-encoded request data
        request_data = parse_qs(m.last_request.text)
        request_data = {k: v[0] for k, v in request_data.items()}  # Convert list values to strings

        assert request_data == {
            "token": PUSHOVER_API_TOKEN,
            "user": AHMET_USER_KEY,
            "message": "Test message",
            "priority": "1",  # Converted to string
            "title": "Test title"
        }
