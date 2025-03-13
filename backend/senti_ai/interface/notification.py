import pushover, requests
from senti_ai.params import *

def send_pushover_notification(
    message,
    title=None,
    priority=1,
    user_key=AHMET_USER_KEY,
    api_token=PUSHOVER_API_TOKEN
    ):
    payload = {
        "token": api_token,
        "user": user_key,
        "message": message,
        "priority": priority
    }
    if title:
        payload["title"] = title

    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data=payload
    )

    return response.json()
