import json
import logging
import os
import time
from getpass import getpass

import requests
from requests.exceptions import HTTPError

LOG = logging.getLogger(__name__)

REFRESH_TOKEN_EXPIRE = 29  # days
CONFIG_PATH = os.path.expanduser("~/.config/anemoi")
CONFIG_FILE = os.path.join(CONFIG_PATH, "mlflow-auth.json")


def get_config():
    try:
        with open(CONFIG_FILE, "r") as file:
            config = json.load(file)
        return config
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config(refresh_token):
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(CONFIG_PATH)

    refresh_expires = time.time() + (REFRESH_TOKEN_EXPIRE * 24 * 60 * 60)
    config = {
        "refresh_token": refresh_token,
        "refresh_expires": int(refresh_expires),
    }

    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file)


class TokenAuthenticator:
    def __init__(self, uri: str = "https://mlflow-test.ecmwf.int", enabled=True):
        self.uri = uri
        self.enabled = enabled
        self.refresh_token = None
        self.auth_token = None
        self.auth_expires = 0

    def login(self):

        config = get_config()

        refresh_token = config.get("refresh_token")
        refresh_expires = config.get("refresh_expires", 0)

        if refresh_token and refresh_expires >= time.time():
            new_refresh_token = self._token_login(refresh_token)
        else:
            username = input("Username: ")
            password = getpass("Password: ")
            new_refresh_token = self._credential_login(username, password)

        if new_refresh_token:
            self.refresh_token = new_refresh_token
            save_config(new_refresh_token)
            LOG.info("Successfully authenticated with MLflow. Happy logging!")
        else:
            raise ValueError("No refresh token received.")

    def _credential_login(self, username: str, password: str):
        payload = {"username": username, "password": password}
        response = self._request("newtoken", payload)

        return response.get("refresh_token")

    def _token_login(self, refresh_token: str):
        payload = {"refresh_token": refresh_token}
        response = self._request("refreshtoken", payload)

        return response.get("refresh_token")

    def _request(self, path: str, payload: dict):

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            response = requests.post(f"{self.uri}/{path}", headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()

            return response_json["response"]
        except HTTPError as http_err:
            LOG.error(f"HTTP error occurred: {http_err}")
            raise
        except Exception as err:
            LOG.error(f"Other error occurred: {err}")
            raise
