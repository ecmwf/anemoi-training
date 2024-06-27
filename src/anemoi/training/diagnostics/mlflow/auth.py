import json
import logging
import os
import time
from getpass import getpass

import requests
from anemoi.utils.config import load_config
from anemoi.utils.config import save_config
from requests.exceptions import HTTPError

LOG = logging.getLogger(__name__)


class TokenAuth:
    def __init__(
        self,
        uri="https://mlflow-test.ecmwf.int",
        refresh_expire_days=29,
        enabled=True,
    ):
        self.uri = uri
        self.refresh_expire_days = refresh_expire_days
        self.enabled = enabled

        self.config_file = "mlflow-token.json"
        config = load_config(self.config_file)

        self.refresh_token = config.get("refresh_token")
        self.refresh_expires = config.get("refresh_expires", 0)
        self.access_token = None
        self.access_expires = 0

    def __call__(self):
        self.authenticate()

    def login(self, force_credentials=False, **kwargs):
        LOG.info(f"Logging in to {self.uri}")

        if not force_credentials and self.refresh_token and self.refresh_expires >= time.time():
            new_refresh_token = self._get_refresh_token(self.refresh_token)
        else:
            LOG.info("Please sign in with your credentials.")
            username = input("Username: ")
            password = getpass("Password: ")
            new_refresh_token = self._get_refresh_token(username=username, password=password)

        if new_refresh_token:
            self.refresh_token = new_refresh_token
            self._save_config(new_refresh_token)

            LOG.info("Successfully logged in to MLflow. Happy logging!")
        else:
            raise ValueError("No refresh token received.")

    def authenticate(self):
        if not self.enabled:
            return

        if self.access_expires > time.time():
            return

        if not self.refresh_token or self.refresh_expires < time.time():
            raise RuntimeError("You are not logged in to MLFlow. Please log in first.")

        self.access_token, self.access_expires = self._get_access_token()

        os.environ["MLFLOW_TRACKING_TOKEN"] = self.access_token
        LOG.debug("Access token refreshed.")

    def _save_config(self, refresh_token):
        refresh_expires = time.time() + (self.refresh_expire_days * 24 * 60 * 60)
        config = {
            "refresh_token": refresh_token,
            "refresh_expires": int(refresh_expires),
        }
        save_config(self.config_file, config)

    def _get_refresh_token(self, refresh_token=None, username=None, password=None):
        if refresh_token:
            path = "refreshtoken"
            payload = {"refresh_token": refresh_token}
        else:
            path = "newtoken"
            payload = {"username": username, "password": password}

        response = self._request(path, payload)

        return response.get("refresh_token")

    def _get_access_token(self):
        payload = {"refresh_token": self.refresh_token}
        response = self._request("refreshtoken", payload)

        token = response.get("access_token")
        expires_in = response.get("expires_in")

        expires = time.time() + (expires_in * 0.7)  # some buffer time

        return token, expires

    def _request(self, path, payload):

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            response = requests.post(f"{self.uri}/{path}", headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()

            if response_json.get("status", "") == "ERROR":
                # TODO: there's a bug in the API that returns the error response as a string instead of a json object.
                # Remove this when the API is fixed.
                if isinstance(response_json["response"], str):
                    error = json.loads(response_json["response"])
                else:
                    error = response_json["response"]
                LOG.warning(error.get("error_description", "Error acquiring token."))
                # don't raise here, let the caller decide what to do if no token is acquired
                return {}

            return response_json["response"]
        except HTTPError as http_err:
            LOG.error(f"HTTP error occurred: {http_err}")
            raise
        except Exception as err:
            LOG.error(f"Other error occurred: {err}")
            raise
