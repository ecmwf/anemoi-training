# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import os
import time
from getpass import getpass

import requests
from anemoi.utils.config import load_config
from anemoi.utils.config import save_config
from requests.exceptions import HTTPError

from anemoi.training.utils.logger import get_code_logger

AUTH = None


def get_auth(*args, **kwargs):
    """Return a global authentication instance, creating it if necessary.
    Assumes the args don't change between calls."""

    global AUTH
    if AUTH is None:
        AUTH = TokenAuth(*args, **kwargs)
    return AUTH


class TokenAuth:
    """Manage authentication with a keycloak token server."""

    def __init__(
        self,
        url,
        refresh_expire_days=29,
        enabled=True,
    ):
        """Parameters
        ----------
        url : str
            URL of the authentication server.
        refresh_expire_days : int, optional
            Number of days before the refresh token expires, by default 29
        enabled : bool, optional
            Set this to False to turn off authentication, by default True
        """

        assert url, "Please provide a URL for the authentication server."

        self.url = url
        self.refresh_expire_days = refresh_expire_days
        self.enabled = enabled

        self.config_file = "mlflow-token.json"
        config = load_config(self.config_file)

        self._refresh_token = config.get("refresh_token")
        self.refresh_expires = config.get("refresh_expires", 0)
        self.access_token = None
        self.access_expires = 0

        # the command line tool adds a default handler to the root logger on runtime,
        # so we init our logger here (on runtime, not on import) to avoid duplicate handlers
        self.log = get_code_logger(__name__)

    def __call__(self):
        self.authenticate()

    @property
    def refresh_token(self):
        return self._refresh_token

    @refresh_token.setter
    def refresh_token(self, value):
        self._refresh_token = value
        self.refresh_expires = time.time() + (self.refresh_expire_days * 24 * 60 * 60)

    def login(self, force_credentials=False, **kwargs):
        """Acquire a new refresh token and save it to disk.

        If an existing valid refresh token is already on disk it will be used.
        If not, or the token has expired, the user will be prompted for credentials.

        This function should be called once, interactively, right before starting a training run.

        Parameters
        ----------
        force_credentials : bool, optional
            Force a username/password prompt even if a refreh token is available, by default False.

        Raises
        ------
        RuntimeError
            A new refresh token could not be acquired.
        """

        if not self.enabled:
            return

        self.log.info(f"Logging in to {self.url}")
        new_refresh_token = None

        if not force_credentials and self.refresh_token and self.refresh_expires > time.time():
            new_refresh_token = self._token_request(ignore_exc=True).get("refresh_token")

        if not new_refresh_token:
            self.log.info("Please sign in with your credentials.")
            username = input("Username: ")
            password = getpass("Password: ")

            new_refresh_token = self._token_request(username=username, password=password).get("refresh_token")

        if not new_refresh_token:
            raise RuntimeError("Failed to log in. Please try again.")

        self.refresh_token = new_refresh_token
        self.save()

        self.log.info("Successfully logged in to MLflow. Happy logging!")

    def authenticate(self, **kwargs):
        """Check the access token and refresh it if necessary.

        The access token is stored in memory and in the environment variable `MLFLOW_TRACKING_TOKEN`.
        If the access token is still valid, this function does nothing.

        This function should be called before every MLflow API request.

        Raises
        ------
        RuntimeError
            No refresh token is available or the token request failed.
        """

        if not self.enabled:
            return

        if self.access_expires > time.time():
            return

        if not self.refresh_token or self.refresh_expires < time.time():
            raise RuntimeError("You are not logged in to MLflow. Please log in first.")

        response = self._token_request()

        self.access_token = response.get("access_token")
        self.access_expires = time.time() + (response.get("expires_in") * 0.7)  # bit of buffer
        self.refresh_token = response.get("refresh_token")

        os.environ["MLFLOW_TRACKING_TOKEN"] = self.access_token

        self.log.debug("Access token refreshed.")

    def save(self, **kwargs):
        """Save the latest refresh token to disk."""

        if not self.refresh_token:
            self.log.warning("No refresh token to save.")
            return

        config = {
            "refresh_token": self.refresh_token,
            "refresh_expires": self.refresh_expires,
        }
        save_config(self.config_file, config)

    def _token_request(self, username=None, password=None, ignore_exc=False):
        if username is not None and password is not None:
            path = "newtoken"
            payload = {"username": username, "password": password}
        else:
            path = "refreshtoken"
            payload = {"refresh_token": self.refresh_token}

        try:
            response = self._request(path, payload)
        except Exception as err:
            if ignore_exc:
                return {}
            raise err

        return response

    def _request(self, path, payload):

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            response = requests.post(f"{self.url}/{path}", headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()

            if response_json.get("status", "") != "OK":
                # TODO: there's a bug in the API that returns the error response as a string instead of a json object.
                # Remove this when the API is fixed.
                if isinstance(response_json["response"], str):
                    error = json.loads(response_json["response"])
                else:
                    error = response_json["response"]
                error_description = error.get("error_description", "Error acquiring token.")
                raise RuntimeError(error_description)

            return response_json["response"]
        except HTTPError as http_err:
            self.log.error(f"HTTP error occurred: {http_err}")
            raise
