# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import time
from datetime import datetime
from datetime import timezone
from functools import wraps
from getpass import getpass

import requests
from requests.exceptions import HTTPError

from anemoi.utils.config import load_config
from anemoi.utils.config import save_config
from anemoi.utils.timer import Timer

REFRESH_EXPIRE_DAYS = 29


class TokenAuth:
    """Manage authentication with a keycloak token server."""

    def __init__(
        self,
        url: str,
        enabled: bool = True,
    ) -> None:
        """Initialise the token authentication object.

        Parameters
        ----------
        url : str
            URL of the authentication server.
        enabled : bool, optional
            Set this to False to turn off authentication, by default True

        """
        self.url = url
        self._enabled = enabled

        self.config_file = "mlflow-token.json"
        config = load_config(self.config_file)

        self._refresh_token = config.get("refresh_token")
        self.refresh_expires = config.get("refresh_expires", 0)
        self.access_token = None
        self.access_expires = 0

        # the command line tool adds a default handler to the root logger on runtime,
        # so we init our logger here (on runtime, not on import) to avoid duplicate handlers
        self.log = logging.getLogger(__name__)

    def __call__(self) -> None:
        self.authenticate()

    @property
    def refresh_token(self) -> str:
        return self._refresh_token

    @refresh_token.setter
    def refresh_token(self, value: str) -> None:
        self._refresh_token = value
        self.refresh_expires = time.time() + (REFRESH_EXPIRE_DAYS * 86400)  # 86400 seconds in a day

    def enabled(fn):  # noqa: ANN201, N805
        """Decorator to call or ignore a function based on the `enabled` flag."""

        @wraps(fn)
        def _wrapper(self, *args: list, **kwargs: dict):  # noqa: ANN001, ANN202
            if self._enabled:
                return fn(self, *args, **kwargs)
            return None

        return _wrapper

    @enabled
    def login(self, force_credentials: bool = False, **kwargs: dict) -> None:
        """Acquire a new refresh token and save it to disk.

        If an existing valid refresh token is already on disk it will be used.
        If not, or the token has expired, the user will be prompted for credentials.

        This function should be called once, interactively, right before starting a training run.

        Parameters
        ----------
        force_credentials : bool, optional
            Force a username/password prompt even if a refreh token is available, by default False.
        kwargs : dict
            Additional keyword arguments.

        Raises
        ------
        RuntimeError
            A new refresh token could not be acquired.

        """
        del kwargs  # unused
        self.log.info("🌐 Logging in to %s", self.url)
        new_refresh_token = None

        if not force_credentials and self.refresh_token and self.refresh_expires > time.time():
            new_refresh_token = self._token_request(ignore_exc=True).get("refresh_token")

        if not new_refresh_token:
            self.log.info("📝 Please sign in with your credentials.")
            username = input("Username: ")
            password = getpass("Password: ")

            new_refresh_token = self._token_request(username=username, password=password).get("refresh_token")

        if not new_refresh_token:
            msg = "❌ Failed to log in. Please try again."
            raise RuntimeError(msg)

        self.refresh_token = new_refresh_token
        self.save()

        self.log.info("✅ Successfully logged in to MLflow. Happy logging!")

    @enabled
    def authenticate(self, **kwargs: dict) -> None:
        """Check the access token and refresh it if necessary.

        The access token is stored in memory and in the environment variable `MLFLOW_TRACKING_TOKEN`.
        If the access token is still valid, this function does nothing.

        This function should be called before every MLflow API request.

        Raises
        ------
        RuntimeError
            No refresh token is available or the token request failed.

        """
        del kwargs  # unused
        if self.access_expires > time.time():
            return

        if not self.refresh_token or self.refresh_expires < time.time():
            msg = "You are not logged in to MLflow. Please log in first."
            raise RuntimeError(msg)

        with Timer("Access token refreshed", self.log):
            response = self._token_request()

        self.access_token = response.get("access_token")
        self.access_expires = time.time() + (response.get("expires_in") * 0.7)  # bit of buffer
        self.refresh_token = response.get("refresh_token")

        os.environ["MLFLOW_TRACKING_TOKEN"] = self.access_token

    @enabled
    def save(self, **kwargs: dict) -> None:
        """Save the latest refresh token to disk."""
        del kwargs  # unused
        if not self.refresh_token:
            self.log.warning("No refresh token to save.")
            return

        config = {
            "refresh_token": self.refresh_token,
            "refresh_expires": self.refresh_expires,
        }
        save_config(self.config_file, config)

        expire_date = datetime.fromtimestamp(self.refresh_expires, tz=timezone.utc)
        self.log.info("Your MLflow login token is valid until %s UTC", expire_date.strftime("%Y-%m-%d %H:%M:%S"))

    def _token_request(self, username=None, password=None, ignore_exc=False) -> dict:  # noqa: ANN001
        if username is not None and password is not None:
            path = "newtoken"
            payload = {"username": username, "password": password}
        else:
            path = "refreshtoken"
            payload = {"refresh_token": self.refresh_token}

        try:
            response = self._request(path, payload)
        except Exception:
            if ignore_exc:
                return {}
            raise

        return response

    def _request(self, path: str, payload: dict) -> dict:

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            response = requests.post(f"{self.url}/{path}", headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            response_json = response.json()

            if response_json.get("status", "") != "OK":
                error_description = response_json.get("response", "Error acquiring token.")
                msg = f"❌ {error_description}"
                raise RuntimeError(msg)

            return response_json["response"]
        except HTTPError:
            self.log.exception("HTTP error occurred")
            raise
