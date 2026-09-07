"""Single fal.ai gateway shared by every paid generation provider."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests


class FalGateway:
    name = "fal"

    def __init__(self, api_key: str, client=None):
        if not api_key and client is None:
            raise ValueError("FAL_KEY is required for paid generation")
        if client is None:
            from fal_client import SyncClient

            client = SyncClient(key=api_key, default_timeout=180)
        self.client = client

    # Transient failures worth retrying. Deliberately narrow: a rate limit
    # or a server hiccup means the request was REJECTED, not started, so
    # retrying cannot double-charge. Anything else — auth, validation, an
    # exhausted balance — is a real answer and must surface immediately
    # rather than being retried three times.
    RETRYABLE_MARKERS = (
        "429", "rate limit", "too many requests",
        "502", "503", "504", "bad gateway", "service unavailable",
        "gateway timeout", "connection reset", "connection aborted",
    )
    MAX_ATTEMPTS = 3
    RETRY_BACKOFF_SECONDS = 5.0

    def _subscribe_with_retry(
        self, endpoint: str, arguments: dict[str, Any], timeout: int
    ) -> Any:
        """One retry layer for every paid call in the project.

        Added when video concurrency was raised: with several clips in
        flight a rate limit becomes likely, and without this a single 429
        aborts the whole run — throwing away every image already paid for.
        """
        last_error: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                return self.client.subscribe(
                    endpoint.strip("/"),
                    arguments=arguments,
                    with_logs=False,
                    client_timeout=timeout,
                )
            except Exception as err:  # noqa: BLE001 - re-raised below if fatal
                message = str(err).lower()
                if not any(marker in message for marker in self.RETRYABLE_MARKERS):
                    raise
                last_error = err
                if attempt < self.MAX_ATTEMPTS - 1:
                    time.sleep(self.RETRY_BACKOFF_SECONDS * (2 ** attempt))
        raise RuntimeError(
            f"fal endpoint {endpoint!r} still failing after {self.MAX_ATTEMPTS} "
            f"attempts: {last_error}"
        )

    def run(self, endpoint: str, arguments: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
        result = self._subscribe_with_retry(endpoint, arguments, timeout)
        if not isinstance(result, dict):
            raise ValueError(f"fal endpoint {endpoint!r} returned a non-object response")
        if result.get("error"):
            raise RuntimeError(f"fal endpoint {endpoint!r} failed: {result['error']}")
        return result

    def download(self, url: str) -> bytes:
        if not url.startswith("https://"):
            raise ValueError("fal media URL must use HTTPS")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        return response.content

    def upload(self, path: Path) -> str:
        """Uploads a local file to fal's storage and returns its access URL —
        needed to feed a locally-generated image into an endpoint (like
        image-to-video) that only accepts a URL, not a local path."""
        return self.client.upload_file(path)


def media_url(result: dict[str, Any], *keys: str) -> str:
    """Resolve common fal media shapes such as audio.url or images[0].url."""
    value: Any = result
    for key in keys:
        if isinstance(value, list):
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value.get(key)
        else:
            value = None
        if value is None:
            break
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("url"), str):
        return value["url"]
    raise ValueError(f"fal response did not contain media at {'.'.join(keys)}")
