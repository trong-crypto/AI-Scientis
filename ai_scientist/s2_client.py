"""Centralized, per-process pacing for every Semantic Scholar API request."""

import os
import threading
import time


BASE = "https://api.semanticscholar.org"
MIN_INTERVAL_SECONDS = 1.25  # Strictly slower than the 1 request/second key limit.
MAX_RETRY_DELAY_SECONDS = 30.0


class RequestPacer:
    def __init__(self, interval=MIN_INTERVAL_SECONDS, clock=None, sleeper=None):
        self.interval = interval
        self.clock = clock or time.monotonic
        self.sleep = sleeper or time.sleep
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self):
        with self.lock:
            delay = max(0.0, self.next_allowed - self.clock())
            if delay:
                self.sleep(delay)
            self.next_allowed = self.clock() + self.interval

    def defer(self, seconds):
        with self.lock:
            self.next_allowed = max(self.next_allowed, self.clock() + seconds)


PACER = RequestPacer()


def _retry_delay(response):
    try:
        supplied = float(response.headers.get("Retry-After", ""))
    except (AttributeError, TypeError, ValueError):
        supplied = 0.0
    return min(MAX_RETRY_DELAY_SECONDS, max(2.5, supplied))


def get_json(session, path, params, timeout=15):
    """Use one pacer for all S2 endpoints; never place the key in a URL or log."""
    if not path.startswith(("/graph/v1/", "/recommendations/v1/", "/datasets/v1/")):
        raise ValueError("Unsupported Semantic Scholar API path.")
    headers = {"User-Agent": "AI-Scientist-Review/1.0 (local evidence review)"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    for attempt in range(2):
        PACER.wait()
        response = session.get(BASE + path, params=params, headers=headers, timeout=timeout)
        if getattr(response, "status_code", None) == 429 and attempt == 0:
            PACER.defer(_retry_delay(response))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Semantic Scholar request did not complete.")
