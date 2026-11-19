"""Bounded OpenML dataset discovery, adapted from PTHTTM's public_datasets."""

import csv
import hashlib
from io import StringIO
import json
import re
import time
from urllib.parse import urljoin, urlsplit

import pandas as pd
import requests
from scipy.io import arff


BASE_URL = "https://www.openml.org"
API = "/api/v1/json"
MAX_BYTES = 25 * 1024 * 1024
MAX_ROWS = 5000
MAX_FEATURES = 50
OPENML_HOSTS = {"www.openml.org", "openml.org", "api.openml.org", "data.openml.org"}


def _get_once(session, path, *, download=False, no_matches=False):
    url = BASE_URL + path
    for _ in range(4):
        with session.get(url, timeout=(5, 15), allow_redirects=False, stream=True) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers["Location"])
                location = urlsplit(url)
                if (location.scheme != "https" or location.netloc not in OPENML_HOSTS):
                    raise ValueError("OpenML redirected outside its permitted HTTPS hosts.")
                continue
            # OpenML can return 412 when a dataset list has no results.
            if no_matches and response.status_code in {404, 412}:
                return {}
            response.raise_for_status()
            content = bytearray()
            limit = MAX_BYTES if download else 2 * 1024 * 1024
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > limit:
                    raise ValueError("OpenML response exceeds the download limit.")
            return bytes(content)
    raise ValueError("Too many OpenML redirects.")


def _get(session, path, *, download=False, no_matches=False):
    """Retry only transient OpenML failures; keep redirects and downloads bounded."""
    for attempt in range(2):
        try:
            return _get_once(session, path, download=download, no_matches=no_matches)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in {429, 500, 502, 503, 504} or attempt == 1:
                raise
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 1:
                raise
        time.sleep(1 + attempt)


def _list(value):
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, dict) else []


def _json_object(content):
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("OpenML returned invalid JSON metadata.")
    return value


def candidates(session, search_terms):
    """Yield at most five unique records from dataset-name searches."""
    seen = set()
    errors = []
    for term in search_terms:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", term).strip("_")
        if not slug:
            continue
        try:
            payload = _get(session, f"{API}/data/list/data_name/{slug}/limit/10",
                           no_matches=True)
        except requests.RequestException as exc:
            errors.append(f"{slug}: {str(exc)[:250]}")
            continue
        listing = _json_object(payload).get("data") if payload else None
        records = listing.get("dataset") if isinstance(listing, dict) else None
        for record in _list(records):
            if not isinstance(record, dict):
                continue
            dataset_id = str(record.get("did", ""))
            if not re.fullmatch(r"[0-9]{1,12}", dataset_id) or dataset_id in seen:
                continue
            seen.add(dataset_id)
            yield dataset_id, record
            if len(seen) == 5:
                return
    if errors and not seen:
        raise RuntimeError("OpenML searches unavailable: " + "; ".join(errors[:2]))


_ATTRIBUTE = re.compile(r"""^@attribute\s+(?:'([^']+)'|"([^"]+)"|(\S+))\s+(.+)$""", re.I)


def _parse_dense_string_arff(text):
    """Read bounded dense ARFF with string fields when SciPy cannot parse them."""
    attributes = []
    data_lines = []
    in_data = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if not in_data:
            if stripped.casefold() == "@data":
                in_data = True
                continue
            match = _ATTRIBUTE.match(stripped)
            if match:
                name = next(value for value in match.groups()[:3] if value is not None)
                kind = match.group(4).strip().casefold()
                if kind not in {"string", "numeric", "real", "integer"} and not (
                        kind.startswith("{") and kind.endswith("}")):
                    raise ValueError("Unsupported ARFF attribute type.")
                attributes.append((name, kind))
        else:
            if stripped.startswith("{"):
                raise ValueError("Sparse ARFF is not supported.")
            data_lines.append(line)
    if not in_data or not attributes or not data_lines:
        raise ValueError("OpenML returned incomplete ARFF data.")
    for quote in ("'", '"'):
        rows = list(csv.reader(StringIO("".join(data_lines)), quotechar=quote,
                               escapechar="\\"))
        if rows and all(len(row) == len(attributes) for row in rows):
            break
    else:
        raise ValueError("OpenML returned malformed dense ARFF rows.")
    result = {}
    for index, (name, kind) in enumerate(attributes):
        values = [row[index] for row in rows]
        if kind in {"numeric", "real", "integer"}:
            result[name] = pd.to_numeric(
                pd.Series([None if value == "?" else value for value in values]),
                errors="raise",
            )
        else:
            result[name] = [None if value == "?" else value for value in values]
    return pd.DataFrame(result)


def load_candidate(session, dataset_id, record, *, max_rows=MAX_ROWS):
    metadata = _json_object(_get(session, f"{API}/data/{dataset_id}")).get("data_set_description")
    if (not isinstance(metadata, dict) or metadata.get("status") != "active"
            or str(metadata.get("visibility", "")).casefold() not in {"public", "everyone"}):
        raise ValueError("Dataset is not active and public.")
    if str(metadata.get("format", "")).casefold() != "arff":
        raise ValueError("Only dense ARFF datasets are supported.")
    target = str(metadata.get("default_target_attribute") or "").strip()
    file_id = str(metadata.get("file_id") or "")
    if not target or "," in target or not re.fullmatch(r"[0-9]{1,12}", file_id):
        raise ValueError("Dataset needs a single default target and a valid file ID.")
    quality = {item.get("name"): item.get("value") for item in _list(record.get("quality"))
               if isinstance(item, dict)}
    for name, minimum, maximum in (("NumberOfInstances", 30, max_rows),
                                    ("NumberOfFeatures", 2, MAX_FEATURES + 1)):
        if name in quality and not minimum <= float(quality[name]) <= maximum:
            raise ValueError("Dataset size is outside the supported benchmark limits.")

    raw = _get(session, f"/data/v1/download/{file_id}", download=True)
    try:
        records, _ = arff.loadarff(StringIO(raw.decode("utf-8-sig")))
        frame = pd.DataFrame.from_records(records)
    except NotImplementedError:
        frame = _parse_dense_string_arff(raw.decode("utf-8-sig"))
    except (arff.ArffError, UnicodeDecodeError) as exc:
        raise ValueError("OpenML returned unreadable ARFF data.") from exc
    if not 30 <= len(frame) <= max_rows or not 2 <= len(frame.columns) <= MAX_FEATURES + 1:
        raise ValueError("Downloaded dataset size is outside the supported benchmark limits.")
    if "NumberOfInstances" in quality and len(frame) != float(quality["NumberOfInstances"]):
        raise ValueError("Downloaded row count does not match OpenML metadata.")
    if target not in frame:
        raise ValueError("The default target is absent from the downloaded dataset.")
    for column in frame.columns:
        if not (pd.api.types.is_object_dtype(frame[column])
                or pd.api.types.is_string_dtype(frame[column])):
            continue
        frame[column] = frame[column].map(
            lambda value: None if value == b"?" else value.decode("utf-8")
            if isinstance(value, bytes) else value
        )

    excluded = []
    for field in ("ignore_attribute", "row_id_attribute"):
        value = metadata.get(field) or []
        excluded.extend(value.split(",") if isinstance(value, str) else _list(value))
    excluded = sorted({str(column).strip() for column in excluded} & set(frame.columns))
    if target in excluded:
        raise ValueError("OpenML marks the target as an excluded column.")
    frame = frame.drop(columns=excluded)
    return frame, {
        "provider": "OpenML",
        "id": dataset_id,
        "name": str(metadata.get("name") or record.get("name") or dataset_id)[:200],
        "version": str(metadata.get("version") or ""),
        "url": f"https://www.openml.org/d/{dataset_id}",
        "license": str(metadata.get("licence") or ""),
        "target": target,
        "description": str(metadata.get("description") or "")[:1800],
        "original_sha256": hashlib.sha256(raw).hexdigest(),
        "excluded_columns": excluded,
    }
