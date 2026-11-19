"""Bounded discovery of importable, labeled UCI datasets."""

import hashlib
from io import BytesIO
import json
import re
from urllib.parse import urlsplit

import pandas as pd
import requests

from ai_scientist.openml_data import MAX_BYTES, MAX_FEATURES, MAX_ROWS


BASE_URL = "https://archive.ics.uci.edu"
_HOST = "archive.ics.uci.edu"
_SKIP = {"data", "dataset", "machine", "learning", "classification", "regression",
         "detection", "prediction", "model", "models", "using", "features"}


def _get(session, path, *, params=None, limit=2 * 1024 * 1024):
    response = session.get(BASE_URL + path, params=params, timeout=(5, 15),
                           allow_redirects=False, stream=True)
    with response:
        if response.is_redirect:
            raise ValueError("UCI redirected a dataset request.")
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content) > limit:
                raise ValueError("UCI response exceeds the download limit.")
        return bytes(content)


def _search_queries(search_terms):
    terms = [term.strip()[:80] for term in search_terms if isinstance(term, str)
             and term.strip()][:3]
    tokens = []
    for term in terms:
        for token in re.findall(r"[a-zA-Z]{5,}", term.casefold()):
            if token not in _SKIP and token not in tokens:
                tokens.append(token)
    return list(dict.fromkeys(terms + tokens))[:6]


def candidates(session, search_terms):
    """Yield at most five UCI IDs from bounded name searches; continue after a miss."""
    seen, errors = set(), []
    for query in _search_queries(search_terms):
        try:
            payload = _get(session, "/api/datasets/list",
                           params={"search": query, "filter": "python"})
            response = json.loads(payload)
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{query}: {str(exc)[:200]}")
            continue
        if response.get("status") != 200:
            errors.append(f"{query}: UCI search returned {response.get('status')}")
            continue
        for record in response.get("data") or []:
            if not isinstance(record, dict):
                continue
            dataset_id = str(record.get("id", ""))
            if not re.fullmatch(r"[0-9]{1,12}", dataset_id) or dataset_id in seen:
                continue
            seen.add(dataset_id)
            yield dataset_id, record
            if len(seen) == 5:
                return
    if errors and not seen:
        raise RuntimeError("UCI searches unavailable: " + "; ".join(errors[:2]))


def load_candidate(session, dataset_id, record, *, max_rows=MAX_ROWS):
    """Use only the standardized CSV for the exact public UCI dataset ID."""
    if not re.fullmatch(r"[0-9]{1,12}", str(dataset_id)):
        raise ValueError("Invalid UCI dataset ID.")
    response = json.loads(_get(session, "/api/dataset", params={"id": dataset_id}))
    info = response.get("data")
    if response.get("status") != 200 or not isinstance(info, dict):
        raise ValueError("UCI did not return dataset metadata.")
    if str(info.get("uci_id")) != str(dataset_id):
        raise ValueError("UCI metadata ID does not match the requested dataset.")
    target = info.get("target_col")
    if not isinstance(target, list) or len(target) != 1 or not isinstance(target[0], str):
        raise ValueError("UCI dataset needs one named target column.")
    target = target[0]
    try:
        rows, features = int(info["num_instances"]), int(info["num_features"])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("UCI dataset lacks validated size metadata.") from exc
    if not (30 <= rows <= max_rows and 1 <= features <= MAX_FEATURES):
        raise ValueError("UCI dataset size is outside the supported benchmark limits.")
    url = str(info.get("data_url") or "")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != _HOST
            or parsed.path != f"/static/public/{dataset_id}/data.csv"
            or parsed.query or parsed.fragment):
        raise ValueError("UCI metadata does not point to its permitted standardized CSV.")
    raw = _get(session, parsed.path, limit=MAX_BYTES)
    try:
        frame = pd.read_csv(BytesIO(raw), nrows=max_rows + 1)
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError("UCI returned unreadable CSV data.") from exc
    if len(frame) != rows or target not in frame or len(frame.columns) > MAX_FEATURES + 1:
        raise ValueError("UCI CSV does not match its row, feature, or target metadata.")
    variables = info.get("variables") or []
    if not isinstance(variables, list):
        raise ValueError("UCI variables metadata is invalid.")
    roles = {item.get("name"): item.get("role") for item in variables
             if isinstance(item, dict)}
    allowed = [name for name in frame if name == target or roles.get(name) == "Feature"]
    if (roles.get(target) != "Target" or len(allowed) < 2
            or any(name not in roles for name in frame)):
        raise ValueError("UCI CSV columns do not match declared feature and target roles.")
    excluded = [name for name in frame if name not in allowed]
    frame = frame[allowed].copy()
    summary = info.get("additional_info") or {}
    description = str(summary.get("summary") or "")[:1800] if isinstance(summary, dict) else ""
    doi = str(info.get("dataset_doi") or "")
    return frame, {
        "provider": "UCI Machine Learning Repository", "id": str(dataset_id),
        "name": str(info.get("name") or record.get("name") or dataset_id)[:200],
        "version": "", "target": target, "target_role": "Target",
        "url": f"https://doi.org/{doi}" if re.fullmatch(r"10\.[0-9]{4,9}/[-._;()/:A-Za-z0-9]+", doi)
        else f"{BASE_URL}/dataset/{dataset_id}",
        "description": description,
        "declared_features": [name for name in allowed if name != target],
        "original_sha256": hashlib.sha256(raw).hexdigest(),
        "excluded_columns": excluded,
    }
