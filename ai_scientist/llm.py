import os
import re
import json
import time
from google import genai
from google.genai import types


RETRYABLE_CODES = {429, 500, 502, 503, 504}


class EmptyResponseError(Exception):
    pass


class GeminiModel:
    def __init__(self, clients, name, active_index=0, fallback_models=None):
        self.clients = clients
        self.model_names = [name] + [item for item in (fallback_models or []) if item != name]
        self.model_index = 0
        self.active_index = active_index

    @property
    def name(self):
        return self.model_names[self.model_index]

    @property
    def key_count(self):
        return len(self.clients)

    def advance_key(self):
        self.active_index = (self.active_index + 1) % self.key_count
        self.model_index = 0

    def advance_model(self):
        if self.model_index + 1 >= len(self.model_names):
            return False
        self.model_index += 1
        return True

    def generate_content(self, prompt, temperature=0.7):
        return self.clients[self.active_index].models.generate_content(
            model=self.name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )


def _configured_keys(api_key):
    if api_key:
        return [api_key.strip()]
    try:
        from google.colab import userdata
        primary = userdata.get("GEMINI_API_KEY")
    except Exception:
        primary = None
    primary = primary or os.environ.get("GEMINI_API_KEY", "")
    if not primary.strip():
        raise ValueError("Thiếu GEMINI_API_KEY. Hãy đặt biến môi trường trước khi chạy.")
    backups = [(int(match.group(1)), value) for name, value in os.environ.items()
               if (match := re.fullmatch(r"GEMINI_API_KEY_(\d+)", name))
               and int(match.group(1)) >= 2]
    ordered = [primary, os.environ.get("GEMINI_API_KEY_BACKUP", "")]
    ordered.extend(value for _, value in sorted(backups))
    return list(dict.fromkeys(key.strip() for key in ordered if key and key.strip()))


def _error_code(exc):
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    match = re.search(r"(?<!\d)(429|500|502|503|504|401|403|404)(?!\d)", str(exc))
    return int(match.group(1)) if match else None


def _select_model(client, preferred):
    names = []
    if preferred:
        names.append(preferred)
    try:
        models = list(client.models.list())
        priority = ("models/gemini-3.5-flash-lite", "models/gemini-3.5-flash",
                    "models/gemini-3.1-flash-lite", "models/gemini-2.5-flash-lite")
        models.sort(key=lambda model: (
            -1 if model.name in {preferred, f"models/{preferred}"} else
            priority.index(model.name) if model.name in priority else len(priority),
            model.name or "",
        ))
        names.extend(model.name for model in models
                     if "generateContent" in (model.supported_actions or [])
                     and (model.name or "").startswith("models/gemini-")
                     and "flash" in model.name.lower()
                     and not any(suffix in model.name.lower() for suffix in
                                 ("image", "tts", "omni", "preview", "latest")))
    except Exception as exc:
        print(f"Gemini model list unavailable (status {_error_code(exc) or 'unknown'}).")
    candidates = list(dict.fromkeys(names))[:3]
    for index, name in enumerate(candidates):
        try:
            probe = client.models.generate_content(model=name, contents="hi")
            if probe.text:
                return name, candidates[index + 1:]
        except Exception as exc:
            if _error_code(exc) in {401, 403}:
                break
    return None, []

def create_client(api_key=None):
    keys = _configured_keys(api_key)
    # The SDK retries transient errors by default. Keep retries at this layer so
    # each key gets a bounded, visible number of attempts before failover.
    options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
    clients = [genai.Client(api_key=key, http_options=options) for key in keys]
    preferred = os.environ.get("GEMINI_MODEL", "").strip()
    for index, client in enumerate(clients):
        name, fallback_models = _select_model(client, preferred)
        if name:
            print(f"Active Gemini model: {name} (key {index + 1}/{len(clients)})")
            return GeminiModel(clients, name, index, fallback_models), name
        if index + 1 < len(clients):
            print(f"Gemini key {index + 1} could not call a model; trying a backup key.")
    raise ValueError("Không tìm thấy mô hình Gemini nào hoạt động với các API key đã cấu hình.")

def get_response_from_llm(prompt, model, system_message="", temperature=0.7, max_retries=5):
    full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
    key_count = max(1, getattr(model, "key_count", 1))
    attempts_per_key = min(max_retries, 2) if key_count > 1 else max_retries
    for key_index in range(key_count):
        while True:
            rejected_key = False
            for attempt in range(1, attempts_per_key + 1):
                try:
                    response = model.generate_content(full_prompt, temperature=temperature)
                    if not response.text:
                        raise EmptyResponseError("Gemini trả về nội dung rỗng.")
                    time.sleep(2)
                    return response.text, []
                except Exception as exc:
                    code = _error_code(exc)
                    active = getattr(model, "active_index", key_index) + 1
                    print(f"Gemini (key {active}/{key_count}, model {model.name}, "
                          f"attempt {attempt}/{attempts_per_key}): status {code or 'unknown'}.")
                    if code == 401:
                        print(f"Gemini key {active} was rejected (401). Verify or replace this API key in Google AI Studio.")
                    elif code == 403:
                        print(f"Gemini key {active} lacks permission for this request (403).")
                    if code in {401, 403}:
                        rejected_key = True
                        break
                    if code == 404:
                        break
                    if code not in RETRYABLE_CODES and not isinstance(exc, EmptyResponseError):
                        raise RuntimeError("Gemini trả về lỗi không thể thử lại.") from None
                    if attempt < attempts_per_key:
                        wait = min(24, (6 if code == 429 else 2) * 2 ** (attempt - 1))
                        print(f"Waiting {wait}s before retry.")
                        time.sleep(wait)
            if rejected_key or not hasattr(model, "advance_model") or not model.advance_model():
                break
            print(f"Switching to another Gemini model: {model.name}.")
        if key_index + 1 < key_count:
            print("Switching to a backup Gemini API key.")
            model.advance_key()
    raise RuntimeError("Đã thử hết các API key Gemini nhưng dịch vụ vẫn chưa phản hồi.")

def extract_json_between_markers(llm_output):
    decoder = json.JSONDecoder()
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
    for candidate in fenced + [llm_output]:
        start = candidate.find("{")
        if start < 0:
            continue
        try:
            value, _ = decoder.raw_decode(candidate[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None
