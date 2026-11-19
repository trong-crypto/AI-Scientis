"""Gemini key failover without live API calls or exposing configured secrets."""

import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from ai_scientist.llm import GeminiModel, _configured_keys, create_client, get_response_from_llm


class FakeModels:
    def __init__(self, outcomes, probe_error=None, names=None):
        self.outcomes = list(outcomes)
        self.probe_error = probe_error
        self.names = names or ["models/gemini-test"]
        self.calls = []

    def list(self):
        return [SimpleNamespace(name=name, supported_actions=["generateContent"])
                for name in self.names]

    def generate_content(self, model, contents, config=None):
        self.calls.append((model, contents))
        if contents == "hi":
            error = (self.probe_error.get(model) if isinstance(self.probe_error, dict)
                     else self.probe_error)
            if error:
                raise error
            return SimpleNamespace(text="ready")
        result = self.outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(text=result)


class FakeClient:
    def __init__(self, outcomes, probe_error=None, names=None):
        self.models = FakeModels(outcomes, probe_error, names)


class LlmFailoverTests(unittest.TestCase):
    def test_keys_are_ordered_and_duplicates_removed(self):
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "primary", "GEMINI_API_KEY_BACKUP": "backup",
            "GEMINI_API_KEY_2": "backup", "GEMINI_API_KEY_3": "third",
        }, clear=True):
            self.assertEqual(_configured_keys(None), ["primary", "backup", "third"])

    def test_503_retries_then_uses_backup_and_keeps_it_active(self):
        primary = FakeClient([RuntimeError("503 UNAVAILABLE"), RuntimeError("503 UNAVAILABLE")])
        backup = FakeClient(["result", "next result"])
        model = GeminiModel([primary, backup], "models/gemini-test")
        with patch("ai_scientist.llm.time.sleep"):
            self.assertEqual(get_response_from_llm("prompt", model)[0], "result")
            self.assertEqual(get_response_from_llm("another", model)[0], "next result")
        self.assertEqual(model.active_index, 1)
        self.assertEqual(len(primary.models.calls), 2)
        self.assertEqual(len(backup.models.calls), 2)

    def test_all_keys_exhausted_without_printing_secrets(self):
        model = GeminiModel([
            FakeClient([RuntimeError("503 UNAVAILABLE secret-primary"),
                        RuntimeError("503 UNAVAILABLE secret-primary")]),
            FakeClient([RuntimeError("503 UNAVAILABLE secret-backup"),
                        RuntimeError("503 UNAVAILABLE secret-backup")]),
        ], "models/gemini-test")
        output = io.StringIO()
        with patch("ai_scientist.llm.time.sleep"), redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "hết các API key"):
                get_response_from_llm("prompt", model)
        self.assertNotIn("secret", output.getvalue().casefold())

    def test_invalid_primary_key_skips_to_backup(self):
        model = GeminiModel([FakeClient([RuntimeError("401 unauthorized")]),
                             FakeClient(["backup result"])], "models/gemini-test")
        output = io.StringIO()
        with patch("ai_scientist.llm.time.sleep"), redirect_stdout(output):
            self.assertEqual(get_response_from_llm("prompt", model)[0], "backup result")
        self.assertEqual(model.active_index, 1)
        self.assertIn("key 1 was rejected (401)", output.getvalue())

    def test_create_client_accepts_backup_without_printing_key(self):
        clients = [FakeClient(["response"]), FakeClient(["response"])]
        output = io.StringIO()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-primary",
                                  "GEMINI_API_KEY_BACKUP": "secret-backup",
                                  "GEMINI_MODEL": "models/gemini-test"}, clear=True), \
             patch("ai_scientist.llm.genai.Client", side_effect=clients) as factory, \
             redirect_stdout(output):
            model, name = create_client()
        self.assertEqual(name, "models/gemini-test")
        self.assertEqual(model.key_count, 2)
        self.assertEqual(factory.call_count, 2)
        self.assertNotIn("secret-primary", output.getvalue())
        self.assertNotIn("secret-backup", output.getvalue())

    def test_create_client_tries_backup_when_primary_probe_is_unavailable(self):
        clients = [FakeClient([], probe_error=RuntimeError("503 UNAVAILABLE")), FakeClient([])]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "primary",
                                  "GEMINI_API_KEY_BACKUP": "backup",
                                  "GEMINI_MODEL": "models/gemini-test"}, clear=True), \
             patch("ai_scientist.llm.genai.Client", side_effect=clients), \
             redirect_stdout(io.StringIO()):
            model, _ = create_client()
        self.assertEqual(model.active_index, 1)

    def test_503_switches_model_before_backup_key(self):
        primary = FakeClient([RuntimeError("503 UNAVAILABLE"),
                              RuntimeError("503 UNAVAILABLE"), "fallback result"])
        backup = FakeClient(["backup should not be used"])
        model = GeminiModel([primary, backup], "models/gemini-flash-a",
                            fallback_models=["models/gemini-flash-b"])
        with patch("ai_scientist.llm.time.sleep"), redirect_stdout(io.StringIO()):
            result, _ = get_response_from_llm("prompt", model)
        self.assertEqual(result, "fallback result")
        self.assertEqual(model.name, "models/gemini-flash-b")
        self.assertEqual(model.active_index, 0)
        self.assertEqual([call[0] for call in primary.models.calls],
                         ["models/gemini-flash-a", "models/gemini-flash-a",
                          "models/gemini-flash-b"])
        self.assertEqual(backup.models.calls, [])

    def test_401_switches_key_without_trying_other_models(self):
        primary = FakeClient([RuntimeError("401 unauthorized")])
        backup = FakeClient(["backup result"])
        model = GeminiModel([primary, backup], "models/gemini-flash-a",
                            fallback_models=["models/gemini-flash-b"])
        with patch("ai_scientist.llm.time.sleep"), redirect_stdout(io.StringIO()):
            result, _ = get_response_from_llm("prompt", model)
        self.assertEqual(result, "backup result")
        self.assertEqual(model.active_index, 1)
        self.assertEqual(model.name, "models/gemini-flash-a")
        self.assertEqual(len(primary.models.calls), 1)

    def test_startup_uses_another_text_model_when_first_is_overloaded(self):
        primary = FakeClient([], probe_error={
            "models/gemini-3.5-flash-lite": RuntimeError("503 UNAVAILABLE")},
            names=["models/gemini-3.5-flash-lite", "models/gemini-3.5-flash",
                   "models/gemini-3.5-flash-image"])
        with patch.dict(os.environ, {"GEMINI_API_KEY": "primary"}, clear=True), \
             patch("ai_scientist.llm.genai.Client", return_value=primary), \
             redirect_stdout(io.StringIO()):
            model, name = create_client()
        self.assertEqual(name, "models/gemini-3.5-flash")
        self.assertEqual(model.model_names, ["models/gemini-3.5-flash"])
        self.assertEqual([call[0] for call in primary.models.calls],
                         ["models/gemini-3.5-flash-lite", "models/gemini-3.5-flash"])


if __name__ == "__main__":
    unittest.main()
