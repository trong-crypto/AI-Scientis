import json
import shutil
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import unittest
from unittest.mock import patch

import pandas as pd
import requests
from pypdf import PdfReader

from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.openml_data import _parse_dense_string_arff, candidates
from ai_scientist.research_ideas import plan_and_run
from ai_scientist.tabular_research import _designs
from ai_scientist.tabular_benchmark import run_benchmark
from ai_scientist.text_benchmark import run_text_benchmark


PLAN = {"topic": "Iris species prediction", "Title": "Iris species prediction",
        "research_question": "Can petal length predict species?"}
DESIGN = {"task": "classification", "outcome": "iris species",
          "predictors": ["petal length"], "source_ids": ["S1"],
          "rationale": "The retrieved study describes petal measurements.",
          "search_terms": ["iris"]}
MATCH = {"suitable": True, "independent_rows": True, "task": "classification",
         "target": "species", "rationale": "Species is measured and petal length is available.",
         "predictors": [{"concept": "petal length", "column": "petal_length"}]}


def _evidence():
    return {"sources": [{"id": "S1", "title": "Iris measurements",
                         "abstract": "A study of petal measurements.",
                         "url": "https://doi.org/10.1234/demo", "year": 2024}],
            "datasets": [], "charts": [], "queries": ["iris petal length"],
            "retrieved_at": "2026-11-11", "warnings": []}


class OpenMLSimulation(BaseHTTPRequestHandler):
    """HTTP fixtures based on OpenML's JSON data API and dense ARFF format."""

    def do_GET(self):
        self.server.paths.append(self.path)
        status, response = self.server.routes.get(self.path, (501, {}))
        self.send_response(status)
        self.end_headers()
        content = response if isinstance(response, str) else json.dumps(response)
        self.wfile.write(content.encode("utf-8"))

    def log_message(self, *args):
        pass


class TabularIntegrationTests(unittest.TestCase):
    def test_verified_uci_schema_recovers_missing_predictors(self):
        schema = {"provider": "UCI Machine Learning Repository", "id": "379",
                  "name": "Website Phishing", "target": "Result",
                  "columns": ["SFH", "popUpWindow", "Result"],
                  "declared_features": ["SFH", "popUpWindow"],
                  "target_counts": {"-1": 20, "0": 10, "1": 20}}
        plan = {"topic": "Phân loại website lừa đảo",
                "research_question": "Can website phishing be classified?",
                "focus_terms": ["phishing"]}
        malformed = {"task": "classification", "outcome": "website class",
                     "rationale": "A relevant study exists.", "source_ids": ["S1"]}
        with patch("ai_scientist.tabular_research._discover_public_schema",
                   return_value=schema), patch(
                       "ai_scientist.tabular_research.get_response_from_llm",
                       return_value=(json.dumps(malformed), [])):
            designs, warnings = _designs(plan, _evidence(), object(), None)
        self.assertEqual(designs[0]["task"], "classification")
        self.assertEqual(designs[0]["predictors"], ["SFH", "popUpWindow"])
        self.assertEqual(designs[0]["search_terms"], ["Website Phishing"])
        self.assertTrue(any("operationally defined predictors" in item for item in warnings))

    def test_unrelated_uci_schema_cannot_supply_fallback_design(self):
        schema = {"provider": "UCI Machine Learning Repository", "id": "379",
                  "name": "Website Phishing", "target": "Result",
                  "columns": ["SFH", "Result"], "declared_features": ["SFH"],
                  "target_counts": {"-1": 20, "1": 20}}
        with patch("ai_scientist.tabular_research._discover_public_schema",
                   return_value=schema), patch(
                       "ai_scientist.tabular_research.get_response_from_llm",
                       return_value=(json.dumps({"candidates": [{"task": "classification",
                           "outcome": "species", "rationale": "Test.",
                           "source_ids": ["S1"]}]}), [])):
            with self.assertRaisesRegex(ValueError, "operationally defined predictors"):
                _designs(PLAN, _evidence(), object(), None)

    def test_openml_string_arff_can_drive_text_experiment(self):
        listing = "/api/v1/json/data/list/data_name/messages/limit/10"
        description = "/api/v1/json/data/777"
        download = "/data/v1/download/888"
        raw = ("@relation messages\n@attribute message string\n"
               "@attribute label {spam,ham}\n@data\n"
               + "".join(f"'urgent prize, claim reward {i}',spam\n" for i in range(25))
               + "".join(f"'team meeting notes agenda {i}',ham\n" for i in range(25)))
        paths = self._simulate({
            listing: (200, {"data": {"dataset": [{"did": 777}]}}),
            description: (200, {"data_set_description": {
                "name": "messages", "version": 1, "status": "active",
                "visibility": "public", "format": "ARFF", "file_id": 888,
                "default_target_attribute": "label", "licence": "CC0",
                "description": "Labeled independent messages.",
            }}),
            download: (200, raw),
        })
        design = {"task": "text_classification", "title": "Message filtering",
                  "outcome": "spam versus ham", "predictors": ["raw message text"],
                  "rationale": "Retrieved study examines labeled messages.",
                  "source_ids": ["S1"], "search_terms": ["messages"]}
        match = {"suitable": True, "independent_rows": True,
                 "task": "text_classification", "target": "label",
                 "text_column": "message",
                 "rationale": "The message is the predictor and label is the outcome."}
        evidence = _evidence()
        evidence["sources"][0]["title"] = "Studies of short message filtering"
        with tempfile.TemporaryDirectory() as folder, patch(
            "ai_scientist.tabular_research.get_response_from_llm",
            side_effect=[(json.dumps(design), []), (json.dumps(match), [])],
        ):
            _, result = plan_and_run(
                {"topic": "Message spam", "Title": "Message spam",
                 "research_question": "Can text predict spam?"},
                evidence, folder, model=object(),
            )
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["dataset"]["provider"], "OpenML")
            self.assertEqual(paths, [listing, description, download])
            self.assertIn(",", pd.read_csv(Path(folder, "experiment_dataset.csv")).loc[0, "message"])

    def test_openml_search_continues_after_one_transient_failure(self):
        session = object()
        listing = json.dumps({"data": {"dataset": [{"did": 61}]}}).encode("utf-8")
        with patch("ai_scientist.openml_data._get",
                   side_effect=[requests.Timeout("slow first query"), listing]) as get:
            found = list(candidates(session, ["first-term", "second-term"]))
        self.assertEqual(found[0][0], "61")
        self.assertEqual(get.call_count, 2)

    def test_failed_prediction_reason_is_not_hidden_by_future_idea(self):
        designs = {"candidates": [
            DESIGN,
            {"task": "future_study", "title": "Prospective trial",
             "hypothesis": "Collect future measurements.",
             "rationale": "The cited work motivates an intervention.",
             "source_ids": ["S1"], "data_requirements": "New observations.",
             "comparison": "Current practice.", "evaluation": "Compare outcomes.",
             "reason": "A new trial has not been conducted."},
        ]}
        with tempfile.TemporaryDirectory() as folder, patch(
            "ai_scientist.tabular_research.get_response_from_llm",
            return_value=(json.dumps(designs), []),
        ), patch("ai_scientist.tabular_research.candidates",
                 side_effect=requests.Timeout("OpenML timed out")), patch(
                     "ai_scientist.tabular_research.uci_candidates", return_value=[]):
            proposal, result = plan_and_run(PLAN, _evidence(), folder, model=object())
        self.assertEqual(proposal["decision"], "review_only")
        self.assertEqual(result["reason_code"], "source_unavailable")
        self.assertIn("timed out", result["reason"])
        self.assertEqual(len(result["candidate_attempts"]), 2)
        self.assertIn("new trial", result["candidate_attempts"][-1]["reason"])

    def test_labeled_text_runs_real_heldout_benchmark_and_enters_paper(self):
        evidence = _evidence()
        evidence["sources"][0]["title"] = "SMS spam classification"
        evidence["sources"][0]["abstract"] = "Machine learning classifies labeled SMS messages."
        design = {"task": "text_classification", "title": "Labeled message filtering",
                  "outcome": "spam versus ham", "predictors": ["raw message text"],
                  "rationale": "The cited study motivates text classification.",
                  "source_ids": ["S1"], "search_terms": ["sms-spam"]}
        match = {"suitable": True, "independent_rows": True,
                 "task": "text_classification", "target": "label",
                 "text_column": "message",
                 "rationale": "The message column is available before its spam label."}
        with tempfile.TemporaryDirectory() as folder, patch(
            "ai_scientist.tabular_research.get_response_from_llm",
            side_effect=[(json.dumps(design), []), (json.dumps(match), [])],
        ), patch("ai_scientist.tabular_research.run_benchmark") as tabular:
            path = Path(folder, "messages.csv")
            pd.DataFrame({
                "message": ([f"urgent free prize claim your offer code {i}" for i in range(40)]
                            + [f"meeting agenda project update from team {i}" for i in range(40)]),
                "label": ["spam"] * 40 + ["ham"] * 40,
            }).to_csv(path, index=False)
            proposal, result = plan_and_run(
                {"topic": "SMS spam with machine learning",
                 "Title": "SMS spam classification",
                 "research_question": "Can text classify spam?"},
                evidence, folder, model=object(), data_path=path,
                target="label", task="text_classification",
            )
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["experiment_type"], "text_classification")
            self.assertEqual(proposal["selected_idea_id"], "I2")
            self.assertGreater(result["improvement_over_baseline"], 0)
            self.assertEqual(len(pd.read_csv(Path(folder, result["predictions_file"]))), 80)
            self.assertEqual(len(result["protocol"]["fold_results"]), 5)
            tabular.assert_not_called()
            if shutil.which("pdflatex"):
                pdf = perform_writeup(
                    {"topic": "SMS spam with machine learning",
                     "Title": "SMS spam classification",
                     "research_question": "Can text classify spam?",
                     "search_queries": ["sms spam"]}, None, folder)
                body = " ".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
                self.assertIn("Text classification experiment", body)
                self.assertIn("TF-IDF", body)

    def test_text_benchmark_rejects_conflicting_duplicate_labels(self):
        frame = pd.DataFrame({
            "message": ["same message"] * 10 + [f"other message {i}" for i in range(30)],
            "label": ["spam"] * 5 + ["ham"] * 5 + ["ham"] * 30,
        })
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                run_text_benchmark(frame, {"target": "label", "name": "test",
                                           "provider": "User CSV"},
                                   {"text_column": "message"}, folder)
            self.assertFalse(Path(folder, "experiment_predictions.csv").exists())

    def test_dense_arff_string_parser_keeps_text_with_commas(self):
        raw = ("@relation messages\n"
               "@attribute message string\n"
               "@attribute label {spam,ham}\n"
               "@data\n"
               "'Free prize, claim now',spam\n"
               "'Team meeting tomorrow',ham\n")
        parsed = _parse_dense_string_arff(raw)
        self.assertEqual(parsed.loc[0, "message"], "Free prize, claim now")
        self.assertEqual(parsed.loc[1, "label"], "ham")

    def _simulate(self, routes):
        server = ThreadingHTTPServer(("127.0.0.1", 0), OpenMLSimulation)
        self._server_port = server.server_port
        server.routes, server.paths = routes, []
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        override = patch("ai_scientist.openml_data.BASE_URL", f"http://127.0.0.1:{server.server_port}")
        override.start()
        self.addCleanup(override.stop)
        proxy_override = patch.dict("os.environ", {"NO_PROXY": "127.0.0.1"})
        proxy_override.start()
        self.addCleanup(proxy_override.stop)
        return server.paths

    def test_multiple_ideas_try_next_feasible_design_and_keep_rejection(self):
        alternatives = {"candidates": [
            {**DESIGN, "outcome": "an unrelated label", "search_terms": ["unrelated"]},
            DESIGN,
            {"task": "future_study", "title": "Prospective validation",
             "hypothesis": "The method transfers to another population.",
             "rationale": "Retrieved work motivates external validation.",
             "source_ids": ["S1"], "data_requirements": "New labeled samples.",
             "comparison": "Existing method.", "evaluation": "Evaluate on held-out samples.",
             "reason": "New samples have not been collected."},
        ]}
        with tempfile.TemporaryDirectory() as folder, patch(
            "ai_scientist.tabular_research.get_response_from_llm",
            side_effect=[(json.dumps(alternatives), []),
                         (json.dumps({"suitable": False, "rationale": "Wrong outcome."}), []),
                         (json.dumps(MATCH), [])],
        ):
            path = Path(folder, "local.csv")
            pd.DataFrame({"petal_length": range(40),
                          "species": ["a"] * 20 + ["b"] * 20}).to_csv(path, index=False)
            proposal, result = plan_and_run(PLAN, _evidence(), folder, model=object(),
                                            data_path=path, target="species", task="classification")
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(proposal["selected_idea_id"], "I3")
            self.assertEqual(len(proposal["ideas"]), 4)
            self.assertIn("Wrong outcome", proposal["ideas"][1]["feasibility_reason"])
            self.assertEqual(proposal["ideas"][2]["status"], "feasible")
            self.assertTrue(Path(folder, result["predictions_file"]).is_file())

    def test_openml_validates_mapping_runs_nested_cv_and_exports_results(self):
        listing = "/api/v1/json/data/list/data_name/iris/limit/10"
        description = "/api/v1/json/data/61"
        download = "/data/v1/download/1"
        arff = ("@relation iris\n@attribute row_id numeric\n@attribute petal_length numeric\n"
                "@attribute species {a,b}\n@data\n"
                + "".join(f"{i},{i / 10},{'a' if i < 20 else 'b'}\n" for i in range(40)))
        paths = self._simulate({
            listing: (200, {"data": {"dataset": [{"did": 61}]}}),
            description: (200, {"data_set_description": {
                "name": "Iris", "version": 1, "status": "active", "visibility": "public",
                "format": "ARFF", "file_id": 1, "default_target_attribute": "species",
                "row_id_attribute": "row_id", "licence": "CC0",
            }}),
            download: (200, arff),
        })
        with tempfile.TemporaryDirectory() as folder, patch(
            "ai_scientist.tabular_research.get_response_from_llm",
            side_effect=[(json.dumps(DESIGN), []), (json.dumps(MATCH), [])],
        ) as llm:
            evidence = _evidence()
            proposal, result = plan_and_run(PLAN, evidence, folder, model=object())
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(paths, [listing, description, download])
            self.assertEqual(proposal["selected_idea_id"], "I2")
            self.assertIn("macro f1", proposal["ideas"][-1]["protocol"]["evaluation"])
            self.assertEqual(result["dataset"]["features"], ["petal_length"])
            self.assertEqual(result["dataset"]["excluded_columns"], ["row_id"])
            self.assertNotIn("row_id", llm.call_args.args[0])
            predictions = pd.read_csv(Path(folder, result["predictions_file"]))
            self.assertEqual(len(predictions), 40)
            self.assertEqual(predictions.outer_fold.nunique(), 5)
            self.assertGreater(result["improvement_over_baseline"], 0)
            self.assertTrue(Path(folder, result["plot_file"]).is_file())
            if shutil.which("pdflatex"):
                pdf = perform_writeup(PLAN, None, folder)
                body = " ".join(page.extract_text() for page in PdfReader(pdf).pages)
                self.assertIn("Tabular predictive benchmark", body)
                self.assertIn("https://www.openml.org/d/61", body)
                self.assertIn("Experiment figure", body)

    def test_future_study_is_written_without_attempting_a_benchmark(self):
        future = {
            "task": "future_study", "title": "Prospective irrigation trial",
            "hypothesis": "The intervention increases yield compared with standard irrigation.",
            "rationale": "Measurements in the retrieved literature motivate a controlled trial.",
            "source_ids": ["S1"], "data_requirements": "Yield measured per randomized plot.",
            "comparison": "Standard irrigation.", "evaluation": "Compare mean yield per plot.",
            "reason": "A randomized field trial requires new measurements.",
        }
        with tempfile.TemporaryDirectory() as folder, patch(
            "ai_scientist.tabular_research.get_response_from_llm",
            return_value=(json.dumps(future), []),
        ), patch("ai_scientist.tabular_research.candidates") as lookup, patch(
            "ai_scientist.tabular_research.run_benchmark",
        ) as benchmark:
            proposal, result = plan_and_run(PLAN, _evidence(), folder, model=object())
            saved = json.loads(Path(folder, "research_proposal.json").read_text(encoding="utf-8"))
            self.assertEqual(proposal["decision"], "review_only")
            self.assertEqual(result["status"], "not_run")
            self.assertEqual(saved["ideas"][-1]["protocol"]["comparison"], future["comparison"])
            self.assertEqual(saved["ideas"][-1]["source_ids"], ["S1"])
            lookup.assert_not_called()
            benchmark.assert_not_called()
            if shutil.which("pdflatex"):
                pdf = perform_writeup(PLAN, None, folder)
                body = " ".join(page.extract_text() for page in PdfReader(pdf).pages)
                self.assertIn("Prospective irrigation trial", body)
                self.assertIn("Standard irrigation.", body)
                self.assertIn("Compare mean yield per plot.", body)
                self.assertNotIn("Tabular predictive benchmark", body)

    def test_openml_503_keeps_the_review_without_experiment_results(self):
        listing = "/api/v1/json/data/list/data_name/iris/limit/10"
        paths = self._simulate({listing: (503, {"error": "unavailable"})})
        with tempfile.TemporaryDirectory() as folder, patch(
            "ai_scientist.tabular_research.get_response_from_llm",
            return_value=(json.dumps(DESIGN), []),
        ), patch("ai_scientist.tabular_research.uci_candidates", return_value=[]):
            proposal, result = plan_and_run(PLAN, _evidence(), folder, model=object())
            self.assertEqual(paths, [listing, listing], result)
            self.assertEqual(proposal["decision"], "review_only")
            self.assertEqual(result["status"], "not_run")
            self.assertIn("503", result["reason"])
            self.assertFalse(Path(folder, "experiment_predictions.csv").exists())

    def test_uci_fallback_runs_benchmark_and_cites_dataset(self):
        listing = "/api/v1/json/data/list/data_name/iris/limit/10"
        uci_list = "/api/datasets/list?search=iris&filter=python"
        uci_info = "/api/dataset?id=61"
        uci_csv = "/static/public/61/data.csv"
        rows = "petal_length,species\n" + "".join(
            f"{i / 10},{'a' if i < 20 else 'b'}\n" for i in range(40))
        paths = self._simulate({
            listing: (412, {}),
            uci_list: (200, {"status": 200, "data": [{"id": 61, "name": "Iris data"}]}),
            uci_info: (200, {"status": 200, "data": {
                "uci_id": 61, "name": "Iris data", "num_instances": 40,
                "num_features": 1, "target_col": ["species"],
                "data_url": "https://archive.ics.uci.edu/static/public/61/data.csv",
                "dataset_doi": "10.24432/C5DEMO", "additional_info": {"summary": "Iris species."},
                "variables": [{"name": "petal_length", "role": "Feature"},
                              {"name": "species", "role": "Target"}],
            }}),
            uci_csv: (200, rows),
        })
        with tempfile.TemporaryDirectory() as folder, patch(
            "ai_scientist.uci_data.BASE_URL", "http://127.0.0.1:" +
            str(self._server_port)
        ), patch("ai_scientist.tabular_research.get_response_from_llm",
                 side_effect=[(json.dumps(DESIGN), []), (json.dumps(MATCH), [])]) as llm:
            proposal, result = plan_and_run(PLAN, _evidence(), folder, model=object())
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["dataset"]["provider"], "UCI Machine Learning Repository")
            self.assertEqual(result["dataset"]["target"], "species")
            assessment_prompt = llm.call_args_list[-1].args[0]
            self.assertIn('"all_columns": ["petal_length", "species"]', assessment_prompt)
            self.assertIn('"predictor_columns": ["petal_length"]', assessment_prompt)
            self.assertIn('"target_role": "Target"', assessment_prompt)
            self.assertEqual(paths, [listing, uci_list, uci_info, uci_csv])
            if shutil.which("pdflatex"):
                pdf = perform_writeup(PLAN, None, folder)
                body = " ".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
                self.assertIn("Tabular predictive benchmark", body)
                self.assertIn("Consistency across held-out folds", body)
                self.assertIn("10.24432/C5DEMO", body)

    def test_csv_rejects_unmapped_predictor_without_running(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "ai_scientist.tabular_research.get_response_from_llm",
            side_effect=[(json.dumps(DESIGN), []), (json.dumps({
                **MATCH, "predictors": [{"concept": "petal length", "column": "missing"}],
            }), [])],
        ), patch("ai_scientist.tabular_research.run_benchmark") as benchmark:
            path = Path(folder, "local.csv")
            pd.DataFrame({"petal_length": range(40), "species": ["a", "b"] * 20}).to_csv(path, index=False)
            proposal, result = plan_and_run(PLAN, _evidence(), folder, model=object(),
                                            data_path=path, target="species", task="classification")
            self.assertEqual(proposal["decision"], "review_only")
            self.assertEqual(result["status"], "not_run")
            benchmark.assert_not_called()

    def test_regression_scores_numeric_and_categorical_columns(self):
        frame = pd.DataFrame({"x": range(50), "category": ["x", "y", None, "x", "y"] * 10,
                              "score": [2 * n + 1 for n in range(50)]})
        with tempfile.TemporaryDirectory() as folder:
            result = run_benchmark(
                frame, {"provider": "User CSV", "name": "example.csv", "target": "score"},
                {"task": "regression", "features": ["x", "category"]}, folder,
            )
            self.assertEqual(result["primary_metric"], "mae")
            self.assertGreater(result["improvement_over_baseline"], 0)
            self.assertIn("r2", result["metrics"]["nested_selection"])

    def test_tabular_rejects_predictors_with_too_few_distinct_patterns(self):
        frame = pd.DataFrame({"row_key": range(80), "feature": [i % 8 for i in range(80)],
                              "label": ["a" if i < 40 else "b" for i in range(80)]})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "fewer than 30 independent feature patterns"):
                run_benchmark(frame, {"target": "label", "name": "discrete",
                                      "provider": "User CSV"},
                              {"task": "classification", "features": ["feature"]}, folder)

    def test_tabular_keeps_identical_predictor_patterns_in_one_outer_fold(self):
        frame = pd.DataFrame({"row_key": range(80), "feature": list(range(40)) * 2,
                              "label": ["a" if i < 20 else "b" for i in range(40)] * 2})
        with tempfile.TemporaryDirectory() as folder:
            result = run_benchmark(frame, {"target": "label", "name": "repeated",
                                           "provider": "User CSV"},
                                   {"task": "classification", "features": ["feature"]}, folder)
            predictions = pd.read_csv(Path(folder, "experiment_predictions.csv"))
            self.assertEqual(result["dataset"]["rows"], 80)
            self.assertEqual(predictions.groupby("feature_group").outer_fold.nunique().max(), 1)
            self.assertEqual(predictions.outer_fold.nunique(), 5)

    def test_tabular_handles_three_numeric_class_labels(self):
        frame = pd.DataFrame({"feature": range(60),
                              "label": [-1] * 20 + [0] * 20 + [1] * 20})
        with tempfile.TemporaryDirectory() as folder:
            result = run_benchmark(frame, {"target": "label", "name": "three classes",
                                           "provider": "User CSV"},
                                   {"task": "classification", "features": ["feature"]}, folder)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["dataset"]["class_counts"],
                             {"-1": 20, "0": 20, "1": 20})
            self.assertEqual(set(pd.read_csv(Path(folder, "experiment_predictions.csv"))
                                 .observed.unique()), {-1, 0, 1})
