import csv
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pypdf import PdfReader

from ai_scientist.analyze_evidence import analyze_series
from ai_scientist.evidence_interpretation import analyze_charts, build_interpretation
from ai_scientist.generate_ideas import generate_ideas
from ai_scientist.fulltext import (
    _read_open_pdf, enrich_with_open_fulltext, fetch_open_fulltext,
    parse_open_article_xml, parse_open_pdf,
)
from ai_scientist.perform_writeup import (
    _draft_sections, _evidence_appraisal_block, _expand_analysis_claims,
    clean_text_for_latex, describe_analysis,
    perform_writeup, summarize_analysis,
)
from ai_scientist.research_evidence import (
    _deduplicate_sources, build_charts, collect_evidence, fetch_owid_chart,
    fetch_world_bank, search_openalex, search_semantic_scholar,
)
from ai_scientist.research_ideas import plan_and_run, propose_ideas
from ai_scientist.s2_client import RequestPacer, get_json as s2_get_json
from ai_scientist.study_evidence import (
    _candidate_passages, analyze_findings, extract_study_findings, rank_sources,
    validate_finding,
)
from ai_scientist.synthesis import evidence_map, profile_sources


class FakeResponse:
    def __init__(self, url, payload=None, text=""):
        self.url = url
        self._payload = payload
        self.text = text
        self.content = text.encode("utf-8")

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class FakeSession:
    def get(self, url, **kwargs):
        if "/indicator/SP.POP.TOTL" in url and "/country/" not in url:
            return FakeResponse(url, [{}, [{"name": "Population, total", "unit": "people",
                                           "sourceOrganization": "World Bank"}]])
        if "/country/WLD/indicator/SP.POP.TOTL" in url:
            return FakeResponse(url, [{}, [{"date": "2022", "value": 100,
                                            "country": {"value": "World"}},
                                           {"date": "2023", "value": 110,
                                            "country": {"value": "World"}},
                                           {"date": "2024", "value": None}]])
        if url.endswith(".metadata.json"):
            return FakeResponse(url, {"chart": {"title": "Life expectancy", "citation": "UN data"},
                                      "columns": {"Life expectancy": {"type": "Numeric", "unit": "years",
                                                                       "citationShort": "UN data"}}})
        if url.endswith(".csv"):
            return FakeResponse(url, text="Entity,Code,Year,Life expectancy\nWorld,OWID_WRL,2022,72\nWorld,OWID_WRL,2023,73\n")
        raise AssertionError(url)


def sample_evidence():
    return {
        "topic": "World population", "retrieved_at": "2026-09-24",
        "queries": ["world population"], "warnings": [],
        "sources": [{"id": "S1", "title": "Population dynamics", "year": 2024,
                     "doi": "10.1234/demo", "url": "https://doi.org/10.1234/demo",
                     "abstract": "A study of population.", "providers": ["Crossref"],
                     "query": "world population"}],
        "datasets": [{"id": "D1", "provider": "World Bank", "title": "Population, total",
                      "entity": "World", "unit": "people", "credit": "World Bank",
                      "source_url": "https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL",
                      "retrieved_at": "2026-09-24", "points": [
                          {"year": 2022, "value": 100}, {"year": 2023, "value": 110}]}],
    }


class PipelineTests(unittest.TestCase):
    def test_extended_synthesis_keeps_only_source_supported_comparisons(self):
        first = ("The first study extracts URL and HTML attributes and evaluates a "
                 "classification model on a held-out sample. Its reported setting is "
                 "limited to a historical collection of websites.")
        second = ("The second study analyzes rendered page text with a language model "
                  "and discusses the difficulty of applying its findings to new domains. "
                  "The abstract does not describe a matched baseline.")
        evidence = {"sources": [{"id": "S1", "title": "Feature study", "abstract": first},
                                {"id": "S2", "title": "Language study", "abstract": second}]}
        text = (("The first paper uses URL and HTML attributes [S1], whereas the second "
                 "examines rendered page text [S2]. They therefore observe different "
                 "parts of a website, which matters when considering whether their "
                 "classifiers answer the same operational question. The first reports "
                 "a held-out evaluation [S1], while the second abstract does not "
                 "describe a matched baseline [S2]. This difference in accessible "
                 "evaluation detail prevents a direct ranking from the supplied "
                 "passages. The reported historical collection [S1] and concerns "
                 "about new domains [S2] also limit any claim that performance "
                 "would transfer to a changing web. A new comparison would need "
                 "the same label definition, features available at prediction time, "
                 "and an evaluation split shared by both methods."))
        claim = {"text": text, "support": [{"source_id": "S1", "quote": first},
                                           {"source_id": "S2", "quote": second}]}
        invalid = {"text": text.replace("[S2]", "[S3]"), "support": claim["support"]}
        with patch("ai_scientist.perform_writeup.get_response_from_llm",
                   return_value=(json.dumps({"analysis_claims": [invalid, claim]}), [])):
            result = _expand_analysis_claims(
                {"Title": "Website detection", "research_question": "Which works?"},
                evidence, object(), [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], text)

    def test_short_model_prose_keeps_substantive_review_fallback(self):
        evidence = sample_evidence()
        evidence["datasets"] = []
        tiny = {key: "A brief unsupported summary." for key in
                ("abstract", "introduction", "related_work", "discussion", "conclusion")}
        with patch("ai_scientist.perform_writeup.get_response_from_llm",
                   return_value=(json.dumps(tiny), [])):
            sections = _draft_sections(
                {"Title": "Population review", "research_question": "What is known?"},
                evidence, object(),
            )
        self.assertGreaterEqual(len(sections["introduction"].split()), 90)
        self.assertGreaterEqual(len(sections["discussion"].split()), 90)
        self.assertIn("What is known?", sections["abstract"])

    def test_evidence_appraisal_compares_methods_without_inventing_validation(self):
        evidence = {
            "sources": [
                {"id": "S1", "title": "Forest defect prediction",
                 "fulltext": {"status": "full_text"}},
                {"id": "S2", "title": "Linear defect prediction",
                 "fulltext": {"status": "abstract_only"}},
            ],
            "source_profiles": [
                {"source_id": "S1", "methods": ["Random forest"],
                 "tasks": ["software defect prediction"],
                 "validation_passage": {"location": "Methods", "quote": "Held-out evaluation."}},
                {"source_id": "S2", "methods": ["Logistic regression"],
                 "tasks": ["software defect prediction"],
                 "validation_passage": None},
            ],
        }
        block = _evidence_appraisal_block(evidence)
        self.assertIn("Random forest [S1] and Logistic regression [S2]", block)
        self.assertIn("Validation detail not verified", block)
        self.assertIn("Open full text", block)
        self.assertIn("head-to-head performance claim", block)
        self.assertNotIn("Random forest is superior", block)

    def test_research_idea_runs_only_on_eligible_series_and_uses_past_values(self):
        evidence = sample_evidence()
        evidence["datasets"][0]["points"] = [
            {"year": 2010 + index, "value": 100 + 2 * index}
            for index in range(12)]
        plan = {"topic": "World population", "Title": "Population review"}
        with tempfile.TemporaryDirectory() as folder:
            proposal, experiment = plan_and_run(plan, evidence, folder)
            self.assertEqual(proposal["decision"], "run_experiment")
            self.assertEqual(experiment["status"], "completed")
            self.assertEqual(experiment["test_count"], 4)
            self.assertEqual(experiment["metrics"]["linear_trend_mae"], 0)
            self.assertEqual(experiment["metrics"]["last_value_mae"], 2)
            self.assertEqual(experiment["lower_mae"], "linear_trend")
            with open(Path(folder) / "experiment_predictions.csv", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["year"], "2018")
            self.assertEqual(float(rows[0]["last_value_prediction"]), 114)
            self.assertEqual(float(rows[0]["linear_trend_prediction"]), 116)
            self.assertTrue((Path(folder) / "experiment_plot.png").is_file())
            self.assertEqual(json.loads((Path(folder) / "experiment_results.json").read_text())
                             ["status"], "completed")

    def test_backtest_keeps_result_and_adds_untested_source_grounded_ideas(self):
        evidence = sample_evidence()
        evidence["datasets"][0]["points"] = [
            {"year": 2010 + index, "value": 100 + 2 * index} for index in range(12)]
        ideas = {"candidates": [{
            "task": "future_study", "title": "Prospective population survey",
            "hypothesis": "A new survey could measure migration effects.",
            "rationale": "The retrieved source motivates further measurement.",
            "source_ids": ["S1"], "data_requirements": "New survey observations.",
            "comparison": "Earlier survey estimates.",
            "evaluation": "Compare measured population changes.",
            "reason": "New observations are unavailable.",
        }]}
        with tempfile.TemporaryDirectory() as folder, patch(
            "ai_scientist.tabular_research.get_response_from_llm",
            return_value=(json.dumps(ideas), []),
        ):
            proposal, experiment = plan_and_run(
                {"topic": "Population", "Title": "Population review"}, evidence,
                folder, model=object(),
            )
            self.assertEqual(experiment["status"], "completed")
            self.assertEqual(proposal["selected_idea_id"], "I1")
            self.assertEqual(len(proposal["ideas"]), 3)
            self.assertEqual(proposal["ideas"][-1]["status"], "not_feasible")
            self.assertEqual(proposal["ideas"][-1]["source_ids"], ["S1"])

    def test_research_idea_falls_back_without_sufficient_contiguous_data(self):
        plan = {"topic": "A topic", "Title": "Review"}
        for points in ([], [{"year": 2010 + index, "value": index}
                            for index in range(11)],
                       [{"year": 2010 + 2 * index, "value": index}
                        for index in range(12)]):
            evidence = sample_evidence()
            evidence["datasets"] = ([{"id": "D1", "title": "Sample", "entity": "World",
                                      "unit": "units", "points": points}] if points else [])
            with self.subTest(count=len(points)), tempfile.TemporaryDirectory() as folder:
                proposal, experiment = plan_and_run(plan, evidence, folder)
                self.assertEqual(proposal["decision"], "review_only")
                self.assertEqual(experiment["status"], "not_run")
                self.assertFalse((Path(folder) / "experiment_predictions.csv").exists())

    def test_experiment_failure_does_not_block_review(self):
        evidence = sample_evidence()
        evidence["datasets"][0]["points"] = [
            {"year": 2010 + index, "value": index} for index in range(12)]
        with tempfile.TemporaryDirectory() as folder, patch(
                "ai_scientist.research_ideas._run_backtest", side_effect=ValueError("test failure")):
            proposal, experiment = plan_and_run({"topic": "Population"}, evidence, folder)
            self.assertEqual(proposal["decision"], "review_only")
            self.assertEqual(experiment["status"], "failed")
            self.assertIn("test failure", proposal["reason"])

    def test_literature_idea_is_source_attributed_and_not_run_without_dataset(self):
        evidence = sample_evidence()
        evidence["datasets"] = []
        evidence["evidence_map"] = [
            {"method": "Survey", "task": "mood", "count": 1, "source_ids": ["S1"]},
            {"method": "Cohort", "task": "mood", "count": 1, "source_ids": ["S2"]},
        ]
        proposal = propose_ideas({"topic": "mood"}, evidence)
        self.assertEqual(proposal["decision"], "review_only")
        self.assertEqual(proposal["ideas"][0]["source_ids"], ["S1", "S2"])
        self.assertIn("[S1]", proposal["ideas"][0]["hypothesis"])
        self.assertIn("[S2]", proposal["ideas"][0]["hypothesis"])
        self.assertEqual(proposal["ideas"][0]["status"], "not_feasible")

    def test_semantic_scholar_key_is_header_only_and_requests_are_spaced(self):
        class Clock:
            now = 100.0

            def monotonic(self):
                return self.now

            def sleep(self, seconds):
                self.now += seconds

        class Session:
            def __init__(self, clock):
                self.clock = clock
                self.calls = []

            def get(self, url, **kwargs):
                self.calls.append((self.clock.now, url, kwargs))
                return FakeResponse(url, {"data": []})

        clock = Clock()
        session = Session(clock)
        with patch.dict(os.environ, {"SEMANTIC_SCHOLAR_API_KEY": "test-only-secret"}), \
             patch("ai_scientist.s2_client.PACER",
                   RequestPacer(clock=clock.monotonic, sleeper=clock.sleep)):
            s2_get_json(session, "/graph/v1/paper/search", {"query": "one"})
            s2_get_json(session, "/graph/v1/paper/search", {"query": "two"})
        self.assertGreaterEqual(session.calls[1][0] - session.calls[0][0], 1.25)
        self.assertEqual(session.calls[0][2]["headers"]["x-api-key"], "test-only-secret")
        self.assertNotIn("test-only-secret", session.calls[0][1])
        self.assertNotIn("test-only-secret", str(session.calls[0][2]["params"]))

    def test_semantic_scholar_429_obeys_retry_after(self):
        class Clock:
            now = 100.0

            def monotonic(self):
                return self.now

            def sleep(self, seconds):
                self.now += seconds

        class RetrySession:
            def __init__(self, clock):
                self.clock = clock
                self.times = []

            def get(self, url, **kwargs):
                self.times.append(self.clock.now)
                response = FakeResponse(url, {"data": []})
                response.status_code = 429 if len(self.times) == 1 else 200
                response.headers = {"Retry-After": "3"}
                return response

        clock = Clock()
        session = RetrySession(clock)
        with patch("ai_scientist.s2_client.PACER",
                   RequestPacer(clock=clock.monotonic, sleeper=clock.sleep)):
            self.assertEqual(s2_get_json(session, "/graph/v1/paper/search", {}), {"data": []})
        self.assertGreaterEqual(session.times[1] - session.times[0], 3.0)

    def test_semantic_scholar_source_merges_open_pdf_without_duplicate(self):
        class ScholarSession:
            def get(self, url, **kwargs):
                return FakeResponse(url, {"data": [{
                    "title": "Machine learning in criminal profiling", "year": 2025,
                    "abstract": "Machine learning for criminal profiling.",
                    "externalIds": {"DOI": "10.1234/FORENSIC"},
                    "openAccessPdf": {"url": "https://example.org/paper.pdf"},
                    "authors": [{"name": "A Researcher"}], "venue": "Example Journal",
                }]})

        scholar = search_semantic_scholar("machine learning criminal profiling", ScholarSession())[0]
        indexed = dict(scholar, providers=["OpenAlex"], oa_pdf_url=None)
        merged = _deduplicate_sources([indexed, scholar])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["oa_pdf_url"], "https://example.org/paper.pdf")
        self.assertEqual(merged[0]["providers"], ["OpenAlex", "Semantic Scholar"])
        without_doi = dict(scholar, doi="", url="https://example.org/record")
        merged_without_doi = _deduplicate_sources([without_doi, indexed])
        self.assertEqual(len(merged_without_doi), 1)
        self.assertEqual(merged_without_doi[0]["doi"], "10.1234/forensic")

    def test_open_pdf_is_used_only_after_oa_xml_miss(self):
        class NoXmlSession:
            def get(self, url, **kwargs):
                self_url = url
                return FakeResponse(self_url, {"resultList": {"result": []}})

        source = {"id": "S1", "doi": "10.1234/forensic", "title": "Forensic study",
                  "oa_pdf_url": "https://example.org/forensic.pdf"}
        with patch("ai_scientist.fulltext._read_open_pdf",
                   return_value=(b"%PDF-test", source["oa_pdf_url"])), \
             patch("ai_scientist.fulltext.parse_open_pdf", return_value=[
                 {"location": "page 2, passage 1", "section": "Page 2",
                  "kind": "pdf_text", "text": "A validated result."}]):
            fulltext = fetch_open_fulltext(source, NoXmlSession())
        self.assertEqual(fulltext["status"], "full_text")
        self.assertEqual(fulltext["provider"], "open-access PDF")
        self.assertEqual(fulltext["passages"][0]["location"], "page 2, passage 1")

    def test_private_pdf_link_is_never_requested(self):
        class NoRequestSession:
            def get(self, url, **kwargs):
                raise AssertionError("Private address was requested")

        with self.assertRaisesRegex(ValueError, "public HTTPS"):
            _read_open_pdf("http://127.0.0.1/private.pdf", NoRequestSession())

    def test_open_pdf_candidates_are_prioritized_for_fulltext_lookup(self):
        sources = [{"id": "S1", "doi": "10.1/one"},
                   {"id": "S2", "doi": "10.1/two", "oa_pdf_url": "https://example.org/two.pdf"}]
        with patch("ai_scientist.fulltext.fetch_open_fulltext", return_value={
                "status": "full_text", "passages": [{"text": "Readable text"}]}) as fetch:
            enrich_with_open_fulltext(sources, object(), limit=1)
        self.assertEqual(fetch.call_args.args[0]["id"], "S2")
        self.assertEqual(sources[0]["fulltext"]["status"], "abstract_only")

    def test_pdf_result_passage_is_prioritized_over_abstract(self):
        source = {"id": "S1", "title": "Machine learning and profiling",
                  "abstract": "The classifier showed accuracy of 0.70 in criminal profiling.",
                  "fulltext": {"status": "full_text", "passages": [{
                      "section": "Page 4", "location": "page 4, passage 7", "kind": "pdf_text",
                      "text": "Machine learning achieved accuracy of 0.84 in criminal profiling."}]}}
        passages = _candidate_passages(source)
        self.assertEqual(passages[0]["location"], "page 4, passage 7")

    @unittest.skipUnless(shutil.which("pdflatex"), "pdflatex unavailable")
    def test_open_pdf_parser_checks_title_and_keeps_page_location(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.tex"
            source.write_text("\\documentclass{article}\n\\begin{document}\n"
                              "Machine learning for criminal profiling.\\par\n"
                              + ("This article examines machine learning for criminal profiling "
                                 "and reports validation methods in forensic settings.\\par\n") * 18
                              + "\\end{document}\n", encoding="utf-8")
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "-output-directory", folder, str(source)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pdf = (Path(folder) / "source.pdf").read_bytes()
            passages = parse_open_pdf(pdf, "Machine learning for criminal profiling")
            self.assertIn("page 1", passages[0]["location"])
            with self.assertRaisesRegex(ValueError, "title does not match"):
                parse_open_pdf(pdf, "Marine geology and ocean currents")

    def test_open_fulltext_requires_matching_open_doi_and_keeps_section_location(self):
        xml = ("<article><body><sec><title>Results</title><p>The machine learning "
               "classifier achieved accuracy of 0.84 for criminal profiling.</p></sec>"
               "<sec><title>Discussion</title><p>A limitation is the small sample "
               "and lack of external validation.</p></sec></body></article>")

        class OpenSession:
            def get(self, url, **kwargs):
                if url.endswith("/search"):
                    return FakeResponse(url, {"resultList": {"result": [
                        {"doi": "10.example/wrong", "pmcid": "PMC111", "isOpenAccess": "Y"},
                        {"doi": "10.example/ai", "pmcid": "PMC123", "isOpenAccess": "Y"}]}})
                if url.endswith("/PMC123/fullTextXML"):
                    return FakeResponse(url, text=xml)
                raise AssertionError(url)

        source = {"id": "S1", "doi": "10.example/ai", "url": "https://doi.org/10.example/ai",
                  "title": "Machine learning for criminal profiling", "abstract": ""}
        fulltext = fetch_open_fulltext(source, OpenSession())
        self.assertEqual(fulltext["status"], "full_text")
        self.assertIn("Results", fulltext["passages"][0]["location"])
        source["fulltext"] = fulltext
        findings, warning = extract_study_findings(
            [source], plan={"exposure_terms": ["machine learning"],
                            "outcome_terms": ["criminal profiling"]})
        self.assertIsNone(warning)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["estimate"], 0.84)
        self.assertEqual(findings[0]["access_level"], "full_text")
        self.assertIn("Results", findings[0]["location"])
        profiles = profile_sources([source], {"outcome_terms": ["criminal profiling"]})
        self.assertIsNotNone(profiles[0]["risk_passage"])
        self.assertIsNotNone(profiles[0]["validation_passage"])
        self.assertEqual(evidence_map(profiles)[0]["count"], 1)

    def test_unreadable_open_article_does_not_count_as_full_text(self):
        with self.assertRaises(ValueError):
            parse_open_article_xml(b"<article><body/></article>")

    def test_qualitative_method_map_replaces_publication_year_histogram(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = sample_evidence()
            evidence["datasets"] = []
            evidence["source_profiles"] = [{"source_id": "S1", "title": "Neural network for profiling",
                                            "access_level": "abstract", "methods": ["Neural network"],
                                            "tasks": ["criminal profiling"],
                                            "risk_passage": None, "validation_passage": None}]
            evidence["evidence_map"] = evidence_map(evidence["source_profiles"])
            build_charts(evidence, folder)
            self.assertEqual([item["kind"] for item in evidence["charts"]], ["method_task_map"])
            self.assertTrue((Path(folder) / "chart_01.png").is_file())
            self.assertTrue((Path(folder) / "chart_01.png").read_bytes().startswith(b"\x89PNG"))
            if evidence["charts"][0]["renderer"] == "graphviz":
                source = (Path(folder) / "chart_01.gv").read_text(encoding="utf-8")
                self.assertIn("Neural network", source)
                self.assertIn("criminal profiling", source)
            self.assertEqual(evidence["charts"][0]["source_ids"], ["S1"])

    def test_method_task_map_falls_back_when_graphviz_is_unavailable(self):
        evidence = sample_evidence()
        evidence["datasets"] = []
        evidence["evidence_map"] = [
            {"method": "Neural network", "task": "criminal profiling",
             "count": 1, "source_ids": ["S1"]},
        ]
        with tempfile.TemporaryDirectory() as folder:
            with patch("ai_scientist.research_evidence.render_method_task_map",
                       return_value=False):
                build_charts(evidence, folder)
            self.assertEqual(evidence["charts"][0]["renderer"], "matplotlib")
            self.assertTrue((Path(folder) / "chart_01.png").is_file())
            self.assertFalse((Path(folder) / "chart_01.gv").exists())
            self.assertTrue(any("Graphviz renderer unavailable" in warning
                                for warning in evidence["warnings"]))

    def test_listing_style_is_replaced_by_comparative_fallback(self):
        evidence = sample_evidence()
        evidence["datasets"] = []
        evidence["sources"].append({"id": "S2", "title": "Neural networks for identity resolution",
                                    "year": 2025, "abstract": "Identity resolution uses neural networks."})
        evidence["source_profiles"] = [
            {"source_id": "S1", "methods": ["Machine learning"], "tasks": ["forensic assessment"],
             "risk_passage": None, "validation_passage": None},
            {"source_id": "S2", "methods": ["Neural network"], "tasks": ["identity resolution"],
             "risk_passage": None, "validation_passage": None},
        ]
        listing = {"related_work": "One study uses machine learning [S1]. Another uses neural networks [S2].",
                   "abstract": "No validated quantitative result was extracted from the retrieved abstracts.",
                   "discussion": "World Bank data are relevant [S1]."}
        with patch("ai_scientist.perform_writeup.get_response_from_llm",
                   return_value=(json.dumps(listing), [])):
            sections = _draft_sections({"Title": "AI review", "research_question": "How?"},
                                       evidence, object())
        self.assertIn("whereas", sections["related_work"])
        self.assertNotIn("No validated quantitative", sections["abstract"])
        self.assertNotIn("World Bank", sections["discussion"])

    def test_analytical_prose_requires_verbatim_support_from_both_sources(self):
        evidence = sample_evidence()
        evidence["datasets"] = []
        evidence["sources"] = [
            {"id": "S1", "title": "Survey of learning", "abstract":
             "A survey study examined student achievement in urban schools."},
            {"id": "S2", "title": "Cohort of learning", "abstract":
             "A cohort study examined student achievement across several years."},
        ]
        text = ("The survey examines urban schools [S1], whereas the cohort follows "
                "achievement over time [S2]; their designs answer different questions.")
        good = {"text": text, "support": [
            {"source_id": "S1", "quote": evidence["sources"][0]["abstract"]},
            {"source_id": "S2", "quote": evidence["sources"][1]["abstract"]},
        ]}
        bad = {"text": text, "support": [
            {"source_id": "S1", "quote": "An invented claim about urban schools."},
            {"source_id": "S2", "quote": evidence["sources"][1]["abstract"]},
        ]}
        with patch("ai_scientist.perform_writeup.get_response_from_llm",
                   return_value=(json.dumps({"analysis_claims": [bad, good]}), [])):
            sections = _draft_sections({"Title": "Education review",
                                        "research_question": "What differs?"}, evidence, object())
        self.assertEqual(len(sections["analysis_claims"]), 1)
        self.assertEqual(sections["analysis_claims"][0]["text"], text)

    @unittest.skipUnless(shutil.which("pdflatex"), "pdflatex unavailable")
    def test_ai_forensic_review_uses_narrative_structure_and_method_map(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = sample_evidence()
            evidence["topic"] = "AI in criminal psychological assessment"
            evidence["datasets"] = []
            evidence["sources"] = [
                {"id": "S1", "title": "Machine learning for forensic assessment", "year": 2024,
                 "abstract": "Machine learning was used in forensic assessment.",
                 "doi": "10.example/one", "url": "https://doi.org/10.example/one"},
                {"id": "S2", "title": "Multilayer perceptron for criminal profiling", "year": 2025,
                 "abstract": "A multilayer perceptron was used for criminal profiling.",
                 "doi": "10.example/two", "url": "https://doi.org/10.example/two"},
            ]
            plan = {"Title": "AI forensic evidence review", "research_question": "How is AI used?",
                    "search_queries": ["AI forensic psychology"],
                    "outcome_terms": ["forensic assessment", "criminal profiling"]}
            evidence["source_profiles"] = profile_sources(evidence["sources"], plan)
            evidence["evidence_map"] = evidence_map(evidence["source_profiles"])
            build_charts(evidence, folder)
            (Path(folder) / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
            pdf = perform_writeup(plan, None, folder)
            tex = (Path(folder) / "paper.tex").read_text(encoding="utf-8")
            self.assertIn("Results: narrative synthesis", tex)
            self.assertIn(r"\section{Results: narrative synthesis}", tex)
            self.assertIn(r"\textit{Figure 1 analysis.}", tex)
            self.assertLess(tex.index("not interchangeable evaluation tasks"),
                            tex.index(r"\section{Evidence figures}"))
            self.assertIn("not interchangeable evaluation tasks", tex)
            self.assertIn("whereas", tex)
            self.assertIn("chart_01.png", tex)
            self.assertEqual(tex.count("did not yield validated, comparable numerical outcomes"), 1)
            self.assertNotIn("Public data series", tex)
            self.assertNotIn("World Bank", tex)
            self.assertNotIn("publication years", tex.casefold())
            self.assertTrue(Path(pdf).is_file())
            pdf_text = " ".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
            self.assertIn("Figure 1 analysis", pdf_text)
            self.assertIn("performance ranking", pdf_text)

    def test_analysis_interprets_reversal_and_keeps_source_trace(self):
        evidence = sample_evidence()
        evidence["datasets"][0]["points"] = [
            {"year": 2010, "value": 100}, {"year": 2018, "value": 150},
            {"year": 2019, "value": 155}, {"year": 2022, "value": 160},
            {"year": 2024, "value": 120},
        ]
        analysis = {"series": [analyze_series(evidence["datasets"][0])]}
        interpretation = build_interpretation(evidence, analysis, {"findings": [], "comparisons": []},
                                              {"outcome_terms": []})
        observation = next(item for item in interpretation["observations"]
                           if item["kind"] == "series_interpretation")
        self.assertIn("recent reversal", observation["text"])
        self.assertEqual(observation["source_ids"], ["D1"])
        self.assertIn("[D1]", observation["text"])

    def test_each_chart_analysis_uses_only_its_plotted_evidence(self):
        evidence = sample_evidence()
        evidence["datasets"][0]["points"] = [
            {"year": 2010, "value": 90}, {"year": 2019, "value": 130},
            {"year": 2020, "value": 120},
            {"year": 2023, "value": 140}, {"year": 2024, "value": 110},
        ]
        findings = [
            {"id": "F1", "source_id": "S1", "measure": "odds_ratio", "estimate": 0.8,
             "exposure": "temperature", "outcome": "mood", "contrast": "per degree",
             "unit": "ratio", "ci_low": 0.7, "ci_high": 0.9},
            {"id": "F2", "source_id": "S2", "measure": "odds_ratio", "estimate": 1.2,
             "exposure": "temperature", "outcome": "mood", "contrast": "per degree",
             "unit": "ratio", "ci_low": 0.9, "ci_high": 1.5},
        ]
        evidence["study_findings"] = findings
        evidence["evidence_map"] = [
            {"method": "Survey", "task": "mood", "count": 1, "source_ids": ["S1"]},
            {"method": "Cohort", "task": "mood", "count": 1, "source_ids": ["S2"]},
        ]
        with tempfile.TemporaryDirectory() as folder:
            build_charts(evidence, folder)
            analyses = analyze_charts(evidence,
                                      {"series": [analyze_series(evidence["datasets"][0])]},
                                      analyze_findings(findings))
        self.assertEqual([item["kind"] for item in analyses],
                         ["published_data", "study_estimates", "method_task_map"])
        self.assertEqual([item["file"] for item in analyses],
                         [chart["file"] for chart in evidence["charts"]])
        self.assertIn("latest available five-year window", analyses[0]["text"])
        self.assertIn("[D1]", analyses[0]["text"])
        self.assertNotIn("[S1]", analyses[0]["text"])
        self.assertIn("opposite sides", analyses[1]["text"])
        self.assertIn("1 displayed interval", analyses[1]["text"])
        self.assertEqual(analyses[1]["source_ids"], ["S1", "S2"])
        self.assertIn("2 distinct methods", analyses[2]["text"])
        self.assertIn("not accuracy", analyses[2]["text"])

    def test_qualitative_analysis_compares_methods_and_validation_visibility(self):
        evidence = sample_evidence()
        evidence["datasets"] = []
        evidence["sources"] = [
            {"id": "S1", "title": "A survey of student achievement", "abstract":
             "A survey study of student achievement used external validation. "
             "A limitation is a small sample."},
            {"id": "S2", "title": "A cohort study of student achievement", "abstract":
             "A cohort study examines student achievement over time."},
        ]
        plan = {"outcome_terms": ["student achievement"]}
        evidence["source_profiles"] = profile_sources(evidence["sources"], plan)
        self.assertIn("limitation", evidence["source_profiles"][0]["risk_passage"]["quote"].lower())
        evidence["evidence_map"] = evidence_map(evidence["source_profiles"])
        interpretation = build_interpretation(evidence, {"series": []},
                                              {"findings": [], "comparisons": []}, plan)
        kinds = {item["kind"] for item in interpretation["observations"]}
        self.assertIn("method_comparison", kinds)
        self.assertIn("validation_appraisal", kinds)
        self.assertIn("limitations_appraisal", kinds)
        self.assertTrue(all(set(item["source_ids"]) <= {"S1", "S2"}
                            for item in interpretation["observations"]))
        self.assertIn("not proof", " ".join(item["text"]
                                           for item in interpretation["observations"]))

    def test_risk_ratio_and_external_validation_are_not_mislabeled_as_limitations(self):
        source = {"id": "S1", "title": "Cohort study of outcomes",
                  "abstract": "The risk ratio was estimated after external validation."}
        profile = profile_sources([source], {"outcome_terms": ["outcomes"]})[0]
        self.assertIsNone(profile["risk_passage"])
        self.assertIsNotNone(profile["validation_passage"])

    def test_analysis_calculates_changes_extrema_and_recent_reversal(self):
        dataset = sample_evidence()["datasets"][0]
        dataset["points"] = [
            {"year": 2010, "value": 100}, {"year": 2018, "value": 150},
            {"year": 2019, "value": 155}, {"year": 2022, "value": 160},
            {"year": 2024, "value": 120},
        ]
        result = analyze_series(dataset)
        self.assertEqual(result["net_change"], 20)
        self.assertEqual(result["percent_change"], 20)
        self.assertAlmostEqual(result["average_net_change_per_year"], 20 / 14)
        self.assertEqual(result["minimum"], {"year": 2010, "value": 100})
        self.assertEqual(result["maximum"], {"year": 2022, "value": 160})
        self.assertEqual(result["consecutive_year_pairs"], 1)
        self.assertIsNone(result["largest_one_year_decrease"])
        self.assertEqual(result["recent_five_year_window"]["net_change"], -35)
        self.assertIn("opposite to the full-span change", describe_analysis(result))
        self.assertIn("net increases in 1 series", summarize_analysis({"series": [result]}))

    def test_analysis_omits_undefined_percentage_and_does_not_fill_gaps(self):
        dataset = sample_evidence()["datasets"][0]
        dataset["points"] = [{"year": 2010, "value": 0}, {"year": 2020, "value": 10}]
        result = analyze_series(dataset)
        self.assertIsNone(result["percent_change"])
        self.assertEqual(result["consecutive_year_pairs"], 0)
        self.assertIsNone(result["recent_five_year_window"])
        self.assertIn("cannot be assessed", describe_analysis(result))

    def test_analysis_identifies_largest_observed_one_year_decline(self):
        dataset = sample_evidence()["datasets"][0]
        dataset["points"] = [{"year": 2020, "value": 100},
                             {"year": 2021, "value": 90},
                             {"year": 2022, "value": 95},
                             {"year": 2023, "value": 80}]
        result = analyze_series(dataset)
        self.assertEqual(result["largest_one_year_decrease"],
                         {"from_year": 2022, "to_year": 2023, "change": -15})
        self.assertIn("largest observed one-year decline was 15 people between 2022 and 2023",
                      describe_analysis(result))

    def test_plan_rejects_untrusted_dataset_identifiers(self):
        payload = {"Title": "Population systematic review", "research_question": "How has population changed?",
                   "search_queries": ["population trend", "World population"],
                   "exposure_terms": ["population"], "outcome_terms": ["affect", "mood"],
                   "world_bank_indicators": ["SP.POP.TOTL", "../../secret"],
                   "owid_charts": ["life-expectancy", "https://private/"],
                   "country_code": "WLD", "entity": "World"}
        with tempfile.TemporaryDirectory() as folder:
            with patch("ai_scientist.generate_ideas.get_response_from_llm",
                       return_value=(json.dumps(payload), [])):
                plan = generate_ideas("World population", object(), folder)
            self.assertEqual(plan["world_bank_indicators"], ["SP.POP.TOTL"])
            self.assertEqual(plan["owid_charts"], ["life-expectancy"])
            self.assertEqual(plan["Title"], "Population evidence review")
            self.assertEqual(plan["outcome_terms"], ["mood"])
            self.assertTrue((Path(folder) / "research_plan.json").is_file())

    def test_balance_scale_topic_stays_on_the_named_benchmark(self):
        misleading = {
            "Title": "Economic equilibrium learning",
            "research_question": "Can models forecast economic equilibrium?",
            "search_queries": ["algorithmic learning of economic equilibrium"],
            "exposure_terms": ["neural networks"],
            "outcome_terms": ["steady-state prediction accuracy"],
            "focus_terms": ["equilibrium"],
            "world_bank_indicators": ["SP.POP.TOTL"],
        }
        with tempfile.TemporaryDirectory() as folder:
            with patch("ai_scientist.generate_ideas.get_response_from_llm",
                       return_value=(json.dumps(misleading), [])):
                plan = generate_ideas(
                    "Đánh giá khả năng học quy luật cân bằng của các thuật toán Machine Learning",
                    object(), folder)
        self.assertEqual(plan["focus_terms"][0], "Balance Scale")
        self.assertEqual(plan["exposure_terms"], ["balance scale"])
        self.assertTrue(all("balance scale" in query.casefold()
                            for query in plan["search_queries"]))
        self.assertEqual(plan["world_bank_indicators"], [])
        records = [
            {"title": "Berg Balance Scale model for clinical assessment", "abstract": "",
             "providers": ["OpenAlex"], "url": "https://example.org/clinical"},
            {"title": "Learning on the balance scale task", "abstract": "",
             "providers": ["Crossref"], "url": "https://example.org/task"},
        ]
        self.assertEqual([item["title"] for item in rank_sources(records, plan)],
                         ["Learning on the balance scale task"])

    def test_public_data_is_read_from_source_responses(self):
        bank = fetch_world_bank("SP.POP.TOTL", "WLD", FakeSession())
        owid = fetch_owid_chart("life-expectancy", "World", FakeSession())
        self.assertEqual([p["value"] for p in bank["points"]], [100, 110])
        self.assertEqual([p["value"] for p in owid["points"]], [72, 73])
        self.assertEqual(owid["credit"], "UN data")

    def test_multimetric_chart_labels_the_actual_plotted_metric(self):
        class MultiMetricSession(FakeSession):
            def get(self, url, **kwargs):
                if url.endswith(".metadata.json"):
                    return FakeResponse(url, {"chart": {"title": "Life expectancy vs. GDP per capita"},
                                              "columns": {"GDP per capita": {"type": "Numeric", "unit": "USD"},
                                                          "Life expectancy": {"type": "Numeric", "unit": "years"}}})
                if url.endswith(".csv"):
                    return FakeResponse(url, text="Entity,Year,GDP per capita,Life expectancy\nWorld,2022,1000,72\nWorld,2023,1100,73\n")
                raise AssertionError(url)

        dataset = fetch_owid_chart("life-expectancy-vs-gdp-per-capita", "World", MultiMetricSession())
        self.assertEqual(dataset["title"], "GDP per capita")
        self.assertEqual(dataset["chart_title"], "Life expectancy vs. GDP per capita")
        self.assertEqual(dataset["unit"], "USD")

    def test_scholarly_metadata_includes_authors_and_venue(self):
        class SourceSession(FakeSession):
            def get(self, url, **kwargs):
                return FakeResponse(url, {"results": [{
                    "display_name": "A population study", "publication_year": 2024,
                    "authorships": [{"author": {"display_name": "Jane Doe"}}],
                    "primary_location": {"source": {"display_name": "Journal of Evidence"}},
                }]})

        source = search_openalex("population", SourceSession())[0]
        self.assertEqual(source["authors"], ["Jane Doe"])
        self.assertEqual(source["venue"], "Journal of Evidence")

    def test_collect_evidence_exports_reproducible_chart_data(self):
        source = sample_evidence()["sources"][0].copy()
        source.pop("id")
        plan = {"topic": "World population", "search_queries": ["world population"],
                "world_bank_indicators": ["SP.POP.TOTL"], "owid_charts": [],
                "country_code": "WLD", "entity": "World"}
        with tempfile.TemporaryDirectory() as folder:
            with patch("ai_scientist.research_evidence.search_openalex", return_value=[source]), \
                 patch("ai_scientist.research_evidence.search_crossref", return_value=[source]), \
                 patch("ai_scientist.research_evidence.search_semantic_scholar", return_value=[]):
                evidence = collect_evidence(plan, folder, FakeSession(), retrieve_fulltext=False)
            self.assertEqual(len(evidence["sources"]), 1)
            self.assertEqual(evidence["datasets"][0]["id"], "D1")
            self.assertEqual(evidence["analysis"]["series"][0]["net_change"], 10)
            self.assertEqual(len(evidence["charts"]), 1)
            with open(Path(folder) / "chart_data.csv", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["value"] for r in rows if r["source_id"] == "D1"], ["100.0", "110.0"])
            self.assertTrue((Path(folder) / "evidence.json").is_file())

    def test_collect_evidence_rejects_unrelated_valid_series(self):
        source = sample_evidence()["sources"][0].copy()
        source.pop("id")
        source["title"] = "Machine learning for criminal profiling"
        source["abstract"] = "Machine learning methods for criminal profiling in forensic settings."
        plan = {"topic": "Machine learning for criminal profiling",
                "search_queries": ["machine learning criminal profiling"],
                "exposure_terms": ["machine learning"],
                "outcome_terms": ["criminal profiling"],
                "focus_terms": ["forensic"],
                "world_bank_indicators": ["SP.POP.TOTL"], "owid_charts": [],
                "country_code": "WLD", "entity": "World"}
        with tempfile.TemporaryDirectory() as folder:
            with patch("ai_scientist.research_evidence.search_openalex", return_value=[source]), \
                 patch("ai_scientist.research_evidence.search_crossref", return_value=[]), \
                 patch("ai_scientist.research_evidence.search_semantic_scholar", return_value=[]):
                evidence = collect_evidence(plan, folder, FakeSession(), retrieve_fulltext=False)
        self.assertEqual(evidence["datasets"], [])
        self.assertTrue(any("did not match the research topic" in warning
                            for warning in evidence["warnings"]))

    def test_latex_escapes_text(self):
        self.assertEqual(clean_text_for_latex("A_B & 50%"), r"A\_B \& 50\%")

    @unittest.skipUnless(shutil.which("pdflatex"), "pdflatex unavailable")
    def test_pdf_contains_references_and_chart_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = sample_evidence()
            build_charts(evidence, folder)
            (Path(folder) / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
            prose = {key: "This review describes the retrieved evidence [S1]." for key in
                     ("abstract", "introduction", "related_work", "discussion", "conclusion")}
            with patch("ai_scientist.perform_writeup.get_response_from_llm",
                       return_value=(json.dumps(prose), [])):
                pdf = perform_writeup({"Title": "World population review",
                                       "research_question": "How has population changed?",
                                       "search_queries": ["world population"]}, object(), folder)
            tex = (Path(folder) / "paper.tex").read_text(encoding="utf-8")
            self.assertIn("[S1]", tex)
            self.assertIn("[D1]", tex)
            self.assertIn("https://doi.org/10.1234/demo", tex)
            self.assertIn("chart_01.png", tex)
            self.assertIn(r"\begin{minipage}{\linewidth}", tex)
            self.assertLess(tex.index("chart_01.png"), tex.index(r"\section{Discussion and limitations}"))
            self.assertIn("Quantitative results", tex)
            self.assertIn("Across 1 validated numerical series", tex)
            self.assertIn("net increase of 10 people", tex)
            self.assertIn(r"net increase was 10.00\%", tex)
            self.assertIn("The series comparisons are descriptive", tex)
            self.assertIn(r"\textit{Figure 1 analysis.}", tex)
            self.assertIn("few observed points", tex)
            self.assertLess(tex.index("few observed points"),
                            tex.index(r"\section{Evidence figures}"))
            self.assertLess(tex.index("chart_01.png"), tex.index("Figure 1 analysis."))
            saved_evidence = json.loads((Path(folder) / "evidence.json").read_text(encoding="utf-8"))
            self.assertTrue(saved_evidence["interpretation"]["observations"])
            self.assertEqual(saved_evidence["chart_analyses"][0]["source_ids"], ["D1"])
            self.assertTrue(Path(pdf).is_file())
            pdf_text = " ".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
            self.assertIn("Figure 1 analysis", pdf_text)
            self.assertIn("few observed points", pdf_text)

    @unittest.skipUnless(shutil.which("pdflatex"), "pdflatex unavailable")
    def test_future_topic_without_numerical_data_states_the_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = sample_evidence()
            evidence["topic"] = "Ethics in artificial intelligence"
            evidence["datasets"] = []
            build_charts(evidence, folder)
            (Path(folder) / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
            plan = {"Title": "Ethics in artificial intelligence", "research_question": "What is known?",
                    "search_queries": ["AI ethics"]}
            with patch("ai_scientist.perform_writeup.get_response_from_llm",
                       return_value=("{}", [])):
                pdf = perform_writeup(plan, object(), folder)
            tex = (Path(folder) / "paper.tex").read_text(encoding="utf-8")
            self.assertIn("Results: narrative synthesis", tex)
            self.assertEqual(tex.count("did not yield validated, comparable numerical outcomes"), 1)
            self.assertNotIn("Quantitative results", tex)
            self.assertNotIn("World Bank", tex)
            self.assertNotIn(r"\section{Evidence figures}", tex)
            self.assertTrue(Path(pdf).is_file())

    @unittest.skipUnless(shutil.which("pdflatex"), "pdflatex unavailable")
    def test_pdf_includes_verified_experiment_and_review_fallback(self):
        plan = {"Title": "Population review", "topic": "World population",
                "research_question": "Can a trend forecast improve prediction?",
                "search_queries": ["world population"]}
        for feasible in (True, False):
            with self.subTest(feasible=feasible), tempfile.TemporaryDirectory() as folder:
                evidence = sample_evidence()
                if feasible:
                    evidence["datasets"][0]["points"] = [
                        {"year": 2010 + index, "value": 100 + 2 * index}
                        for index in range(12)]
                build_charts(evidence, folder)
                plan_and_run(plan, evidence, folder)
                pdf = perform_writeup(plan, None, folder)
                tex = (Path(folder) / "paper.tex").read_text(encoding="utf-8")
                pdf_text = " ".join(page.extract_text() or ""
                                    for page in PdfReader(pdf).pages)
                self.assertIn("Proposed research idea", pdf_text)
                if feasible:
                    self.assertIn("Computational experiment on published data", pdf_text)
                    self.assertIn("experiment_plot.png", tex)
                    self.assertIn("mean absolute error", pdf_text)
                    self.assertIn("[D1]", pdf_text)
                else:
                    self.assertNotIn("Computational experiment on published data", pdf_text)
                    self.assertIn("not tested in this run", pdf_text)
                    self.assertNotIn("experiment_plot.png", tex)

    def test_relevance_screen_requires_both_exposure_and_outcome(self):
        sources = [
            {"title": "Ambient temperature and mood", "abstract": "Temperature affects mood."},
            {"title": "Emotion recognition methods", "abstract": "Sensors measure mood."},
            {"title": "Temperature and crops", "abstract": "Heat affects crop yield."},
            {"title": "Ambient temperature affect beetles", "abstract": "Copulation duration changes."},
            {"title": "Table 1: ambient temperature and mood", "abstract": ""},
            {"title": "The influence of light on thermal responses",
             "abstract": "Ambient temperature and mood are mentioned, but light is the exposure."},
        ]
        plan = {"exposure_terms": ["temperature", "heat"],
                "outcome_terms": ["affect", "mood", "emotion"]}
        selected = rank_sources(sources, plan)
        self.assertEqual([s["title"] for s in selected], ["Ambient temperature and mood"])

    def test_relevance_screen_keeps_title_exposure_with_abstract_outcome(self):
        sources = [
            {"title": "Machine learning for forensic investigations",
             "abstract": "The study evaluates criminal profiling with machine learning.",
             "oa_pdf_url": "https://example.org/study.pdf"},
            {"title": "Machine learning for criminal profiling",
             "abstract": "An evaluation of the method."},
        ]
        plan = {"exposure_terms": ["machine learning"],
                "outcome_terms": ["criminal profiling"], "focus_terms": ["forensic"]}
        selected = rank_sources(sources, plan)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["title"], "Machine learning for criminal profiling")

    def test_abstract_estimate_is_verified_and_sample_count_rejected(self):
        source = {"id": "S1", "title": "Ambient temperature and mood", "url": "https://doi.org/10.example/a",
                  "abstract": "Higher ambient temperature was associated with a 12% increase in negative mood."}
        candidate = {"quote": source["abstract"], "estimate_text": "12%", "measure": "percentage",
                     "exposure": "ambient temperature", "outcome": "negative mood"}
        item = validate_finding(candidate, source)
        self.assertEqual(item["estimate"], 12)
        self.assertEqual(item["outcome"], "negative mood")
        self.assertIsNone(validate_finding(candidate | {"estimate_text": "30%"}, source))
        self.assertIsNone(validate_finding(candidate | {"estimate_text": "12"}, source))
        study = source | {"abstract": "A total of 906 participants reported their mood."}
        self.assertIsNone(validate_finding({"quote": study["abstract"], "estimate_text": "906",
                                            "measure": "other"}, study))

    def test_extraction_fallback_and_descriptive_comparison(self):
        sources = [
            {"id": "S1", "title": "Temperature and mood A", "url": "https://doi.org/10.example/a",
             "abstract": "Higher ambient temperature per 1 degree Celsius was associated with an increase of 12% in negative mood."},
            {"id": "S2", "title": "Temperature and mood B", "url": "https://doi.org/10.example/b",
             "abstract": "Higher ambient temperature per 1 degree Celsius was associated with an increase of 8% in negative mood."},
        ]
        findings, warning = extract_study_findings(
            sources, model=None,
            plan={"exposure_terms": ["ambient temperature"],
                  "outcome_terms": ["negative mood"]})
        self.assertIsNone(warning)
        self.assertEqual([item["estimate"] for item in findings], [12, 8])
        self.assertEqual(analyze_findings(findings)["comparisons"], [])
        for item in findings:
            item["exposure"] = "ambient temperature"
            item["outcome"] = "negative mood"
            item["contrast"] = "per 1 degree Celsius"
            item["unit"] = "%"
        comparison = analyze_findings(findings)["comparisons"][0]
        self.assertEqual((comparison["minimum"], comparison["maximum"], comparison["median"]), (8, 12, 10))

    def test_collect_evidence_accepts_only_indexed_verbatim_llm_result(self):
        source = {"title": "Ambient temperature and mood", "year": 2024,
                  "doi": "10.example/mood", "url": "https://doi.org/10.example/mood",
                  "abstract": "A higher ambient temperature was associated with a 12% increase in negative mood.",
                  "providers": ["OpenAlex"], "query": "temperature mood"}
        plan = {"topic": "temperature and mood", "search_queries": ["temperature mood"],
                "exposure_terms": ["ambient temperature"], "outcome_terms": ["mood"],
                "world_bank_indicators": [], "owid_charts": []}
        proposed = {"findings": [{"source_id": "S1", "passage_index": 0,
                                  "estimate_text": "12%", "measure": "percentage",
                                  "exposure": "ambient temperature", "outcome": "negative mood"},
                                 {"source_id": "S1", "passage_index": 0,
                                  "estimate_text": "90%", "measure": "percentage"}]}
        with tempfile.TemporaryDirectory() as folder:
            with patch("ai_scientist.research_evidence.search_openalex", return_value=[source]), \
                 patch("ai_scientist.research_evidence.search_crossref", return_value=[]), \
                 patch("ai_scientist.research_evidence.search_semantic_scholar", return_value=[]), \
                 patch("ai_scientist.study_evidence.get_response_from_llm",
                       return_value=(json.dumps(proposed), [])):
                evidence = collect_evidence(plan, folder, FakeSession(), model=object(),
                                            retrieve_fulltext=False)
            self.assertEqual(len(evidence["study_findings"]), 1)
            self.assertEqual(evidence["study_findings"][0]["estimate"], 12)
            self.assertEqual(evidence["study_analysis"]["comparisons"], [])
            self.assertTrue((Path(folder) / "study_findings.csv").is_file())
            self.assertTrue(any(chart["kind"] == "study_estimates" for chart in evidence["charts"]))

    def test_temperature_mood_regression_survives_gemini_extraction_failure(self):
        abstract = ("RESULTS: Overall, the probability of having a bad mood for the entire day "
                    "decreased by 7.0% (OR: 0.93: 95% CI 0.88, 0.99) for each 5 °C increase "
                    "in maximum temperature. A smaller effect (-3%; OR: 0.97) was found when "
                    "controlling for sunshine duration.")
        source = {"id": "S1", "title": "How ambient temperature affects mood",
                  "url": "https://doi.org/example", "abstract": abstract}
        plan = {"exposure_terms": ["ambient temperature", "outdoor temperature"],
                "outcome_terms": ["mood", "emotional state"]}
        with patch("ai_scientist.study_evidence.get_response_from_llm",
                   side_effect=RuntimeError("503 unavailable")):
            findings, warning = extract_study_findings([source], object(), plan=plan)
        self.assertIn("503 unavailable", warning)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["estimate"], 0.93)
        self.assertEqual(findings[0]["ci_low"], 0.88)
        self.assertEqual(findings[0]["contrast"], "for each 5 °C increase")
        self.assertEqual(findings[0]["exposure"], "temperature")

    @unittest.skipUnless(shutil.which("pdflatex"), "pdflatex unavailable")
    def test_pdf_writes_verified_study_estimate_with_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = sample_evidence()
            evidence["datasets"] = []
            quote = "Higher ambient temperature was associated with a 12% increase in negative mood."
            evidence["sources"][0].update(title="Ambient temperature and mood", abstract=quote)
            finding = validate_finding({"quote": quote, "estimate_text": "12%",
                                        "measure": "percentage", "exposure": "ambient temperature",
                                        "outcome": "negative mood"}, evidence["sources"][0])
            finding["id"] = "F1"
            evidence["study_findings"] = [finding]
            evidence["study_analysis"] = analyze_findings([finding])
            build_charts(evidence, folder)
            (Path(folder) / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
            plan = {"Title": "Temperature and mood", "research_question": "How does heat affect mood?",
                    "search_queries": ["temperature mood"]}
            with patch("ai_scientist.perform_writeup.get_response_from_llm", return_value=("{}", [])):
                pdf = perform_writeup(plan, object(), folder)
            tex = (Path(folder) / "paper.tex").read_text(encoding="utf-8")
            self.assertIn("12\\%", tex)
            self.assertIn("ambient temperature and negative mood [S1]", tex)
            self.assertIn(r"\textit{Figure 1 analysis.}", tex)
            self.assertIn("Only one work is represented", tex)
            self.assertLess(tex.index("Only one work is represented"),
                            tex.index(r"\section{Discussion and limitations}"))
            self.assertNotIn("Public data series", tex)
            self.assertNotIn("cannot make a quantitative finding", tex)
            self.assertTrue(Path(pdf).is_file())


if __name__ == "__main__":
    unittest.main()
