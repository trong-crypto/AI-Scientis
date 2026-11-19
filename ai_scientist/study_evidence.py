"""Screen scholarly sources and retain only traceable quantitative findings.

An LLM may propose extractions, but it cannot create evidence: every accepted
estimate and its verbatim passage must occur in a retrieved abstract or open text.
"""

import json
import math
import re
from collections import defaultdict

from ai_scientist.llm import extract_json_between_markers, get_response_from_llm


NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
EFFECT_CUES = re.compile(
    r"\b(?:odds ratio|risk ratio|relative risk|hazard ratio|correlation|"
    r"coefficient|beta|β|effect size|mean difference|difference|increase|decrease|"
    r"increased|decreased|higher|lower|associated|association|prevalence|"
    r"incidence|rate|probability|reduction|change|compared|versus|accuracy|"
    r"sensitivity|specificity|precision|recall|f1|auc|performance)\b",
    re.I,
)
MEASURES = {"odds_ratio", "risk_ratio", "hazard_ratio", "correlation",
            "beta", "mean_difference", "percentage", "rate", "other",
            "accuracy", "sensitivity", "specificity", "precision", "recall", "f1", "auc"}


def _contains_term(text, term):
    if not term:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", text.casefold()))


def _specific_terms(terms):
    return [term for term in terms if isinstance(term, str) and term.casefold() not in
            {"affect", "effect", "impact", "state", "conditions", "study"}]


def _matching_term(quote, terms):
    matches = [term for term in _specific_terms(terms) if _contains_term(quote, term)]
    if matches:
        return max(matches, key=len)
    # A result sentence may say "maximum temperature" while its study title and
    # plan use "ambient temperature". Retain the explicit shared measured noun.
    suffixes = [term.split()[-1] for term in _specific_terms(terms) if len(term.split()) > 1]
    suffixes = [term for term in suffixes if len(term) >= 5 and term not in
                {"state", "conditions", "response", "well-being", "exposure"}]
    matches = [term for term in suffixes if _contains_term(quote, term)]
    return max(matches, key=len) if matches else ""


def _has_effect_cue(text):
    return bool(EFFECT_CUES.search(text) or re.search(r"\b(?:OR|RR|HR)\s*[=:]", text))


def rank_sources(records, plan, limit=30):
    """Require both sides of a question when the plan supplies screening terms."""
    exposures = _specific_terms(plan.get("exposure_terms") or [])
    outcomes = _specific_terms(plan.get("outcome_terms") or [])
    focus_terms = [term for term in _specific_terms(plan.get("focus_terms") or [])
                   if len(term.split()) == 1]
    required_context = _specific_terms(plan.get("required_context_terms") or [])
    ranked = []
    for source in records:
        title = source.get("title", "")
        if re.match(r"^\s*(?:table|figure|fig\.)\s*\d+\s*[:.]", title, re.I):
            continue
        abstract = source.get("abstract", "")
        content = f"{title} {abstract}"
        if required_context and not any(_contains_term(content, term)
                                        for term in required_context):
            continue
        exposure_hit = any(_contains_term(content, term) for term in exposures)
        outcome_hit = any(_contains_term(content, term) for term in outcomes)
        if exposures and outcomes and not (exposure_hit and outcome_hit):
            continue
        title_exposure = any(_contains_term(title, term) for term in exposures)
        title_outcome = any(_contains_term(title, term) for term in outcomes)
        focus_hit = any(_contains_term(content, term) for term in focus_terms)
        if exposures and outcomes and not (title_exposure or title_outcome or focus_hit):
            continue
        title_hits = sum(_contains_term(title, term) for term in exposures + outcomes)
        body_hits = sum(_contains_term(abstract, term) for term in exposures + outcomes)
        ranked.append((int(title_exposure and title_outcome), int(focus_hit),
                       3 * title_hits + body_hits + bool(abstract) +
                       int(bool(source.get("oa_pdf_url"))), source))
    ranked.sort(key=lambda pair: (pair[0], pair[1], pair[2]), reverse=True)
    sources = [source for _, _, _, source in ranked[:limit]]
    for index, source in enumerate(sources, 1):
        source["id"] = f"S{index}"
    return sources


def _source_segments(source):
    fulltext = source.get("fulltext") or {}
    segments = []
    if fulltext.get("status") == "full_text":
        segments.extend(fulltext.get("passages") or [])
    if source.get("abstract"):
        segments.append({"section": "Abstract", "location": "retrieved abstract",
                         "kind": "abstract", "text": source["abstract"]})
    return segments


def _candidate_passages(source):
    """Select source-addressable result passages, including open full text."""
    candidates = []
    for segment in _source_segments(source):
        value = segment["text"]
        pieces = [value] if len(value) <= 900 else re.split(r"(?<=[.!?])\s+(?=[A-Z])", value)
        for piece in pieces:
            piece = piece.strip()
            if not (NUMBER.search(piece) and _has_effect_cue(piece) and len(piece) <= 900):
                continue
            section = segment.get("section", "").casefold()
            priority = (4 if "result" in section or "finding" in section else
                        3 if segment.get("kind") in ("table", "pdf_text") else
                        2 if segment.get("kind") == "abstract" else
                        1 if "discussion" in section else 0)
            candidates.append((priority, {"quote": piece,
                                          "location": segment["location"]}))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in candidates[:16]]


def _numeric_value(text):
    match = NUMBER.fullmatch(text.strip().rstrip("%"))
    if not match:
        return None
    value = float(match.group().replace(",", ""))
    return value if math.isfinite(value) else None


def _exact_number_span(text, number_text):
    match = re.search(r"(?<![\d.])" + re.escape(number_text) + r"(?![\d.%])", text)
    return match.start() if match else None


def validate_finding(candidate, source):
    """Reject invented quotes, values, measures, sample counts and p-values."""
    if not isinstance(candidate, dict):
        return None
    quote = candidate.get("quote")
    estimate_text = candidate.get("estimate_text")
    measure = candidate.get("measure")
    if not all(isinstance(value, str) for value in (quote, estimate_text, measure)):
        return None
    quote = quote.strip()
    estimate_text = estimate_text.strip()
    segments = _source_segments(source)
    position = _exact_number_span(quote, estimate_text)
    provenance = next((segment for segment in segments if quote in segment["text"]), None)
    if (not quote or provenance is None or position is None
            or measure not in MEASURES or not _has_effect_cue(quote)):
        return None
    value = _numeric_value(estimate_text)
    if value is None or (1900 <= value <= 2100 and measure == "other"):
        return None
    before = quote[max(0, position - 16):position].casefold()
    after = quote[position + len(estimate_text):position + len(estimate_text) + 22].casefold()
    if re.search(r"(?:\bp\s*[=<]|\bn\s*[=(]|sample\s+(?:of|size)\s*)$", before):
        return None
    if re.match(r"\s*(?:studies|participants|subjects|patients|papers|articles|people)\b", after):
        return None
    if estimate_text.endswith("%") and (re.search(r"\b(?:confidence|credible)\s*(?:interval|level)?\s*$", before)
                                       or re.match(r"\s*(?:CI|confidence|credible)\b", after, re.I)):
        return None
    if measure == "other" and not _has_effect_cue(quote[max(0, position - 50):position + len(estimate_text) + 50]):
        return None
    if measure == "percentage" and not estimate_text.endswith("%"):
        return None
    if measure in {"odds_ratio", "risk_ratio", "hazard_ratio"} and value <= 0:
        return None
    if measure == "correlation" and not -1 <= value <= 1:
        return None
    if measure in {"accuracy", "sensitivity", "specificity", "precision", "recall", "f1", "auc"}:
        if not (0 <= value <= (100 if estimate_text.endswith("%") else 1)):
            return None
        if not re.search(r"\b" + re.escape(measure) + r"\b", quote, re.I):
            return None
    if measure in {"odds_ratio", "risk_ratio", "hazard_ratio", "correlation", "beta"}:
        cues = {"odds_ratio": r"\b(?:odds ratio|OR)\b",
                "risk_ratio": r"\b(?:risk ratio|relative risk|RR)\b",
                "hazard_ratio": r"\b(?:hazard ratio|HR)\b",
                "correlation": r"\b(?:correlation|r\s*=)\b",
                "beta": r"\b(?:beta|coefficient)\b"}
        if not re.search(cues[measure], quote, re.I):
            return None
    ci_low = _numeric_value(str(candidate.get("ci_low", "")))
    ci_high = _numeric_value(str(candidate.get("ci_high", "")))
    if ci_low is None or ci_high is None or not (ci_low <= value <= ci_high):
        ci_low = ci_high = None
    elif not (_exact_number_span(quote, str(candidate["ci_low"])) is not None
              and _exact_number_span(quote, str(candidate["ci_high"])) is not None):
        ci_low = ci_high = None
    exposure = str(candidate.get("exposure", "")).strip()[:100]
    outcome = str(candidate.get("outcome", "")).strip()[:100]
    contrast = str(candidate.get("contrast", "")).strip()[:100]
    unit = str(candidate.get("unit", "")).strip()[:60]
    # Semantic labels are accepted only when explicitly present in the quote.
    if not exposure or not _contains_term(quote, exposure):
        exposure = "not specified in quoted passage"
    if not outcome or not _contains_term(quote, outcome):
        outcome = "not specified in quoted passage"
    if exposure.startswith("not specified") or outcome.startswith("not specified"):
        return None
    if not contrast or not _contains_term(quote, contrast):
        contrast = "not specified in quoted passage"
    if not unit or not _contains_term(quote, unit):
        unit = "not specified in quoted passage"
    return {"source_id": source["id"], "source_url": source.get("url", ""),
            "location": provenance["location"],
            "access_level": "full_text" if provenance.get("kind") != "abstract" else "abstract",
            "quote": quote,
            "estimate_text": estimate_text, "estimate": value,
            "measure": measure, "exposure": exposure, "outcome": outcome,
            "contrast": contrast, "unit": unit,
            "ci_low": ci_low, "ci_high": ci_high}


def _deterministic_candidates(source):
    """Limited no-LLM fallback for explicitly labelled ratios and percentages."""
    for passage in _candidate_passages(source):
        quote = passage["quote"]
        patterns = [
            ("odds_ratio", r"\b(?:odds ratio|OR)\s*[:=(]?\s*(\d+(?:\.\d+)?)"),
            ("risk_ratio", r"\b(?:risk ratio|relative risk|RR)\s*[:=(]?\s*(\d+(?:\.\d+)?)"),
            ("hazard_ratio", r"\b(?:hazard ratio|HR)\s*[:=(]?\s*(\d+(?:\.\d+)?)"),
            ("correlation", r"\b(?:correlation|r)\s*[:=]\s*(-?\d+(?:\.\d+)?)"),
            ("percentage", r"\b(?:increase|decrease|increased|decreased|higher|lower|reduction|change)\s+(?:of|by)\s+(\d+(?:\.\d+)?%)"),
        ]
        patterns.extend((metric, rf"\b{metric}\s*(?:of|was|=|:)\s*(\d+(?:\.\d+)?%?)")
                        for metric in ("accuracy", "sensitivity", "specificity", "precision", "recall", "f1", "auc"))
        for measure, pattern in patterns:
            match = re.search(pattern, quote, re.I)
            if match:
                candidate = {"quote": quote, "estimate_text": match.group(1),
                             "measure": measure}
                nearby = quote[match.end():match.end() + 70]
                interval = re.search(r"\b(?:95%\s*)?CI\s*[:=(]?\s*"
                                     r"(-?\d+(?:\.\d+)?)\s*[,;\-]\s*"
                                     r"(-?\d+(?:\.\d+)?)", nearby, re.I)
                if interval:
                    candidate["ci_low"], candidate["ci_high"] = interval.groups()
                contrast = re.search(r"\b(?:for\s+)?(?:each|per)\s+\d+(?:\.\d+)?\s*"
                                     r"(?:°\s?[CF]|degrees?\s+\w+|[A-Za-z%]+)\s+"
                                     r"(?:increase|decrease)", quote, re.I)
                if contrast:
                    candidate["contrast"] = contrast.group()
                yield candidate


def extract_study_findings(sources, model=None, max_sources=18, plan=None):
    candidates_by_source = {s["id"]: _candidate_passages(s)
                            for s in sources}
    selected = [s for s in sources if candidates_by_source[s["id"]]][:max_sources]
    proposed = []
    warning = None
    if selected and model is not None:
        packet = [{"source_id": s["id"], "passages": candidates_by_source[s["id"]][:6]}
                  for s in selected]
        prompt = (
            "Extract quantitative effect/outcome estimates relevant to the research question "
            "from these retrieved abstract or open-article passages. Return JSON object with key 'findings' "
            "(at most 12 objects). Each object: source_id, passage_index (zero-based index "
            "within that source's passages), estimate_text (EXACT number as printed, "
            "including % if present), "
            "measure (odds_ratio, risk_ratio, hazard_ratio, correlation, beta, "
            "mean_difference, percentage, rate, accuracy, sensitivity, specificity, "
            "precision, recall, f1, auc, other), exposure, outcome, contrast "
            "(the exact exposure increment or comparison), unit, ci_low, ci_high. "
            "Do not return sample sizes, study counts, dates, p-values, or confidence "
            "levels as estimates. Do not infer unseen results or convert units. If no "
            "usable estimate appears, return an empty list. Accessible passages:\n"
            + json.dumps(packet, ensure_ascii=False)
        )
        try:
            raw, _ = get_response_from_llm(prompt, model, temperature=0, max_retries=2)
            parsed = extract_json_between_markers(raw)
            proposed = parsed.get("findings", []) if isinstance(parsed, dict) else []
            if not isinstance(proposed, list):
                proposed = []
        except RuntimeError as exc:
            warning = f"Study estimate extraction unavailable: {exc}"
    by_id = {source["id"]: source for source in selected}
    findings = []
    seen = set()
    per_source = defaultdict(int)
    for candidate in proposed[:12] + [item | {"source_id": source["id"]}
                                     for source in selected for item in _deterministic_candidates(source)]:
        if not isinstance(candidate, dict):
            continue
        source = by_id.get(candidate.get("source_id"))
        if not source:
            continue
        passage_index = candidate.get("passage_index")
        if isinstance(passage_index, int) and not isinstance(passage_index, bool):
            passages = candidates_by_source[source["id"]]
            if 0 <= passage_index < len(passages):
                candidate = candidate | {"quote": passages[passage_index]["quote"]}
        quote = candidate.get("quote", "")
        if plan and isinstance(quote, str):
            for field, terms_key in (("exposure", "exposure_terms"),
                                     ("outcome", "outcome_terms")):
                if not _contains_term(quote, str(candidate.get(field, ""))):
                    match = _matching_term(quote, plan.get(terms_key, []))
                    if match:
                        candidate = candidate | {field: match}
        finding = validate_finding(candidate, source)
        if not finding:
            continue
        key = (finding["source_id"], finding["quote"])
        if key in seen or per_source[finding["source_id"]] >= 3:
            continue
        seen.add(key)
        per_source[finding["source_id"]] += 1
        finding["id"] = f"F{len(findings) + 1}"
        findings.append(finding)
    return findings, warning


def analyze_findings(findings):
    """Describe estimates; do not pool unlike studies or assume causality."""
    groups = defaultdict(list)
    for finding in findings:
        key = (finding["measure"], finding["exposure"].casefold(),
               finding["outcome"].casefold(), finding["contrast"].casefold(),
               finding["unit"].casefold())
        groups[key].append(finding)
    comparisons = []
    for (measure, exposure, outcome, contrast, unit), items in groups.items():
        if (len({item["source_id"] for item in items}) < 2 or any(value.startswith("not specified")
                                 for value in (exposure, outcome, contrast, unit))):
            continue
        values = sorted(item["estimate"] for item in items)
        comparisons.append({"measure": measure, "exposure": exposure, "outcome": outcome,
                            "contrast": contrast, "unit": unit,
                            "count": len(items), "minimum": values[0], "maximum": values[-1],
                            "median": values[len(values) // 2] if len(values) % 2 else
                            (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2,
                            "finding_ids": [item["id"] for item in items],
                            "source_ids": list(dict.fromkeys(item["source_id"] for item in items))})
    return {"method": "Verbatim estimates from accessible abstracts or open full text are reported with source links and locations. "
                      "Only estimates with the same measure, exposure/outcome labels, exposure "
                      "contrast and unit receive descriptive ranges and medians. No meta-analysis, "
                      "significance test or causal inference is performed.",
            "findings": findings, "comparisons": comparisons}
