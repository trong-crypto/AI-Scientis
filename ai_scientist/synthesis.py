"""Traceable qualitative profiles and a conservative method-to-task evidence map."""

import re
from collections import defaultdict


METHOD_PATTERNS = {
    "Multilayer perceptron": r"\b(?:multilayer perceptron|multi-layer perceptron|MLP)\b",
    "LSTM": r"\b(?:long short-term memory|LSTM)\b",
    "Neural network": r"\b(?:neural networks?|artificial neural networks?)\b",
    "Random forest": r"\brandom forests?\b",
    "Support vector machine": r"\b(?:support vector machines?|SVM)\b",
    "Logistic regression": r"\blogistic regressions?\b",
    "Transformer": r"\btransformers?\b",
    "Natural language processing": r"\b(?:natural language processing|NLP)\b",
    "Deep learning": r"\bdeep learning\b",
    "Machine learning": r"\bmachine learning\b",
    "Randomized trial": r"\b(?:randomi[sz]ed (?:controlled )?trial|randomi[sz]ed experiment)\b",
    "Cohort study": r"\b(?:cohort study|longitudinal cohort)\b",
    "Cross-sectional study": r"\b(?:cross.sectional (?:study|survey|analysis))\b",
    "Case study": r"\bcase stud(?:y|ies)\b",
    "Qualitative interview": r"\b(?:qualitative interviews?|semi.structured interviews?)\b",
    "Survey": r"\b(?:questionnaire survey|survey study|survey data)\b",
    "Regression analysis": r"\b(?:regression analysis|linear regression|multivariable regression)\b",
    "Time-series analysis": r"\btime.series analys(?:is|es)\b",
    "Systematic review": r"\bsystematic review\b",
    "Meta-analysis": r"\bmeta.analysis\b",
    "Simulation": r"\b(?:simulation study|computer simulation)\b",
}
RISK = re.compile(r"\b(?:limitation|bias|imbalanc|poor generalizab|"
                  r"lack of external validat|small sample|overfit|data leakage|"
                  r"not representativ|uncertain|caution|not generaliz)\w*\b", re.I)
VALIDATION = re.compile(r"\b(?:cross.validat\w*|external validat\w*|hold.out|test set|"
                        r"validation cohort|train(?:ing)?.test|independent sample)\b", re.I)


def _match(text, phrase):
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I))


def _segments(source):
    title = {"location": "title", "text": source.get("title", ""), "section": "Title"}
    abstract = {"location": "retrieved abstract", "text": source.get("abstract", ""),
                "section": "Abstract"}
    full = (source.get("fulltext") or {}).get("passages") or []
    return [title, abstract] + full


def _supported_excerpt(segments, pattern, preferred_sections=()):
    prioritized = sorted(segments, key=lambda segment: int(any(
        word in segment.get("section", "").casefold() for word in preferred_sections)),
        reverse=True)
    for segment in prioritized:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", segment.get("text", ""))
        for sentence in sentences:
            sentence = sentence.strip()
            if pattern.search(sentence) and 20 <= len(sentence) <= 500:
                return {"location": segment.get("location", ""), "quote": sentence}
    return None


def profile_sources(sources, plan):
    """Record method/task labels and explicit risk/validation passages per work."""
    tasks = [term for term in plan.get("outcome_terms", []) if isinstance(term, str)
             and len(term) >= 4 and term.casefold() not in {"affect", "effect", "impact"}]
    profiles = []
    for source in sources:
        segments = _segments(source)
        method_segments = [segment for segment in segments if segment["section"].casefold() in
                           ("title", "abstract") or "method" in segment["section"].casefold()
                           or segment.get("kind") == "pdf_text"]
        method_text = " ".join(segment["text"] for segment in method_segments)
        task_text = " ".join(segment["text"] for segment in segments[:2])
        methods = [label for label, pattern in METHOD_PATTERNS.items()
                   if re.search(pattern, method_text, re.I)]
        if len(methods) > 1:
            methods = [label for label in methods if label not in ("Machine learning", "Deep learning")]
        found_tasks = [task for task in tasks if _match(task_text, task)]
        risk = _supported_excerpt(segments, RISK, ("limitation", "discussion", "conclusion"))
        validation = _supported_excerpt(segments, VALIDATION, ("method", "result"))
        profiles.append({"source_id": source["id"], "title": source["title"],
                         "access_level": "full_text" if (source.get("fulltext") or {}).get("status") == "full_text"
                         else "abstract", "methods": methods, "tasks": found_tasks,
                         "risk_passage": risk, "validation_passage": validation})
    return profiles


def evidence_map(profiles):
    pairs = defaultdict(set)
    for profile in profiles:
        for method in profile["methods"]:
            for task in profile["tasks"]:
                pairs[(method, task)].add(profile["source_id"])
    return [{"method": method, "task": task, "source_ids": sorted(ids), "count": len(ids)}
            for (method, task), ids in sorted(pairs.items())]
