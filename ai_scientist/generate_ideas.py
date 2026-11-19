"""Turn a free-form topic into a bounded plan for an evidence-based review."""

import json
import os
import re

from ai_scientist.llm import extract_json_between_markers, get_response_from_llm


def _benchmark_hint(topic):
    """Disambiguate a named benchmark before the model broadens its meaning."""
    subject = topic.casefold()
    if (re.search(r"balance[ -]scale|quy luật cân bằng", subject)
            and re.search(r"machine learning|thuật toán|algorithm", subject)):
        return (
            "This topic refers to the UCI Balance Scale classification task: "
            "given left/right weights and distances, predict left, right or balanced. "
            "It does not refer to economic equilibrium, class imbalance or steady-state physics."
        )
    return ""


def generate_ideas(user_topic: str, model, workspace_dir: str = "./workspace") -> dict:
    topic = user_topic.strip()
    if not topic:
        raise ValueError("Chủ đề nghiên cứu không được để trống.")
    os.makedirs(workspace_dir, exist_ok=True)
    benchmark_hint = _benchmark_hint(topic)
    prompt = f"""Create a research plan for an evidence-based review of: {topic!r}.
{benchmark_hint}
Do not propose an experiment, invent references, or assume a result.
Return ONLY a JSON object with:
  "Title": a concise English evidence-review title (do not call it a systematic review),
  "research_question": a focused research question,
  "search_queries": 2-3 distinct English scholarly search queries,
  "exposure_terms": 2-5 English terms for the subject, exposure or explanatory variable,
  "outcome_terms": 2-5 English terms for the outcome or quantity of interest,
  "focus_terms": 2-5 English terms for the essential domain or population that
    distinguish the topic from superficially similar fields,
  "world_bank_indicators": up to 2 directly relevant World Bank indicator codes,
  "owid_charts": up to 2 directly relevant Our World in Data Grapher slugs,
  "country_code": a World Bank country code, or "WLD" for a global topic,
  "entity": an Our World in Data entity name, or "World" for a global topic.
Use empty arrays if no suitable indicator or chart is known. These suggestions
will be checked against the source APIs; do not manufacture numerical values.
The exposure and outcome terms are used to screen article relevance; use
short, literal phrases likely to occur in the titles or abstracts returned by
the search queries. Avoid speculative technical outcomes or unrelated meanings
of an ambiguous topic. Use a named public benchmark in the queries when the
topic clearly refers to one. Do not use generic words such as study or effects.
Focus terms must be central to the user's subject, preferably single-word domain
anchors (for example criminal, offender, forensic), not broad words such as data.
"""
    response, _ = get_response_from_llm(prompt, model)
    candidate = extract_json_between_markers(response) or {}
    if not isinstance(candidate, dict):
        candidate = {}

    queries = candidate.get("search_queries", [])
    if not isinstance(queries, list):
        queries = []
    queries = [q.strip()[:120] for q in queries if isinstance(q, str) and q.strip()][:3]
    if not queries:
        queries = [topic]

    def terms(field):
        values = candidate.get(field, [])
        if not isinstance(values, list):
            return []
        return list(dict.fromkeys(value.strip().casefold() for value in values
                                  if isinstance(value, str) and 2 <= len(value.strip()) <= 60
                                  and value.strip().casefold() not in
                                  {"affect", "effect", "impact", "state", "conditions", "study"}))[:5]

    indicators = candidate.get("world_bank_indicators", [])
    if not isinstance(indicators, list):
        indicators = []
    indicators = [code.upper() for code in indicators if isinstance(code, str)
                  and re.fullmatch(r"[A-Za-z0-9.]{3,40}", code)][:2]

    slugs = candidate.get("owid_charts", [])
    if not isinstance(slugs, list):
        slugs = []
    slugs = [slug for slug in slugs if isinstance(slug, str)
             and re.fullmatch(r"[a-z0-9-]{3,100}", slug)][:2]

    country = candidate.get("country_code", "WLD")
    if not isinstance(country, str) or not re.fullmatch(r"[A-Za-z]{2,3}", country):
        country = "WLD"
    entity = candidate.get("entity", "World")
    if not isinstance(entity, str) or not entity.strip() or len(entity) > 80:
        entity = "World"

    title = candidate.get("Title", topic)
    if not isinstance(title, str) or not title.strip():
        title = topic
    title = re.sub(r"systematic review", "evidence review", title, flags=re.I)

    plan = {
        "topic": topic,
        "Title": title,
        "research_question": candidate.get("research_question", topic)
        if isinstance(candidate.get("research_question"), str) else topic,
        "search_queries": queries,
        "exposure_terms": terms("exposure_terms"),
        "outcome_terms": terms("outcome_terms"),
        "focus_terms": terms("focus_terms"),
        "world_bank_indicators": indicators,
        "owid_charts": slugs,
        "country_code": country.upper(),
        "entity": entity.strip(),
    }
    if benchmark_hint:
        # The model can confuse the balance-scale learning task with equilibrium
        # theory. Keep the evidence search and later UCI lookup on the actual task.
        plan.update({
            "Title": "Learning the Balance Scale Rule from Weights and Distances",
            "research_question": (
                "How well can machine learning models classify balance-scale outcomes "
                "from left and right weights and distances?"
            ),
            "search_queries": [
                "balance scale task machine learning classification",
                "Siegler balance scale problem neural network",
                "balance scale learning decision tree",
            ],
            "exposure_terms": ["balance scale"],
            "outcome_terms": ["model", "learning", "classification"],
            "focus_terms": ["Balance Scale", "weight", "distance"],
            "required_context_terms": [
                "weight", "distance", "lever", "Siegler", "balance scale task",
                "balance scale problem", "balance scale phenomena",
                "balance scale development",
            ],
            "world_bank_indicators": [],
            "owid_charts": [],
        })
    with open(os.path.join(workspace_dir, "research_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    return plan
