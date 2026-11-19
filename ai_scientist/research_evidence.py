"""Collect verifiable secondary sources and plot only values returned by those sources."""

import csv
import io
import json
import math
import os
import re
from datetime import date
from html import unescape
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from ai_scientist.analyze_evidence import analyze_datasets
from ai_scientist.fulltext import enrich_with_open_fulltext
from ai_scientist.graphviz_figures import render_method_task_map
from ai_scientist.s2_client import get_json as s2_get_json
from ai_scientist.study_evidence import analyze_findings, extract_study_findings, rank_sources
from ai_scientist.synthesis import evidence_map, profile_sources


MAX_SOURCES = 35
MAX_QUERY_COUNT = 5
REQUEST_TIMEOUT = 15
HEADERS = {"User-Agent": "AI-Scientist-Review/1.0 (local evidence review)"}


def _json_get(session, url, params=None):
    response = session.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json(), response.url


def _abstract_from_index(index):
    if not isinstance(index, dict):
        return ""
    words = [(position, word) for word, positions in index.items()
             if isinstance(positions, list) for position in positions if isinstance(position, int)]
    return " ".join(word for _, word in sorted(words))[:10000]


def _normalize_doi(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I).lower()


def search_openalex(query, session, page=1):
    payload, _ = _json_get(session, "https://api.openalex.org/works",
                           {"search": query, "per_page": 25, "page": page})
    records = []
    for item in payload.get("results", []):
        title = item.get("display_name")
        if not isinstance(title, str) or not title.strip():
            continue
        doi = _normalize_doi(item.get("doi"))
        year = item.get("publication_year")
        authors = [entry.get("author", {}).get("display_name", "")
                   for entry in item.get("authorships", []) if isinstance(entry, dict)]
        authors = [name for name in authors if isinstance(name, str) and name.strip()]
        venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        oa = item.get("best_oa_location") or {}
        records.append({
            "title": title.strip(), "year": year if isinstance(year, int) else None,
            "doi": doi, "url": f"https://doi.org/{doi}" if doi else item.get("id", ""),
            "authors": authors, "venue": venue,
            "abstract": _abstract_from_index(item.get("abstract_inverted_index")),
            "providers": ["OpenAlex"], "query": query,
            "oa_pdf_url": oa.get("pdf_url") if oa.get("is_oa") else None,
            "oa_license": oa.get("license") if oa.get("is_oa") else None,
            "oa_version": oa.get("version") if oa.get("is_oa") else None,
        })
    return records


def search_crossref(query, session):
    payload, _ = _json_get(session, "https://api.crossref.org/works",
                           {"query": query, "rows": 25})
    records = []
    for item in payload.get("message", {}).get("items", []):
        titles = item.get("title", [])
        title = titles[0] if isinstance(titles, list) and titles else ""
        if not isinstance(title, str) or not title.strip():
            continue
        doi = _normalize_doi(item.get("DOI"))
        parts = (item.get("published") or item.get("issued") or {}).get("date-parts", [])
        year = parts[0][0] if parts and parts[0] else None
        abstract = re.sub(r"<[^>]+>", " ", item.get("abstract", ""))
        authors = [" ".join(str(entry.get(part, "")).strip() for part in ("given", "family")).strip()
                   for entry in item.get("author", []) if isinstance(entry, dict)]
        authors = [name for name in authors if name]
        venues = item.get("container-title") or []
        venue = venues[0] if isinstance(venues, list) and venues else ""
        records.append({
            "title": unescape(title.strip()), "year": year if isinstance(year, int) else None,
            "doi": doi, "url": f"https://doi.org/{doi}" if doi else item.get("URL", ""),
            "authors": authors, "venue": venue,
            "abstract": unescape(re.sub(r"\s+", " ", abstract)).strip()[:10000],
            "providers": ["Crossref"], "query": query,
        })
    return records


def search_semantic_scholar(query, session):
    """One bounded relevance-ranked page; missing/rate-limited API remains optional."""
    payload = s2_get_json(session, "/graph/v1/paper/search",
                          {"query": query, "limit": 25,
                           "fields": "title,year,abstract,authors,venue,url,externalIds,openAccessPdf"},
                          timeout=REQUEST_TIMEOUT)
    records = []
    for item in payload.get("data", []):
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        doi = _normalize_doi((item.get("externalIds") or {}).get("DOI"))
        year = item.get("year")
        open_pdf = item.get("openAccessPdf") or {}
        records.append({
            "title": title.strip(), "year": year if isinstance(year, int) else None,
            "doi": doi, "url": f"https://doi.org/{doi}" if doi else item.get("url", ""),
            "authors": [a["name"] for a in item.get("authors", []) if isinstance(a, dict)
                        and isinstance(a.get("name"), str)],
            "venue": item.get("venue") or "",
            "abstract": (item.get("abstract") or "")[:10000],
            "providers": ["Semantic Scholar"], "query": query,
            "oa_pdf_url": open_pdf.get("url"),
        })
    return records


def _deduplicate_sources(records):
    by_doi = {}
    by_title = {}
    unique = []
    for item in records:
        doi = item.get("doi") or ""
        title_key = re.sub(r"\W+", "", item["title"].casefold())
        if not doi and not title_key:
            continue
        previous = by_doi.get(doi) if doi else None
        title_match = by_title.get(title_key) if title_key else None
        if previous is None and title_match is not None and (
                not doi or not title_match.get("doi") or title_match["doi"] == doi):
            previous = title_match
        if previous is not None:
            previous["providers"] = sorted(set(previous["providers"] + item["providers"]))
            if len(item["abstract"]) > len(previous["abstract"]):
                previous["abstract"] = item["abstract"]
            if not previous.get("authors") and item.get("authors"):
                previous["authors"] = item["authors"]
            if not previous.get("venue") and item.get("venue"):
                previous["venue"] = item["venue"]
            for field in ("oa_pdf_url", "oa_license", "oa_version"):
                if not previous.get(field) and item.get(field):
                    previous[field] = item[field]
            if doi and not previous.get("doi"):
                previous["doi"] = doi
                previous["url"] = f"https://doi.org/{doi}"
            if doi:
                by_doi[doi] = previous
        else:
            unique.append(item)
            if doi:
                by_doi[doi] = item
            if title_key and title_key not in by_title:
                by_title[title_key] = item
    return unique


def _finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _dataset_relevant(dataset, plan):
    terms = [term.casefold() for term in (plan.get("exposure_terms") or [])
             + (plan.get("outcome_terms") or []) + (plan.get("focus_terms") or [])
             if isinstance(term, str) and len(term.strip()) >= 4]
    if not terms:
        return True
    title = " ".join(str(dataset.get(field, "")) for field in ("title", "chart_title", "metric"))
    title = title.casefold()
    for term in terms:
        variants = [term]
        if len(term.split()) > 1 and len(term.split()[-1]) >= 6:
            variants.append(term.split()[-1])
        if any(re.search(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", title)
               for variant in variants):
            return True
    return False


def fetch_world_bank(code, country, session):
    if not re.fullmatch(r"[A-Z0-9.]{3,40}", code) or not re.fullmatch(r"[A-Z]{2,3}", country):
        return None
    base = "https://api.worldbank.org/v2"
    metadata, metadata_url = _json_get(session, f"{base}/indicator/{quote(code)}", {"format": "json"})
    if not isinstance(metadata, list) or len(metadata) < 2 or not metadata[1]:
        return None
    indicator = metadata[1][0]
    observations, data_url = _json_get(
        session, f"{base}/country/{country}/indicator/{quote(code)}",
        {"format": "json", "per_page": 100, "date": f"2000:{date.today().year}"},
    )
    if not isinstance(observations, list) or len(observations) < 2 or not observations[1]:
        return None
    values = {}
    entity = country
    for row in observations[1]:
        year = str(row.get("date", ""))
        number = _finite_number(row.get("value"))
        if re.fullmatch(r"\d{4}", year) and number is not None:
            values[int(year)] = number
            entity = (row.get("country") or {}).get("value") or entity
    if len(values) < 2:
        return None
    indicator_title = indicator.get("name", code)
    title_unit = re.search(r"\(([^()]*)\)\s*$", indicator_title)
    return {
        "provider": "World Bank", "title": indicator_title,
        "entity": entity, "unit": indicator.get("unit") or
        (title_unit.group(1) if title_unit else "as reported by source"),
        "source_url": data_url, "metadata_url": metadata_url,
        "credit": indicator.get("sourceOrganization", ""),
        "points": [{"year": year, "value": values[year]} for year in sorted(values)],
        "retrieved_at": date.today().isoformat(),
    }


def fetch_owid_chart(slug, entity, session):
    if not re.fullmatch(r"[a-z0-9-]{3,100}", slug):
        return None
    base = f"https://ourworldindata.org/grapher/{slug}"
    metadata, metadata_url = _json_get(session, base + ".metadata.json")
    response = session.get(base + ".csv", headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    if len(response.content) > 8_000_000:
        return None
    reader = csv.DictReader(io.StringIO(response.text))
    headers = reader.fieldnames or []
    if "Entity" not in headers or "Year" not in headers:
        return None
    columns = metadata.get("columns", {}) if isinstance(metadata, dict) else {}
    candidates = [name for name in headers if name in columns
                  and columns[name].get("type") == "Numeric"]
    if not candidates:
        candidates = [name for name in headers if name not in ("Entity", "Code", "Year", "Day")]
    if not candidates:
        return None
    metric = candidates[0]
    values = {}
    for row in reader:
        if row.get("Entity", "").casefold() != entity.casefold():
            continue
        year = row.get("Year", "")
        number = _finite_number(row.get(metric))
        if re.fullmatch(r"\d{4}", year) and number is not None:
            values[int(year)] = number
    if len(values) < 2:
        return None
    column_meta = columns.get(metric, {})
    chart_meta = metadata.get("chart", {})
    return {
        "provider": "Our World in Data", "title": metric,
        "chart_title": chart_meta.get("title") or metric,
        "entity": entity, "unit": column_meta.get("unit") or "as reported by source",
        "source_url": base, "data_url": response.url, "metadata_url": metadata_url,
        "credit": column_meta.get("citationShort") or chart_meta.get("citation", ""),
        "metric": metric,
        "points": [{"year": year, "value": values[year]} for year in sorted(values)],
        "retrieved_at": date.today().isoformat(),
    }


def build_charts(evidence, workspace_dir):
    charts = []
    rows = []
    for dataset in evidence["datasets"]:
        points = dataset["points"]
        filename = f"chart_{len(charts) + 1:02d}.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot([p["year"] for p in points], [p["value"] for p in points], marker="o", markersize=3)
        ax.set_title(f"{dataset['title']} — {dataset['entity']}")
        ax.set_xlabel("Year")
        ax.set_ylabel(dataset["unit"])
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(workspace_dir, filename), dpi=180)
        plt.close(fig)
        charts.append({"file": filename, "kind": "published_data", "source_ids": [dataset["id"]],
                       "dataset_id": dataset["id"],
                       "caption": f"Published values for {dataset['entity']}: {dataset['title']} ({dataset['id']})."})
        rows.extend({"chart": filename, "source_id": dataset["id"], "year": p["year"],
                     "value": p["value"], "unit": dataset["unit"], "entity": dataset["entity"]}
                    for p in points)

    findings = evidence.get("study_findings", [])
    plotted = set()
    for finding in findings:
        if any(finding[field].startswith("not specified") for field in ("exposure", "outcome")):
            continue
        compatible = all(not finding[field].startswith("not specified")
                         for field in ("contrast", "unit"))
        key = ((finding["measure"], finding["exposure"].casefold(),
                finding["outcome"].casefold(), finding["contrast"].casefold(),
                finding["unit"].casefold()) if compatible else (finding["id"],))
        if key in plotted or len(plotted) >= 2:
            continue
        plotted.add(key)
        group = ([item for item in findings if (item["measure"], item["exposure"].casefold(),
                  item["outcome"].casefold(), item["contrast"].casefold(),
                  item["unit"].casefold()) == key][:10] if compatible else [finding])
        filename = f"chart_{len(charts) + 1:02d}.png"
        fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(group) + 1.5)))
        labels = [f"{item['source_id']} ({item['id']})" for item in group]
        values = [item["estimate"] for item in group]
        ax.scatter(values, range(len(group)), color="#2B5C8F", s=55)
        for index, item in enumerate(group):
            if item.get("ci_low") is not None and item.get("ci_high") is not None:
                ax.plot([item["ci_low"], item["ci_high"]], [index, index], color="#2B5C8F")
        ax.set_yticks(range(len(group)), labels)
        ax.set_xlabel(f"Reported {finding['measure'].replace('_', ' ')} (source units)")
        ax.set_title("Verified estimates from retrieved sources")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(workspace_dir, filename), dpi=180)
        plt.close(fig)
        charts.append({"file": filename, "kind": "study_estimates",
                       "source_ids": list(dict.fromkeys(item["source_id"] for item in group)),
                       "finding_ids": [item["id"] for item in group],
                       "caption": "Source-reported estimates; points are not pooled. "
                       + ", ".join(f"{item['id']} [{item['source_id']}]" for item in group)
                       + ". Check study_findings.csv for the verbatim passages."})
        rows.extend({"chart": filename, "source_id": item["source_id"], "year": "",
                     "value": item["estimate"], "unit": item["measure"],
                     "entity": item["outcome"]} for item in group)

    mapped = evidence.get("evidence_map", [])
    if mapped:
        filename = f"chart_{len(charts) + 1:02d}.png"
        methods = sorted({item["method"] for item in mapped})[:8]
        tasks = sorted({item["task"] for item in mapped})[:6]
        visible = [item for item in mapped if item["method"] in methods and item["task"] in tasks]
        visible = sorted(visible, key=lambda item: (-item["count"], item["method"], item["task"]))[:12]
        methods = sorted({item["method"] for item in visible})
        tasks = sorted({item["task"] for item in visible})
        values = [[next((item["count"] for item in visible if item["method"] == method
                         and item["task"] == task), 0) for task in tasks] for method in methods]
        rendered = render_method_task_map(visible, os.path.join(workspace_dir, filename))
        if not rendered:
            fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(tasks)), max(3.5, 0.6 * len(methods) + 2)))
            image = ax.imshow(values, cmap="Blues", aspect="auto", vmin=0)
            ax.set_xticks(range(len(tasks)), tasks, rotation=25, ha="right")
            ax.set_yticks(range(len(methods)), methods)
            for row, method in enumerate(methods):
                for column, task in enumerate(tasks):
                    value = values[row][column]
                    if value:
                        ax.text(column, row, str(value), ha="center", va="center",
                                color="white" if value >= max(max(line) for line in values) * 0.6 else "black")
            ax.set_title("Methods and research tasks explicitly mentioned in retrieved works")
            fig.colorbar(image, ax=ax, label="Number of retrieved works")
            fig.tight_layout()
            fig.savefig(os.path.join(workspace_dir, filename), dpi=180)
            plt.close(fig)
            evidence.setdefault("warnings", []).append(
                "Graphviz renderer unavailable; used the method-task heatmap instead.")
        charts.append({"file": filename, "kind": "method_task_map",
                       "renderer": "graphviz" if rendered else "matplotlib",
                       "source_ids": sorted({source_id for item in visible for source_id in item["source_ids"]}),
                       "methods": methods, "tasks": tasks,
                       "pairs": [{"method": item["method"], "task": item["task"]} for item in visible],
                       "caption": ("Evidence network" if rendered else "Evidence map")
                       + " of explicitly mentioned methods and research tasks; "
                       + ("edge labels" if rendered else "cell counts")
                       + " count retrieved works, not model effectiveness."})
        rows.extend({"chart": filename, "source_id": ",".join(item["source_ids"]), "year": "",
                     "value": item["count"], "unit": item["method"], "entity": item["task"]}
                    for item in visible)

    data_path = os.path.join(workspace_dir, "chart_data.csv")
    with open(data_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["chart", "source_id", "year", "value", "unit", "entity"])
        writer.writeheader()
        writer.writerows(rows)
    evidence["charts"] = charts
    return charts


def collect_evidence(plan, workspace_dir="./workspace", session=None, model=None,
                     retrieve_fulltext=True):
    os.makedirs(workspace_dir, exist_ok=True)
    session = session or requests.Session()
    warnings = []
    records = []
    queries = list(plan["search_queries"][:3])
    exposures = (plan.get("exposure_terms") or [])[:2]
    outcomes = sorted((term for term in plan.get("outcome_terms") or []
                       if term.casefold() not in {"affect", "effect", "impact", "state", "conditions"}),
                      key=len)[:2]
    for exposure in exposures:
        for outcome in outcomes:
            query = f"{exposure} {outcome}"
            if query.casefold() not in {existing.casefold() for existing in queries}:
                queries.append(query)
    queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))[:MAX_QUERY_COUNT]
    scholar_available = True
    for index, query in enumerate(queries):
        searches = [("OpenAlex", search_openalex), ("Crossref", search_crossref)]
        if index < (3 if os.environ.get("SEMANTIC_SCHOLAR_API_KEY") else 1) and scholar_available:
            searches.append(("Semantic Scholar", search_semantic_scholar))
        for provider, search in searches:
            try:
                records.extend(search(query, session))
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                if provider == "Semantic Scholar" and getattr(
                        getattr(exc, "response", None), "status_code", None) == 429:
                    warnings.append("Semantic Scholar: rate limited (429); "
                                    "continued with OpenAlex and Crossref.")
                else:
                    warnings.append(f"{provider} ({query}): {exc}")
                if provider == "Semantic Scholar":
                    scholar_available = False
        if index < 2:
            try:
                records.extend(search_openalex(query, session, page=2))
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                warnings.append(f"OpenAlex page 2 ({query}): {exc}")
    unique_records = _deduplicate_sources(records)
    sources = rank_sources(unique_records, plan, MAX_SOURCES)
    if records and not sources:
        warnings.append("Retrieved works did not pass exposure/outcome relevance screening.")
    if retrieve_fulltext:
        warnings.extend(enrich_with_open_fulltext(sources, session, limit=22))
    profiles = profile_sources(sources, plan)
    study_findings, extraction_warning = extract_study_findings(sources, model, plan=plan)
    if extraction_warning:
        warnings.append(extraction_warning)
    if sources and not study_findings:
        warnings.append("No source-attributed quantitative result was recoverable from accessible text.")

    datasets = []
    for code in plan.get("world_bank_indicators", [])[:2]:
        try:
            item = fetch_world_bank(code, plan.get("country_code", "WLD"), session)
            if item and _dataset_relevant(item, plan):
                datasets.append(item)
            elif item:
                warnings.append(f"World Bank {code}: title did not match the research topic.")
            else:
                warnings.append(f"World Bank {code}: no usable time series for the selected region.")
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"World Bank {code}: {exc}")
    for slug in plan.get("owid_charts", [])[:2]:
        try:
            item = fetch_owid_chart(slug, plan.get("entity", "World"), session)
            if item and _dataset_relevant(item, plan):
                datasets.append(item)
            elif item:
                warnings.append(f"Our World in Data {slug}: title did not match the research topic.")
            else:
                warnings.append(f"Our World in Data {slug}: no usable time series for the selected entity.")
        except (requests.RequestException, ValueError, KeyError, TypeError, csv.Error) as exc:
            warnings.append(f"Our World in Data {slug}: {exc}")
    for number, item in enumerate(datasets, 1):
        item["id"] = f"D{number}"

    if not sources and not datasets:
        raise RuntimeError("Không tìm thấy nguồn tài liệu hoặc số liệu có thể kiểm tra. "
                           + ("; ".join(warnings[:3]) if warnings else "Hãy thử chủ đề cụ thể hơn."))
    evidence = {"topic": plan["topic"], "retrieved_at": date.today().isoformat(),
                "queries": queries, "sources": sources, "datasets": datasets,
                "retrieval_summary": {
                    "raw_hits": len(records), "unique_records": len(unique_records),
                    "relevant_sources": len(sources),
                    "full_text_read": sum((s.get("fulltext") or {}).get("status") == "full_text"
                                          for s in sources),
                    "abstract_only": sum((s.get("fulltext") or {}).get("status") == "abstract_only"
                                         for s in sources),
                },
                "study_findings": study_findings,
                "source_profiles": profiles, "evidence_map": evidence_map(profiles),
                "warnings": warnings}
    evidence["analysis"] = analyze_datasets(datasets)
    evidence["study_analysis"] = analyze_findings(study_findings)
    build_charts(evidence, workspace_dir)
    with open(os.path.join(workspace_dir, "study_findings.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["id", "source_id", "source_url", "location", "access_level", "quote", "estimate_text",
                  "estimate", "measure", "exposure", "outcome", "contrast", "unit",
                  "ci_low", "ci_high"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in fields} for item in study_findings)
    with open(os.path.join(workspace_dir, "evidence.json"), "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    return evidence
