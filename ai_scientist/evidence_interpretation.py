"""Deterministic, source-attributed interpretation of retrieved evidence.

These observations interpret computed values and explicit source profiles. They
never infer why a trend occurred or claim that a missing excerpt proves a study
did not perform a particular validation step.
"""

from collections import defaultdict

from ai_scientist.synthesis import evidence_map, profile_sources


def _refs(ids):
    return " ".join(f"[{source_id}]" for source_id in dict.fromkeys(ids))


def _number(value):
    return f"{value:,.3f}".rstrip("0").rstrip(".") if abs(value) >= 1 else f"{value:.4g}"


def analyze_charts(evidence, analysis, study_analysis):
    """Interpret exactly the observations drawn in each chart, with auditable IDs."""
    series = {item["source_id"]: item for item in analysis.get("series", [])}
    findings = {item["id"]: item for item in study_analysis.get("findings", [])}
    mapped = {(item["method"], item["task"]): item
              for item in evidence.get("evidence_map", [])}
    chart_analyses = []
    for chart in evidence.get("charts", []):
        kind = chart["kind"]
        if kind == "published_data":
            source_id = chart.get("dataset_id") or chart["source_ids"][0]
            if source_id not in series:
                raise ValueError(f"No validated series for {chart['file']}.")
            item = series[source_id]
            first, last = item["first"], item["last"]
            detail = (f"The plotted {item['observation_count']} observations span "
                      f"{first['year']} to {last['year']}; the value moves from "
                      f"{_number(first['value'])} to {_number(last['value'])} "
                      f"{item['unit']}, a net {item['direction']} of "
                      f"{_number(abs(item['net_change']))} {item['unit']} {_refs([source_id])}.")
            recent = item.get("recent_five_year_window")
            if recent and recent["direction"] != item["direction"]:
                detail += (f" The latest available five-year window instead shows a "
                           f"{recent['direction']} of {_number(abs(recent['net_change']))} "
                           f"{item['unit']}; the endpoint trend hides this reversal.")
            elif item["increasing_pairs"] and item["decreasing_pairs"]:
                detail += (" The visible upward and downward adjacent-year changes show "
                           "that the net direction is not a uniform annual trend.")
            elif item["consecutive_year_pairs"] == 0:
                detail += (" No two plotted years are adjacent, so the intervening "
                           "year-to-year path is unknown.")
            else:
                detail += (f" The observed minimum is {_number(item['minimum']['value'])} "
                           f"in {item['minimum']['year']} and the maximum is "
                           f"{_number(item['maximum']['value'])} in {item['maximum']['year']}.")
            detail += (" These are within-series descriptions, not evidence of a cause; "
                       "missing years and source definitions constrain interpretation.")
            basis = "validated observations and within-series calculations"
        elif kind == "study_estimates":
            ids = chart.get("finding_ids") or []
            group = [findings[finding_id] for finding_id in ids if finding_id in findings]
            if len(group) != len(ids) or not group:
                raise ValueError(f"No verified findings for {chart['file']}.")
            values = [item["estimate"] for item in group]
            measure = group[0]["measure"].replace("_", " ")
            detail = (f"This panel plots {len(group)} source-reported {measure} "
                      f"estimate{'s' if len(group) != 1 else ''} "
                      f"({_number(min(values))} to {_number(max(values))}) "
                      f"{_refs(chart['source_ids'])}.")
            if len({item["source_id"] for item in group}) >= 2:
                if group[0]["measure"] in {"odds_ratio", "risk_ratio", "hazard_ratio"} \
                        and min(values) < 1 < max(values):
                    detail += (" The point estimates lie on opposite sides of the neutral "
                               "ratio of 1, so the displayed directions disagree.")
                elif min(values) != max(values):
                    detail += (" The spread indicates variation among the displayed reports, "
                               "not a pooled effect.")
                else:
                    detail += (" The identical plotted values do not establish comparable "
                               "study designs or populations.")
            else:
                detail += (" Only one work is represented; this chart cannot show "
                           "between-study consistency.")
            with_intervals = [item for item in group if item.get("ci_low") is not None
                              and item.get("ci_high") is not None]
            if with_intervals:
                detail += (f" Reported intervals are drawn for {len(with_intervals)} of "
                           f"{len(group)} points; missing intervals must not be read as "
                           "zero uncertainty.")
                if group[0]["measure"] in {"odds_ratio", "risk_ratio", "hazard_ratio"}:
                    crossing = sum(item["ci_low"] <= 1 <= item["ci_high"] for item in with_intervals)
                    detail += (f" {crossing} displayed interval{'s' if crossing != 1 else ''} "
                               "include the neutral ratio of 1; this observation depends on "
                               "the source-reported interval definition.")
            detail += (" The estimates are not pooled or causal; their outcome definitions, "
                       "contrasts, populations and interval meanings require checking in the sources.")
            basis = "verified source passages for plotted finding IDs"
        elif kind == "method_task_map":
            methods = chart.get("methods") or []
            tasks = chart.get("tasks") or []
            pairs = chart.get("pairs") or [
                {"method": method, "task": task} for method in methods for task in tasks
                if (method, task) in mapped]
            visible = [mapped[(pair["method"], pair["task"])] for pair in pairs
                       if (pair["method"], pair["task"]) in mapped]
            if not visible:
                raise ValueError(f"No documented method-task cells for {chart['file']}.")
            leader = max(visible, key=lambda item: (item["count"], item["method"], item["task"]))
            tied = sum(item["count"] == leader["count"] for item in visible)
            detail = (f"The map displays {len(visible)} documented method-task pairings "
                      f"across {len(methods)} methods and {len(tasks)} tasks. "
                      f"{'One of the ' if tied > 1 else 'The '}most frequently recorded "
                      "pairing in the displayed sample is "
                      f"{leader['method']} for {leader['task']} "
                      f"({leader['count']} {'work' if leader['count'] == 1 else 'works'}) "
                      f"{_refs(leader['source_ids'])}.")
            multiple = [(task, len({item["method"] for item in visible if item["task"] == task}))
                        for task in tasks]
            shared = next(((task, count) for task, count in multiple if count > 1), None)
            if shared:
                detail += (f" {shared[1]} distinct methods are mentioned for {shared[0]}; "
                           "this identifies a comparison question, not an answer about superiority.")
            else:
                detail += (" No displayed task has multiple documented methods, so this "
                           "sample cannot support a within-task method comparison.")
            detail += (" Displayed counts measure mentions in the retrieved works, not "
                       "accuracy, effectiveness or strength of evidence.")
            basis = "displayed method-task cells and their source IDs"
        else:
            raise ValueError(f"Unsupported chart kind: {kind}")
        chart_analyses.append({"file": chart["file"], "kind": kind,
                               "source_ids": chart["source_ids"], "text": detail,
                               "basis": basis})
    return chart_analyses


def build_interpretation(evidence, analysis, study_analysis, plan):
    """Return auditable analytical observations for the paper, not a source list."""
    sources = evidence.get("sources") or []
    profiles = evidence.get("source_profiles")
    if profiles is None:
        profiles = profile_sources(sources, plan)
    mapped = evidence.get("evidence_map")
    if mapped is None:
        mapped = evidence_map(profiles)
    observations = []

    def add(kind, text, source_ids, basis):
        observations.append({"kind": kind, "text": text, "source_ids": list(dict.fromkeys(source_ids)),
                             "basis": basis})

    for item in analysis.get("series", []):
        source_id = item["source_id"]
        recent = item.get("recent_five_year_window")
        direction = item["direction"]
        if recent and direction != "no change" and recent["direction"] != "no change" \
                and recent["direction"] != direction:
            text = (f"The full observed span of {item['title']} moves upward"
                    if direction == "increase" else
                    f"The full observed span of {item['title']} moves downward")
            text += (f", but its latest available five-year window moves in the opposite "
                     f"direction {_refs([source_id])}. A single start-to-end change therefore "
                     "conceals a recent reversal; it should not be projected forward.")
            basis = "opposite full-span and recent-window directions"
        elif item["increasing_pairs"] and item["decreasing_pairs"]:
            text = (f"The net direction of {item['title']} {_refs([source_id])} masks both rising "
                    "and falling directly observed year-to-year intervals. The endpoint "
                    "comparison is therefore not a description of a uniform trend.")
            basis = "mixed consecutive-year changes"
        elif item["consecutive_year_pairs"] == 0:
            text = (f"The observations for {item['title']} {_refs([source_id])} support a "
                    "first-to-last comparison, but contain no adjacent-year pair. Annual "
                    "variation and the shape of the intervening trajectory cannot be inferred.")
            basis = "no consecutive-year observations"
        elif item["observation_count"] <= 3:
            text = (f"The few observed points for {item['title']} {_refs([source_id])} establish "
                    "a descriptive direction, not a stable long-term trajectory. Additional "
                    "observations would be needed to assess whether that direction persists.")
            basis = "sparse observed series"
        else:
            text = (f"The observed annual transitions for {item['title']} {_refs([source_id])} "
                    "should be interpreted alongside its first-to-last change; neither the "
                    "direction nor the extrema identify a cause without an appropriate study design.")
            basis = "descriptive series without causal design"
        add("series_interpretation", text, [source_id], basis)

    comparisons = study_analysis.get("comparisons") or []
    findings = study_analysis.get("findings") or []
    for group in comparisons:
        ids = group["source_ids"]
        if group["measure"] in {"odds_ratio", "risk_ratio", "hazard_ratio"} \
                and group["minimum"] < 1 < group["maximum"]:
            text = (f"For the matched {group['outcome']} comparison, reported "
                    f"{group['measure'].replace('_', ' ')} estimates fall on both sides of "
                    f"the neutral ratio {_refs(ids)}. The direction is not consistent "
                    "across these works; differences in populations and models still need "
                    "checking before a pooled conclusion.")
            basis = "matched reported ratios straddle one"
        else:
            text = (f"The matched {group['outcome']} estimates {_refs(ids)} vary across "
                    "the cited studies despite sharing the recorded measure, exposure, "
                    "contrast and unit. Their range describes between-study variation, "
                    "not a pooled effect or proof of a causal difference.")
            basis = "matched descriptive comparison"
        add("study_comparison", text, ids, basis)
    if findings and not comparisons:
        ids = list(dict.fromkeys(item["source_id"] for item in findings))
        if len(ids) >= 2:
            text = (f"The verified numerical estimates {_refs(ids)} do not pass the common "
                    "measure, exposure, outcome, contrast and unit checks required for a "
                    "like-for-like summary. Their values should be interpreted within "
                    "their original study contexts rather than ranked together.")
        else:
            text = (f"The verified estimate {_refs(ids)} has no matched result from a second "
                    "work in this collection. Its direction is study-specific; the retrieved "
                    "evidence does not establish between-study consistency.")
        add("study_comparability", text, ids, "no matched cross-study estimate")

    by_task = defaultdict(list)
    for item in mapped:
        by_task[item["task"]].append(item)
    compared_task = False
    for task, entries in sorted(by_task.items()):
        if len(entries) < 2:
            continue
        first, second = entries[:2]
        ids = first["source_ids"] + second["source_ids"]
        text = (f"For {task}, the accessible works mention {first['method']} "
                f"{_refs(first['source_ids'])} and {second['method']} "
                f"{_refs(second['source_ids'])}. This is a comparison of documented "
                "approaches, not evidence that either performs better: a ranking would "
                "require comparable tasks, populations and validation metrics.")
        add("method_comparison", text, ids, "multiple methods linked to one task")
        compared_task = True
        break
    if not compared_task and len(mapped) >= 2:
        first, second = mapped[:2]
        ids = first["source_ids"] + second["source_ids"]
        text = (f"The evidence map links {first['method']} to {first['task']} "
                f"{_refs(first['source_ids'])}, whereas it links {second['method']} "
                f"to {second['task']} {_refs(second['source_ids'])}. These are not "
                "interchangeable evaluation tasks, so the retrieved records alone "
                "cannot support a performance ranking.")
        add("task_comparison", text, ids, "different method-task pairs")

    validated = [item["source_id"] for item in profiles if item.get("validation_passage")]
    risks = [item["source_id"] for item in profiles if item.get("risk_passage")]
    if validated:
        missing = [item["source_id"] for item in profiles if not item.get("validation_passage")]
        text = (f"An explicit validation passage is available for {_refs(validated[:4])}. ")
        if missing:
            text += (f"No comparable passage was extracted from the accessible text of "
                     f"{_refs(missing[:4])}; this is a visibility gap, not proof that those "
                     "studies omitted validation.")
        else:
            text += ("The presence of validation language alone does not establish "
                     "external validity or comparable test conditions.")
        add("validation_appraisal", text, validated[:4] + missing[:4], "explicit validation excerpts")
    if risks:
        add("limitations_appraisal",
            f"The accessible text explicitly discusses a limitation or bias in "
            f"{_refs(risks[:4])}. Those caveats constrain generalization; their exact scope "
            "must be checked against each cited study before using its conclusions elsewhere.",
            risks[:4], "explicit limitation excerpts")

    if not observations and sources:
        ids = [source["id"] for source in sources[:2]]
        text = (f"The accessible record {_refs(ids)} does not provide extractable, "
                "comparable outcomes or method-task evidence for a substantive "
                "cross-study interpretation. The review can document this evidence gap, "
                "but cannot infer an effect or rank approaches from these records.")
        add("evidence_gap", text, ids, "insufficient structured evidence")
    if not observations:
        raise ValueError("No analyzable sources or numerical series were available for the paper.")
    return {"mode": "quantitative" if analysis.get("series") or findings else "qualitative",
            "observations": observations}
