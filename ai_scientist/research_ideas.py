"""Source-grounded research proposals and bounded computational experiments."""

import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MIN_ANNUAL_POINTS = 12
MIN_TRAIN_POINTS = 8


def _eligible_dataset(dataset):
    points = list(dataset.get("points") or [])
    if len(points) < MIN_ANNUAL_POINTS:
        return False, f"fewer than {MIN_ANNUAL_POINTS} annual observations", points
    if any(not isinstance(point, dict) or isinstance(point.get("year"), bool)
           or not isinstance(point.get("year"), int) for point in points):
        return False, "observation years are invalid", points
    points.sort(key=lambda point: point["year"])
    years = [point["year"] for point in points]
    values = [point["value"] for point in points]
    if (any(isinstance(year, bool) or not isinstance(year, int) for year in years)
            or any(next_year - year != 1 for year, next_year in zip(years, years[1:]))):
        return False, "annual observations contain gaps or duplicate years", points
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        return False, "observations are not finite numbers", points
    return True, "eligible uninterrupted annual series", points


def _linear_next(values):
    """Least-squares one-step forecast on the observed training prefix only."""
    count = len(values)
    mean_x = (count - 1) / 2
    mean_y = sum(values) / count
    numerator = sum((index - mean_x) * (value - mean_y)
                    for index, value in enumerate(values))
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    return mean_y + numerator / denominator * (count - mean_x)


def _idea(dataset, eligible, reason):
    source_id = dataset["id"]
    return {
        "id": "I1",
        "title": f"Test short-horizon predictability of {dataset['title']} for {dataset['entity']}",
        "hypothesis": ("An expanding-window linear trend forecast has a lower one-step "
                       "mean absolute error than a last-observation baseline on the "
                       f"published annual series [{source_id}]."),
        "rationale": ("The retrieved series makes a direct comparison between a simple trend "
                      "and a persistence baseline possible. This is a testable proposal "
                      "for this review, not proof that the idea is globally novel."),
        "source_ids": [source_id],
        "method": "Expanding-window one-step backtest with no randomized split",
        "status": "feasible" if eligible else "not_feasible",
        "feasibility_reason": reason,
        "experiment_type": "annual_forecast_backtest" if eligible else None,
        "protocol": {
            "data_requirements": (f"At least {MIN_ANNUAL_POINTS} consecutive annual observations "
                                  f"of {dataset['title']} for {dataset['entity']}, with finite values."),
            "comparison": "Forecast each held-out year using the last observed value.",
            "evaluation": (f"Start with {MIN_TRAIN_POINTS} years and expand the training window "
                           "one year at a time. Compare MAE on identical held-out years. Lower MAE "
                           "for the linear trend supports the forecast hypothesis in this series."),
        },
    }


def _literature_idea(plan, evidence, idea_id):
    mapped = evidence.get("evidence_map") or []
    tasks = sorted({item["task"] for item in mapped})
    for task in tasks:
        entries = [item for item in mapped if item["task"] == task]
        if len(entries) < 2:
            continue
        first, second = entries[:2]
        source_ids = list(dict.fromkeys(first["source_ids"] + second["source_ids"]))
        citations = " ".join(f"[{source_id}]" for source_id in source_ids)
        return {
            "id": idea_id,
            "title": f"Compare {first['method']} and {second['method']} for {task}",
            "hypothesis": (f"Under the same future dataset, outcome definition and evaluation "
                           f"split, {first['method']} and {second['method']} may yield "
                           f"different performance for {task} {citations}."),
            "rationale": ("These method-task pairings were documented in retrieved works; "
                          "the accessible evidence does not establish a fair head-to-head "
                          "comparison or verify that this proposal is globally novel."),
            "source_ids": source_ids,
            "method": "Matched dataset, common baseline, held-out evaluation required",
            "status": "not_feasible", "feasibility_reason": "no matched evaluation dataset and labels",
            "experiment_type": None,
        }
    sources = evidence.get("sources") or []
    source_ids = [source["id"] for source in sources[:2]]
    citations = " ".join(f"[{source_id}]" for source_id in source_ids)
    topic = plan.get("topic") or plan.get("Title") or "the research question"
    return {
        "id": idea_id, "title": f"Design a measurable follow-up study on {topic}",
        "hypothesis": (f"A future study could test a measurable outcome relevant to {topic} "
                       f"using an independently verified dataset {citations}."),
        "rationale": ("The retrieved literature motivates a follow-up question, but the "
                      "outcome, dataset and design need independent validation. This is "
                      "a proposed direction, not a verified novelty claim."),
        "source_ids": source_ids, "method": "Dataset and evaluation design required",
        "status": "not_feasible", "feasibility_reason": "no matched evaluation dataset",
        "experiment_type": None,
    }


def propose_ideas(plan, evidence):
    """Always propose a testable direction; only validated data can enable execution."""
    datasets = evidence.get("datasets") or []
    ranked = sorted(datasets, key=lambda dataset: len(dataset.get("points") or []), reverse=True)
    if ranked:
        eligible_choices = [(dataset, *_eligible_dataset(dataset)) for dataset in ranked]
        selected = next((choice for choice in eligible_choices if choice[1]), eligible_choices[0])
        dataset, eligible, reason, _ = selected
        return {"ideas": [_idea(dataset, eligible, reason),
                          _literature_idea(plan, evidence, "I2")],
                "selected_idea_id": "I1" if eligible else None,
                "decision": "run_experiment" if eligible else "review_only",
                "reason": reason}
    idea = _literature_idea(plan, evidence, "I1")
    idea["feasibility_reason"] = "no validated relevant public series"
    return {"ideas": [idea], "selected_idea_id": None,
            "decision": "review_only", "reason": idea["feasibility_reason"]}


def _run_backtest(dataset, workspace_dir):
    eligible, reason, points = _eligible_dataset(dataset)
    if not eligible:
        raise ValueError(reason)
    rows = []
    values = [float(point["value"]) for point in points]
    for index in range(MIN_TRAIN_POINTS, len(points)):
        training = values[:index]
        observed = values[index]
        baseline = training[-1]
        trend = _linear_next(training)
        if not math.isfinite(trend):
            raise ValueError("non-finite trend prediction")
        rows.append({"source_id": dataset["id"], "year": points[index]["year"],
                     "observed": observed, "last_value_prediction": baseline,
                     "linear_trend_prediction": trend,
                     "last_value_absolute_error": abs(observed - baseline),
                     "linear_trend_absolute_error": abs(observed - trend)})
    metrics = {
        "last_value_mae": sum(row["last_value_absolute_error"] for row in rows) / len(rows),
        "linear_trend_mae": sum(row["linear_trend_absolute_error"] for row in rows) / len(rows),
    }
    metrics["mae_difference_baseline_minus_trend"] = (metrics["last_value_mae"]
                                                       - metrics["linear_trend_mae"])
    winner = ("linear_trend" if metrics["mae_difference_baseline_minus_trend"] > 0 else
              "last_value" if metrics["mae_difference_baseline_minus_trend"] < 0 else "tie")
    csv_path = os.path.join(workspace_dir, "experiment_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    years = [row["year"] for row in rows]
    ax.plot(years, [row["observed"] for row in rows], marker="o", label="Observed")
    ax.plot(years, [row["last_value_prediction"] for row in rows], marker=".",
            label="Last-value baseline")
    ax.plot(years, [row["linear_trend_prediction"] for row in rows], marker=".",
            label="Linear trend")
    ax.set_xlabel("Held-out year")
    ax.set_ylabel(dataset["unit"])
    ax.set_title("One-step expanding-window backtest")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(workspace_dir, "experiment_plot.png"), dpi=180)
    plt.close(fig)
    return {
        "status": "completed", "experiment_type": "annual_forecast_backtest",
        "source_id": dataset["id"], "dataset_title": dataset["title"],
        "entity": dataset["entity"], "unit": dataset["unit"],
        "method": ("For each held-out year after the first eight consecutive annual "
                   "observations, fit an ordinary least-squares line on preceding years "
                   "only and predict the next year; compare with the last observed value. "
                   "Report mean absolute error across all held-out years. No future "
                   "observations enter a training window."),
        "training_minimum": MIN_TRAIN_POINTS, "test_count": len(rows),
        "first_test_year": rows[0]["year"], "last_test_year": rows[-1]["year"],
        "metrics": metrics, "lower_mae": winner,
        "interpretation": ("The linear trend has lower observed one-step MAE in this "
                           "backtest." if winner == "linear_trend" else
                           "The last-value baseline has lower observed one-step MAE in "
                           "this backtest." if winner == "last_value" else
                           "The methods have equal observed one-step MAE in this backtest."),
        "limitations": ("This is a computational backtest on a published series, not a "
                        "new measurement, causal experiment, or proof of future forecasting "
                        "performance. The same series was used to formulate the question "
                        "and evaluate it; no statistical significance or global novelty "
                        "claim is made."),
        "predictions_file": "experiment_predictions.csv",
        "plot_file": "experiment_plot.png",
    }


def plan_and_run(plan, evidence, workspace_dir, *, model=None, data_path=None,
                 target=None, task=None):
    """Persist the decision and never let an experiment failure block a review."""
    proposal = propose_ideas(plan, evidence)
    evidence["research_proposal"] = proposal
    experiment = {"status": "not_run", "reason": proposal["reason"]}
    if proposal["decision"] == "run_experiment" and not data_path:
        idea = proposal["ideas"][0]
        dataset = next(item for item in evidence["datasets"]
                       if item["id"] == idea["source_ids"][0])
        try:
            experiment = _run_backtest(dataset, workspace_dir)
        except Exception as exc:
            experiment = {"status": "failed", "reason": str(exc)}
            proposal["decision"] = "review_only"
            proposal["reason"] = f"experiment failed: {exc}"
            idea["status"] = "not_feasible"
            idea["feasibility_reason"] = proposal["reason"]
            proposal["selected_idea_id"] = None
    if model is not None and experiment["status"] != "completed":
        if data_path and proposal["selected_idea_id"]:
            proposal["ideas"][0]["feasibility_reason"] = "A user-supplied CSV was selected for this run."
            proposal["selected_idea_id"] = None
        try:
            from ai_scientist.tabular_research import attempt_tabular_candidates

            new_ideas, experiment, screening_errors = attempt_tabular_candidates(
                plan, evidence, model, workspace_dir, f"I{len(proposal['ideas']) + 1}",
                data_path=data_path, target=target, task=task,
            )
            proposal["ideas"].extend(new_ideas)
            if screening_errors:
                proposal["screening_warnings"] = screening_errors
        except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
            experiment = {"status": "not_run", "reason": str(exc)[:450]}
        if experiment["status"] == "completed":
            idea = next(item for item in new_ideas if item["id"] == experiment["idea_id"])
            proposal.update(decision="run_experiment", selected_idea_id=idea["id"],
                            reason=idea["feasibility_reason"])
        else:
            proposal.update(decision="review_only", selected_idea_id=None, reason=experiment["reason"])
    if (model is not None and experiment["status"] == "completed"
            and experiment["experiment_type"] == "annual_forecast_backtest"):
        try:
            from ai_scientist.tabular_research import attempt_tabular_candidates

            additional, _, screening_errors = attempt_tabular_candidates(
                plan, evidence, model, workspace_dir, f"I{len(proposal['ideas']) + 1}",
                run_experiments=False,
            )
            proposal["ideas"].extend(additional)
            if screening_errors:
                proposal["screening_warnings"] = screening_errors
        except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
            proposal["screening_warnings"] = [
                "Additional idea generation unavailable: " + str(exc)[:300]
            ]
    evidence["experiment"] = experiment
    with open(os.path.join(workspace_dir, "research_proposal.json"), "w", encoding="utf-8") as stream:
        json.dump(proposal, stream, ensure_ascii=False, indent=2)
    with open(os.path.join(workspace_dir, "experiment_results.json"), "w", encoding="utf-8") as stream:
        json.dump(experiment, stream, ensure_ascii=False, indent=2)
    with open(os.path.join(workspace_dir, "evidence.json"), "w", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
    return proposal, experiment
