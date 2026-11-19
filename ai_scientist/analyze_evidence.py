"""Reproducible descriptive analysis of retrieved numerical time series."""

import math


ANALYSIS_METHOD = (
    "For each validated series, compute the first-to-last net change, its average per elapsed "
    "calendar year, observed minimum and maximum, and changes between directly consecutive "
    "years, including the largest observed one-year decline when present. Report percentage "
    "change only for a positive starting value and nonnegative ending "
    "value. Compare the latest observation with the earliest available observation within the "
    "preceding five calendar years when this is a distinct interval. Missing years are not "
    "interpolated. These are descriptive statistics, not tests of significance or causation."
)


def _direction(change):
    if change > 0:
        return "increase"
    if change < 0:
        return "decrease"
    return "no change"


def analyze_series(dataset):
    """Calculate only quantities derivable from a single source's observed values."""
    points_by_year = {}
    for point in dataset["points"]:
        year = point["year"]
        value = point["value"]
        if isinstance(year, bool) or not isinstance(year, int) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid observation in {dataset['id']}.")
        if not math.isfinite(value) or year in points_by_year:
            raise ValueError(f"Non-finite or duplicate observation in {dataset['id']}.")
        points_by_year[year] = float(value)
    if len(points_by_year) < 2:
        raise ValueError(f"At least two observations are required for {dataset['id']}.")

    points = sorted(points_by_year.items())
    first_year, first_value = points[0]
    last_year, last_value = points[-1]
    span = last_year - first_year
    if span <= 0:
        raise ValueError(f"The observation years are invalid for {dataset['id']}.")
    change = last_value - first_value
    low_year, low_value = min(points, key=lambda item: (item[1], item[0]))
    high_year, high_value = max(points, key=lambda item: (item[1], -item[0]))

    adjacent_changes = [{"from_year": year, "to_year": next_year,
                         "change": next_value - value}
                        for (year, value), (next_year, next_value) in zip(points, points[1:])
                        if next_year - year == 1]
    increases = [pair for pair in adjacent_changes if pair["change"] > 0]
    decreases = [pair for pair in adjacent_changes if pair["change"] < 0]
    recent_points = [point for point in points if point[0] >= last_year - 5]
    recent = None
    if len(recent_points) >= 2 and recent_points[0][0] > first_year:
        recent_change = last_value - recent_points[0][1]
        recent = {
            "first_year": recent_points[0][0], "first_value": recent_points[0][1],
            "last_year": last_year, "last_value": last_value,
            "net_change": recent_change, "direction": _direction(recent_change),
        }

    return {
        "source_id": dataset["id"], "title": dataset["title"],
        "entity": dataset["entity"], "unit": dataset["unit"],
        "observation_count": len(points),
        "first": {"year": first_year, "value": first_value},
        "last": {"year": last_year, "value": last_value},
        "net_change": change, "direction": _direction(change),
        "percent_change": (change / first_value * 100
                           if first_value > 0 and last_value >= 0 else None),
        "average_net_change_per_year": change / span,
        "minimum": {"year": low_year, "value": low_value},
        "maximum": {"year": high_year, "value": high_value},
        "consecutive_year_pairs": len(adjacent_changes),
        "increasing_pairs": len(increases),
        "decreasing_pairs": len(decreases),
        "unchanged_pairs": len(adjacent_changes) - len(increases) - len(decreases),
        "largest_one_year_increase": max(increases, key=lambda pair: pair["change"])
        if increases else None,
        "largest_one_year_decrease": min(decreases, key=lambda pair: pair["change"])
        if decreases else None,
        "recent_five_year_window": recent,
    }


def analyze_datasets(datasets):
    return {"method": ANALYSIS_METHOD,
            "series": [analyze_series(dataset) for dataset in datasets]}
