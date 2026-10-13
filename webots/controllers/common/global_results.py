from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .shared_state import atomic_write_json, read_json


def _sample_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values))


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def build_global_result_entry(
    *,
    campaign_id: str,
    environment: str,
    eval_report: dict[str, Any],
) -> dict[str, Any]:
    """Build the compact cross-campaign metrics entry from an evaluation report.

    Target recovery and visited-cell metrics use all runs. Time to success uses
    only runs whose terminal reason is ``done``, matching the success-rate
    definition used by the evaluation supervisors. Percentage standard
    deviations are stored in percentage points.
    """
    run_metrics = eval_report.get("run_metrics")
    if not isinstance(run_metrics, list) or not run_metrics:
        raise ValueError("Evaluation report has no non-empty run_metrics list.")

    successful_runs = [
        run for run in run_metrics if str(run.get("reason", "")) == "done"
    ]
    recovery_percent = [
        float(run["detection_rate_percent"]) for run in run_metrics
    ]
    success_time_s = [float(run["elapsed_s"]) for run in successful_runs]
    visited_percent = [
        100.0 * float(run["coverage_ratio"]) for run in run_metrics
    ]

    return {
        "campaign_id": str(campaign_id),
        "environment": str(environment),
        "policy": str(eval_report.get("policy", "unknown")),
        "detector": str(eval_report.get("detector_used", "unknown")),
        "num_runs": int(len(run_metrics)),
        "success_rate_percent": _rounded(
            100.0 * float(len(successful_runs)) / float(len(run_metrics))
        ),
        "target_recovery_percent": {
            "mean": _rounded(_mean(recovery_percent)),
            "std": _rounded(_sample_std(recovery_percent)),
        },
        "time_to_success_s": {
            "mean": _rounded(_mean(success_time_s)),
            "std": _rounded(_sample_std(success_time_s)),
        },
        "cell_visited_percent": {
            "mean": _rounded(_mean(visited_percent)),
            "std": _rounded(_sample_std(visited_percent)),
        },
    }


def update_global_result(
    path: str | Path,
    *,
    campaign_id: str,
    environment: str,
    eval_report: dict[str, Any],
) -> dict[str, Any]:
    """Atomically append or replace one campaign in ``global_results.json``."""
    output_path = Path(path)
    document = read_json(output_path, {"results": []})
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        raise ValueError(
            f"{output_path} must be a JSON object containing a 'results' array."
        )

    entry = build_global_result_entry(
        campaign_id=campaign_id,
        environment=environment,
        eval_report=eval_report,
    )

    results = list(document["results"])
    for index, existing in enumerate(results):
        if (
            isinstance(existing, dict)
            and str(existing.get("campaign_id", "")) == str(campaign_id)
        ):
            results[index] = entry
            break
    else:
        results.append(entry)

    output = {"results": results}
    atomic_write_json(output_path, output)
    return entry
