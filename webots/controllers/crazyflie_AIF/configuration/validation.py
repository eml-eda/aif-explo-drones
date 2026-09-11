from __future__ import annotations

from typing import Mapping, Optional

from .experiment_config import ExperimentSelection
from .registry import normalize_environment_id


def _environment_from_init(experiment_init: Mapping) -> Optional[str]:
    for key in ("environment", "scenario", "target_setup"):
        value = experiment_init.get(key, None)
        if value in (None, ""):
            continue
        try:
            return normalize_environment_id(str(value))
        except ValueError:
            continue
    return None


def _run_mode_from_init(experiment_init: Mapping) -> Optional[str]:
    value = experiment_init.get("run_mode", None)
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def validate_runtime_selection(
    selection: ExperimentSelection,
    experiment_init: Mapping,
) -> None:
    """Fail fast when a launched supervisor disagrees with controller config.

    Missing identity fields are tolerated for compatibility with old shared-state
    payloads. Newly refactored supervisors write both `environment` and `run_mode`,
    so mismatches are detected before the episode proceeds.
    """
    supervisor_environment = _environment_from_init(experiment_init)
    supervisor_run_mode = _run_mode_from_init(experiment_init)

    if (
        supervisor_environment is not None
        and supervisor_environment != selection.environment
    ):
        raise RuntimeError(
            "Experiment environment mismatch: "
            f"controller='{selection.environment}', "
            f"supervisor='{supervisor_environment}'."
        )

    if supervisor_run_mode is not None and supervisor_run_mode != selection.run_mode:
        raise RuntimeError(
            "Experiment run-mode mismatch: "
            f"controller='{selection.run_mode}', "
            f"supervisor='{supervisor_run_mode}'."
        )
