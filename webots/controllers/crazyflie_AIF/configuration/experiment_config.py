from __future__ import annotations

from dataclasses import dataclass

from .registry import get_environment_spec, normalize_environment_id, normalize_policy_id


VALID_RUN_MODES = {"single", "evaluation"}


@dataclass(frozen=True)
class ExperimentSelection:
    environment: str
    run_mode: str
    policy: str

    @property
    def environment_spec(self):
        return get_environment_spec(self.environment)


def resolve_experiment_selection(
    *,
    environment: str,
    run_mode: str,
    policy: str,
) -> ExperimentSelection:
    environment_id = normalize_environment_id(environment)
    policy_id = normalize_policy_id(policy)
    mode = str(run_mode).strip().lower()

    if mode not in VALID_RUN_MODES:
        raise ValueError(
            f"Unknown run mode '{run_mode}'. Available: {', '.join(sorted(VALID_RUN_MODES))}"
        )

    env_spec = get_environment_spec(environment_id)
    if policy_id not in env_spec.supported_policies:
        raise ValueError(
            f"Policy '{policy_id}' is not supported in environment '{environment_id}'. "
            f"Supported: {', '.join(env_spec.supported_policies)}"
        )

    return ExperimentSelection(
        environment=environment_id,
        run_mode=mode,
        policy=policy_id,
    )
