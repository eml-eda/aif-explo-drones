from __future__ import annotations


DETECTOR_DIRECTORY_LABELS = {
    "mobilenet_ssd": "mbnetv2",
    "yolo_nano": "yolonano",
}


def evaluation_directory_name(
    *,
    policy: str,
    detector: str,
    environment: str,
    timestamp: str,
) -> str:
    """Return the canonical directory name for one evaluation campaign."""
    try:
        detector_label = DETECTOR_DIRECTORY_LABELS[str(detector)]
    except KeyError as exc:
        raise ValueError(f"No result-directory label for detector {detector!r}.") from exc

    return f"eval_{policy}_{detector_label}_{environment}_{timestamp}"
