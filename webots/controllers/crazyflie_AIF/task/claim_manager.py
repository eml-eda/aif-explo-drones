from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class ControllerClaim:
    cx: int
    cy: int
    class_name: str
    class_id: int
    confidence: float

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.cx, self.cy, self.class_name)


class ClaimManager:
    """Create permanent controller claims from qualifying target detections."""

    def __init__(
        self,
        *,
        confidence_threshold: float,
        target_class_names: Iterable[str],
        target_class_ids: Iterable[int],
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.target_class_names = {str(name) for name in target_class_names}
        self.target_class_ids = {int(class_id) for class_id in target_class_ids}
        self._claimed_object_keys: set[tuple[int, int, str]] = set()

    @property
    def claimed_object_keys(self) -> set[tuple[int, int, str]]:
        return set(self._claimed_object_keys)

    @property
    def claimed_cells(self) -> set[tuple[int, int]]:
        return {(cx, cy) for (cx, cy, _) in self._claimed_object_keys}

    def _is_target_class(self, class_name: str, class_id: Optional[int]) -> bool:
        if class_name in self.target_class_names:
            return True
        if class_id is None:
            return False
        return int(class_id) in self.target_class_ids

    def update(self, detected_target_events: Iterable[dict], *, perception_processed: bool) -> list[ControllerClaim]:
        if not perception_processed:
            return []

        new_claims: list[ControllerClaim] = []

        for event in detected_target_events:
            cell = event.get("cell", None)
            if cell is None:
                continue

            confidence = float(event.get("confidence", 0.0))
            if confidence < self.confidence_threshold:
                continue

            raw_class_id = event.get("class_id", None)
            class_id = None if raw_class_id is None else int(raw_class_id)
            class_name = str(event.get("class_name", event.get("label", "unknown")))

            if not self._is_target_class(class_name, class_id):
                continue

            claim = ControllerClaim(
                cx=int(cell[0]),
                cy=int(cell[1]),
                class_name=class_name,
                class_id=-1 if class_id is None else class_id,
                confidence=confidence,
            )

            if claim.key in self._claimed_object_keys:
                continue

            self._claimed_object_keys.add(claim.key)
            new_claims.append(claim)

        return new_claims
