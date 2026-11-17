import tempfile
import unittest
from pathlib import Path

from configuration.detector_config import (
    resolve_detector_selection,
    validate_model_profile,
)
from configuration.model_registry import MODEL_PROFILES, get_model_profile
from configuration.registry import normalize_detector_id


class DetectorConfigurationTests(unittest.TestCase):
    def test_unknown_detector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown detector"):
            normalize_detector_id("other_detector")

    def test_final_detector_aliases_normalize(self):
        self.assertEqual(normalize_detector_id("nano"), "yolo_nano")
        self.assertEqual(normalize_detector_id("mobilenet"), "mobilenet_ssd")

    def test_easy_model_uses_canonical_checkpoint_path(self):
        profile = get_model_profile("easy", "yolo_nano")
        self.assertEqual(
            profile.model_path,
            "detection_model_training/checkpoints/easy/yolo_nano.pt",
        )
        selection = resolve_detector_selection(
            environment="easy",
            detector="yolo_nano",
            require_model_file=False,
        )
        self.assertEqual(selection.model_path_report, profile.model_path)
        self.assertEqual(selection.backend, "yolo")
        self.assertEqual(selection.model_size, "nano")

    def test_explicit_checkpoint_override_resolves_for_unverified_profile(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            selection = resolve_detector_selection(
                environment="hard",
                detector="mobilenet_ssd",
                model_path_override=tmp.name,
            )
        self.assertEqual(selection.detector_id, "mobilenet_ssd")
        self.assertEqual(
            selection.class_names,
            ("target_bottle", "target_can", "desk"),
        )
        self.assertTrue(selection.model_path.endswith(".pt"))

    def test_all_registered_profiles_match_environment_class_schema(self):
        for profile in MODEL_PROFILES.values():
            validate_model_profile(profile)

    def test_detector_metadata_keeps_inference_and_claim_concepts_separate(self):
        selection = resolve_detector_selection(
            environment="medium",
            detector="yolo_nano",
            confidence_threshold=0.45,
            iou_threshold=0.40,
            image_size=320,
            device="cuda:0",
            require_model_file=False,
        )
        metadata = selection.metadata()
        self.assertEqual(metadata["detector_used"], "yolo_nano")
        self.assertEqual(metadata["detector_confidence_threshold"], 0.45)
        self.assertEqual(metadata["detector_iou_threshold"], 0.40)
        self.assertEqual(metadata["detector_image_size"], 320)
        self.assertNotIn("claim_confidence_threshold", metadata)


if __name__ == "__main__":
    unittest.main()
