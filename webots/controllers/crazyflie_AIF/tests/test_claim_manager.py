import unittest

from task.claim_manager import ClaimManager


class ClaimManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = ClaimManager(
            confidence_threshold=0.45,
            target_class_names={"target_bottle", "target_can"},
            target_class_ids={0, 1},
        )

    @staticmethod
    def event(confidence=0.45, class_name="target_bottle", class_id=0, cell=(3, 4)):
        return {
            "cell": cell,
            "confidence": confidence,
            "class_name": class_name,
            "class_id": class_id,
        }

    def test_threshold_is_inclusive(self):
        self.assertEqual(self.manager.update([self.event(confidence=0.449)], perception_processed=True), [])
        claims = self.manager.update([self.event(confidence=0.45)], perception_processed=True)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].key, (3, 4, "target_bottle"))

    def test_skipped_perception_step_creates_no_claim(self):
        claims = self.manager.update([self.event(confidence=1.0)], perception_processed=False)
        self.assertEqual(claims, [])
        self.assertEqual(self.manager.claimed_object_keys, set())

    def test_context_class_is_not_claimed(self):
        claims = self.manager.update(
            [self.event(confidence=0.99, class_name="desk", class_id=2)],
            perception_processed=True,
        )
        self.assertEqual(claims, [])

    def test_claim_is_permanent_and_not_duplicated(self):
        first = self.manager.update([self.event(confidence=0.9)], perception_processed=True)
        second = self.manager.update([self.event(confidence=0.9)], perception_processed=True)
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(self.manager.claimed_cells, {(3, 4)})

    def test_same_cell_can_hold_distinct_class_keys(self):
        self.manager.update([self.event(class_name="target_bottle", class_id=0)], perception_processed=True)
        claims = self.manager.update(
            [self.event(class_name="target_can", class_id=1)],
            perception_processed=True,
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(
            self.manager.claimed_object_keys,
            {(3, 4, "target_bottle"), (3, 4, "target_can")},
        )


if __name__ == "__main__":
    unittest.main()
