import unittest

from policies.pseudo_random import PseudoRandomPolicy


class PseudoRandomPolicyTests(unittest.TestCase):
    @staticmethod
    def turn_targets(seed: int, count: int = 8):
        policy = PseudoRandomPolicy(seed=seed)
        targets = []
        for _ in range(count):
            policy.sm.has_turn_target = False
            policy.compute_action(
                {
                    "ranges": {"front": 0.5},
                    "imu": {"yaw": 0.0},
                    "step": 0,
                }
            )
            targets.append(policy.sm.target_yaw)
        return targets

    def test_same_seed_reproduces_turn_sequence(self):
        self.assertEqual(self.turn_targets(17), self.turn_targets(17))

    def test_different_seed_changes_turn_sequence(self):
        self.assertNotEqual(self.turn_targets(17), self.turn_targets(18))


if __name__ == "__main__":
    unittest.main()
