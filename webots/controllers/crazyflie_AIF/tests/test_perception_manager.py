import unittest

from vision.perception_manager import PerceptionSchedule


class PerceptionScheduleTests(unittest.TestCase):
    def test_time_schedule_matches_existing_0625_second_cadence(self):
        schedule = PerceptionSchedule(mode="time", time_interval_s=0.625)
        self.assertFalse(
            schedule.should_process(
                step_count=10,
                time_now_s=1.624,
                last_step=0,
                last_time_s=1.0,
            )
        )
        self.assertTrue(
            schedule.should_process(
                step_count=11,
                time_now_s=1.625,
                last_step=0,
                last_time_s=1.0,
            )
        )

    def test_metadata_preserves_detector_limitation_format(self):
        schedule = PerceptionSchedule(mode="time", time_interval_s=0.625)
        self.assertEqual(
            schedule.metadata(),
            {
                "mode": "time",
                "frames_interval": None,
                "time_interval_s": 0.625,
            },
        )


if __name__ == "__main__":
    unittest.main()
