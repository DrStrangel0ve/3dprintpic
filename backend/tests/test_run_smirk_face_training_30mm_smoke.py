import unittest

from backend.benchmark.run_smirk_face_training_30mm_smoke import (
    physical_decision,
)


def _row(split, baseline, candidate, *, passed=True):
    return {
        "split": split,
        "baseline_quality": {
            "combined_named_part_failures": baseline,
        },
        "candidate_quality": {
            "combined_named_part_failures": candidate,
        },
        "checks": {"passed": passed},
    }


class SMIRKFaceTraining30mmSmokeTests(unittest.TestCase):
    def test_decision_accepts_three_split_non_regressing_rows(self):
        decision = physical_decision(
            [
                _row("train", 8, 7),
                _row("validation", 6, 6),
                _row("sealed", 5, 5),
            ]
        )

        self.assertTrue(decision["passed"])

    def test_decision_rejects_named_part_regression(self):
        decision = physical_decision(
            [
                _row("train", 8, 7),
                _row("validation", 6, 7),
                _row("sealed", 5, 5),
            ]
        )

        self.assertFalse(decision["no_30mm_named_part_regression"])
        self.assertFalse(decision["passed"])


if __name__ == "__main__":
    unittest.main()
