import pytest

from backend.benchmark.evaluate_lam_small_face_exact_gate import (
    _decision_from_variants,
    _verify_input_hashes,
)


def _variant(*, eligible: bool, passed: int, failures: int, alpha: str) -> dict:
    return {
        "variant_id": alpha,
        "eligible": eligible,
        "comparison": {
            "passed_checks": passed,
            "total_checks": 42,
        },
        "quality": {
            "combined_part_failures": failures,
            "shape_correlation": 0.80,
            "gradient_correlation": 0.62,
            "normalized_rmse": 0.17,
        },
        "checks": {
            "all_six_part_metrics_non_regressing": eligible,
            "background_value_exact": True,
        },
    }


def test_decision_requires_an_eligible_all_part_variant():
    ordered, decision = _decision_from_variants(
        [
            _variant(eligible=False, passed=41, failures=9, alpha="weak"),
            _variant(eligible=True, passed=42, failures=8, alpha="strong"),
        ]
    )

    assert ordered[0]["variant_id"] == "strong"
    assert decision["eligible_for_30mm_stl_replay"] is True
    assert decision["next_action"] == "run-bounded-30mm-stl-replay"


def test_decision_holds_best_noneligible_variant():
    _ordered, decision = _decision_from_variants(
        [_variant(eligible=False, passed=15, failures=10, alpha="weak")]
    )

    assert decision["status"] == "hold"
    assert decision["selected_passed_part_metric_checks"] == 15
    assert decision["next_action"] == "close-lam-face-depth-lane"


def test_decision_rejects_empty_variant_set():
    with pytest.raises(ValueError, match="no variants"):
        _decision_from_variants([])


def test_input_hash_verifier_rejects_incomplete_schema():
    with pytest.raises(ValueError, match="schema changed"):
        _verify_input_hashes({})
