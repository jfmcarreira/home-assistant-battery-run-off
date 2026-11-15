"""Regression checks for the recorded-history analysis, not an exact live replay."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_history_reconstruction_explains_revisions_without_claiming_uniform_accuracy():
    output = json.loads(
        subprocess.check_output(
            [sys.executable, "tools/analyze_forecast_history.py"], cwd=ROOT, text=True
        )
    )
    nov12, nov13 = output["revisions"]
    # The independently reconstructed pre-event model explains the scale and
    # direction; it is intentionally not fitted to the recorded runtime.
    assert (
        abs(
            nov12["baseline_depletion_revision_h"]
            - nov12["recorded_depletion_revision_h"]
        )
        < 1
    )
    assert (
        abs(
            nov13["baseline_depletion_revision_h"]
            - nov13["recorded_depletion_revision_h"]
        )
        < 0.1
    )
    assert nov12["ratio_change_alone_runtime_effect_h"] < -8
    assert nov13["ratio_change_alone_runtime_effect_h"] < -3
    for row in output["revisions"]:
        assert abs(row["corrected_depletion_revision_h"]) < 0.3

    # Do not replace real errors with assertions about smooth output: improvement
    # is mixed, including a worse next-hour prediction on November 13 at 17:00.
    first_errors = output["snapshots"][0]["fixed_horizon_errors"]
    assert first_errors[1]["measured_battery_kwh"] == pytest.approx(0.724213114)
    assert abs(first_errors[1]["baseline_error_kwh"]) == pytest.approx(0.38497817)
    assert abs(first_errors[1]["corrected_error_kwh"]) == pytest.approx(0.08902073)
    other_errors = output["snapshots"][2]["fixed_horizon_errors"]
    assert other_errors[1]["measured_battery_kwh"] == pytest.approx(0.360391852)
    assert abs(other_errors[1]["corrected_error_kwh"]) > abs(
        other_errors[1]["baseline_error_kwh"]
    )
    assert output["nonoverlapping_one_hour_mae_kwh"]["baseline"] == pytest.approx(
        0.23962458
    )
    assert output["nonoverlapping_one_hour_mae_kwh"]["corrected"] == pytest.approx(
        0.20759088
    )
    # Exclude November 12 19:00-20:00: the actual battery reaches the cutoff.
    assert len(output["snapshots"][1]["fixed_horizon_errors"]) == 1
    for window in output["soc_validation"]:
        assert window["minimum_soc"] > 50
        assert window["invalid_states"] == 0
        assert not window["truncated"]
