"""Analyze measured energy against no-lookahead reconstructed forecasts.

This is not exact replay: neither stored profiles, historical correction factors,
nor historical learned shares were recorded. Whole-runtime average discharge is
NOT used as if it were a fixed-horizon energy forecast.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from compare_forecast_stability import ROOT, baseline_module

from custom_components.battery_runtime import model as corrected


def make_model(module, fixture, snapshot, share=None, adjustment=1.0):
    values = list(fixture["profiles"][snapshot["profile"]])
    for slot, value in snapshot["updates"].items():
        values[int(slot)] = value
    model = module.BatteryRuntimeModel(module.ModelConfig(11.8, 30, "Europe/Lisbon"))
    timestamp = datetime.fromisoformat(snapshot["time"])
    for cell, value in zip(model.profile.cells, values, strict=True):
        cell.update(value, timestamp)
    for _ in range(3):
        model.battery_share.update(
            (snapshot["share"] if share is None else share) * 1000, 1000, timestamp
        )
    model._load_adjustment.value = adjustment
    return model


def run():
    fixture = json.loads(
        (ROOT / "tests/fixtures/forecast_stability_history.json").read_text()
    )
    baseline = baseline_module()
    output = {"method": fixture["method"], "snapshots": [], "revisions": []}
    for snapshot in fixture["snapshots"]:
        start = datetime.fromisoformat(snapshot["time"])
        model = make_model(corrected, fixture, snapshot)
        row = {
            "time": snapshot["time"],
            "share": snapshot["share"],
            "recorded_runtime_h": snapshot["observed_runtime_h"],
            "fixed_horizon_errors": [],
            "reconstructed_runtime_h": {},
        }
        for horizon, measured in snapshot["measured_battery_kwh"].items():
            end = start + timedelta(minutes=int(horizon))
            household_forecast = model.estimate_energy_kwh(start, end)
            old = household_forecast * snapshot["share"]
            new = household_forecast * max(1, snapshot["share"])
            row["fixed_horizon_errors"].append(
                {
                    "minutes": int(horizon),
                    "measured_house_kwh": snapshot["measured_house_kwh"][horizon],
                    "measured_battery_kwh": measured,
                    "baseline_predicted_kwh": old,
                    "corrected_predicted_kwh": new,
                    "baseline_error_kwh": old - measured,
                    "corrected_error_kwh": new - measured,
                }
            )
        for label, module in [("baseline", baseline), ("corrected", corrected)]:
            # Show the permitted correction range, not a fitted historical value.
            row["reconstructed_runtime_h"][label] = {
                str(adjustment): make_model(
                    module, fixture, snapshot, adjustment=adjustment
                )
                .predict(module.PredictionInput(start, snapshot["soc"], 1000, 1000))
                .runtime_hours
                for adjustment in [0.6, 1.0, 1.6]
            }
        output["snapshots"].append(row)
    for first, second in zip(
        fixture["snapshots"][::2], fixture["snapshots"][1::2], strict=True
    ):
        row = {"start": first["time"], "end": second["time"]}
        elapsed = (
            datetime.fromisoformat(second["time"])
            - datetime.fromisoformat(first["time"])
        ).total_seconds() / 3600
        row["recorded_depletion_revision_h"] = (
            elapsed + second["observed_runtime_h"] - first["observed_runtime_h"]
        )
        for label, module in [("baseline", baseline), ("corrected", corrected)]:
            predictions = []
            for snapshot in [first, second]:
                m = make_model(module, fixture, snapshot)
                predictions.append(
                    m.predict(
                        module.PredictionInput(
                            datetime.fromisoformat(snapshot["time"]),
                            snapshot["soc"],
                            1000,
                            1000,
                        )
                    ).runtime_hours
                )
            row[f"{label}_depletion_revision_h"] = (
                elapsed + predictions[1] - predictions[0]
            )
        # Freeze the earlier ratio to isolate its change from the updated
        # profile, actual SoC change and shifted forecast start.
        frozen = make_model(baseline, fixture, second, share=first["share"]).predict(
            baseline.PredictionInput(
                datetime.fromisoformat(second["time"]), second["soc"], 1000, 1000
            )
        )
        changed = make_model(baseline, fixture, second).predict(
            baseline.PredictionInput(
                datetime.fromisoformat(second["time"]), second["soc"], 1000, 1000
            )
        )
        row["ratio_change_alone_runtime_effect_h"] = (
            changed.runtime_hours - frozen.runtime_hours
        )
        output["revisions"].append(row)
    hourly = [
        error
        for row in output["snapshots"]
        for error in row["fixed_horizon_errors"]
        if error["minutes"] == 60
    ]
    output["nonoverlapping_one_hour_mae_kwh"] = {
        name: sum(abs(error[f"{name}_error_kwh"]) for error in hourly) / len(hourly)
        for name in ["baseline", "corrected"]
    }
    output["soc_validation"] = fixture["soc_validation"]
    return output


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
