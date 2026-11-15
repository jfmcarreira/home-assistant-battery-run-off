"""Controlled comparison, not a reconstruction of historical HA model snapshots.

Run from any directory with:
    .venv/bin/python tools/compare_forecast_stability.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from custom_components.battery_runtime import model as corrected  # noqa: E402


def baseline_module() -> ModuleType:
    """Load the verified deployed baseline without modifying the working tree."""
    source = subprocess.check_output(
        ["git", "show", "031feb0:custom_components/battery_runtime/model.py"],
        cwd=ROOT,
        text=True,
    )
    module = ModuleType("forecast_stability_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, "baseline_model.py", "exec"), module.__dict__)
    return module


def forecast(module, timestamp, soc, ratio, adjustment=1.0):
    """Keep every input except the explicitly varied factors fixed."""
    model = module.BatteryRuntimeModel(module.ModelConfig(11.8, 30, "Europe/Lisbon"))
    for index, cell in enumerate(model.profile.cells):
        cell.update(2000 if 18 * 4 <= index < 20 * 4 else 300, timestamp)
    for _ in range(3):
        model.battery_share.update(2000 * ratio, 2000, timestamp)
    model._load_adjustment.value = adjustment
    result = model.predict(module.PredictionInput(timestamp, soc, 2000, 2000))
    # Three-hour energy from 18:00: 2 h * 2 kW + 1 h * 0.3 kW.
    effective = getattr(
        model.battery_share, "forecast_multiplier", model.battery_share.value
    )
    return result, effective


def main() -> None:
    baseline = baseline_module()
    start = datetime(2026, 11, 12, 18, tzinfo=UTC)
    print(
        "CONTROLLED SIMULATION: complete 96-slot profile, capacity 11.8 kWh, reserve 30%."
    )
    print(
        "Dinner 18:00-20:00 at 2 kW; all other slots 0.3 kW. No source histories replayed."
    )
    print("Only evaluate above 50% SoC; do not use behavior at the inverter cutoff.")
    for label, module in [("baseline", baseline), ("corrected", corrected)]:
        print(f"\n{label}:")
        for ratio in [0.5, 0.704097458322842, 1.0, 1.2]:
            for adjustment in [1.0, 1.6]:
                result, factor = forecast(module, start, 72, ratio, adjustment)
                print(
                    f"observed_share={ratio:.3f} factor={factor:.3f} "
                    f"adjustment={adjustment:.1f} runtime={result.runtime_hours:.3f}h "
                    f"depletion={result.depletion_time.isoformat()}"
                )
        before, factor = forecast(module, start, 72, 0.704097458322842)
        # Hypothetical next hour: the battery supplies 2 kWh. No learning
        # noise or temporary-load correction; partial supply ratio becomes one.
        after_soc = 72 - 2 / 11.8 * 100
        assert after_soc > 50
        after, _ = forecast(module, start + timedelta(hours=1), after_soc, 1.0)
        revision = (after.depletion_time - before.depletion_time).total_seconds() / 3600
        print(f"After 1 h: SoC={after_soc:.3f}%, depletion revision={revision:+.3f}h")
        print(
            f"Initial 3-hour baseline demand={4.3 * factor:.3f} kWh; "
            f"full-load scenario truth=4.300 kWh; error={4.3 * factor - 4.3:+.3f} kWh"
        )


if __name__ == "__main__":
    main()
