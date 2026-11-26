
import json
import numpy as np
from pathlib import Path

ERRORS_PATH = Path("models/orby_validation_errors.npz")
OUTPUT_PATH = Path("models/orby_thresholds.json")

errors_file = np.load(ERRORS_PATH)

thresholds = {}

print("ORBY THRESHOLD CALIBRATION")
print("-" * 75)

for channel in sorted(errors_file.files):
    errors = errors_file[channel]

    threshold_99 = float(np.percentile(errors, 99))
    threshold_995 = float(np.percentile(errors, 99.5))
    threshold_999 = float(np.percentile(errors, 99.9))

    thresholds[channel] = {
        "threshold": threshold_995,
        "p99": threshold_99,
        "p99_5": threshold_995,
        "p99_9": threshold_999,
        "validation_samples": int(len(errors)),
        "confidence": (
            "low" if len(errors) < 200 else "reasonable"
        )
    }

    print(
        f"{channel:6} | "
        f"N={len(errors):4} | "
        f"P99={threshold_99:9.4f} | "
        f"P99.5={threshold_995:9.4f} | "
        f"P99.9={threshold_999:9.4f}"
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w") as file:
    json.dump(thresholds, file, indent=4)

print("\nTHRESHOLD CALIBRATION COMPLETE")
print("Channels:", len(thresholds))
print("Primary threshold: P99.5")
print("Saved:", OUTPUT_PATH)
