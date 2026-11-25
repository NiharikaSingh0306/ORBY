
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_DIR = DATA_DIR / "train"


def analyze_all_channels():
    results = []

    for file_path in sorted(TRAIN_DIR.glob("*.npy")):
        channel = file_path.stem
        data = np.load(file_path)

        telemetry = data[:, 0]
        commands = data[:, 1:]

        constant_commands = np.sum(np.std(commands, axis=0) == 0)
        binary_commands = sum(
            len(np.unique(commands[:, i])) <= 2
            for i in range(commands.shape[1])
        )

        results.append({
            "channel": channel,
            "samples": len(data),
            "features": data.shape[1],
            "telemetry_min": np.min(telemetry),
            "telemetry_max": np.max(telemetry),
            "telemetry_mean": np.mean(telemetry),
            "telemetry_std": np.std(telemetry),
            "constant_commands": int(constant_commands),
            "binary_commands": int(binary_commands),
            "nan_values": int(np.isnan(data).sum()),
            "inf_values": int(np.isinf(data).sum())
        })

    df = pd.DataFrame(results)

    print("\nALL CHANNELS SUMMARY")
    print("=" * 75)
    print("Total channels:", len(df))
    print("\nSummary statistics:")
    print(df.describe().round(4).to_string())

    print("\nChannels with constant command features:")
    print(df[["channel", "constant_commands"]].to_string(index=False))

    output_path = Path(__file__).resolve().parent / "feature_summary.csv"
    df.to_csv(output_path, index=False)

    print(f"\nDetailed report saved to: {output_path}")


def compare_train_test_features():
    test_dir = DATA_DIR / "test"
    mismatches = []

    for train_file in sorted(TRAIN_DIR.glob("*.npy")):
        channel = train_file.stem
        test_file = test_dir / train_file.name

        if not test_file.exists():
            mismatches.append((channel, "Missing test file"))
            continue

        train = np.load(train_file, mmap_mode="r")
        test = np.load(test_file, mmap_mode="r")

        if train.shape[1] != test.shape[1]:
            mismatches.append(
                (channel, train.shape[1], test.shape[1])
            )

    print("\nTRAIN/TEST FEATURE COMPARISON")
    print("=" * 50)

    if mismatches:
        for item in mismatches:
            print("Mismatch:", item)
    else:
        print("All channels have matching train/test feature counts.")



if __name__ == "__main__":
    analyze_all_channels()
    compare_train_test_features()
