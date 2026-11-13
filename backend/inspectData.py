#sanity check
import numpy as np
import pandas as pd
from pathlib import Path

#dataset path
data_dir=Path("data")
train_dir=data_dir/"train"
test_dir=data_dir/"test"
labels_files=data_dir/"labeled_anomalies.csv"

def inspect_dataset():
    print("dataset inspection")
    for path in [train_dir,test_dir]:
        if not path.exists():
            print(f"error:directory missing: {path}")
            return
    if not labels_files.exists():
        print("Error: file not found")
        return
    
    # finding channel files
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))

    print(f"training channels: {len(train_files)}")
    print(f"testing channels: {len(test_files)}")

    # Load anomaly labels
    labels = pd.read_csv(labels_files)

    print(f"\nTotal labeled channels: {len(labels)}")
    print("\nLabel columns:")
    print(labels.columns.tolist())

    # Check channel consistency
    train_channels = {f.stem for f in train_files}
    test_channels = {f.stem for f in test_files}

    print("\nTrain/Test channel comparison:")
    print("Missing in train:", len(test_channels - train_channels))
    print("Missing in test:", len(train_channels - test_channels))
    
    # Inspect a sample channel
    if not train_files or not test_files:
        print("\nerror: no channel files found!")
        return

    channel = "P-1" if "P-1" in train_channels else train_files[0].stem

    train = np.load(train_dir / f"{channel}.npy")
    test = np.load(test_dir / f"{channel}.npy")

    print(f"\n CHANNEL: {channel} ")

    print("\nTraining data:")
    print("Shape:", train.shape)
    print("Data type:", train.dtype)
    print("First 5 telemetry values:", train[:5, 0])

    print("\nTesting data:")
    print("Shape:", test.shape)
    print("First 5 telemetry values:", test[:5, 0])

    # Check anomaly labels
    channel_labels = labels[labels["chan_id"] == channel]

    if not channel_labels.empty:
        row = channel_labels.iloc[0]

        print("\nSpacecraft:", row["spacecraft"])
        print("Anomaly sequences:", row["anomaly_sequences"])
        print("Anomaly class:", row["class"])
        print("Number of anomaly values:", row["num_values"])
    else:
        print("\nNo anomaly labels found for this channel.")

    # Data validation
    print("\n DATA VALIDATION ")
    print("Training contains NaN:", np.isnan(train).any())
    print("Testing contains NaN:", np.isnan(test).any())
    print("Training contains Inf:", np.isinf(train).any())
    print("Testing contains Inf:", np.isinf(test).any())




if __name__ == "__main__":
    inspect_dataset()





