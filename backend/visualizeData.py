
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("data")
output_dir = Path("plots")
train_dir= data_dir/ "train"
test_dir = data_dir/ "test"
labels_file = data_dir/ "labeled_anomalies.csv"
output_dir.mkdir(exist_ok=True)

def parse_sequences(value):
    return ast.literal_eval(value)

def parse_classes(value):
    return re.findall(r"point|contextual", str(value).lower())

def find_channel(labels, anomaly_type):
    for _, row in labels.iterrows():
        classes = parse_classes(row["class"])
        sequences = parse_sequences(row["anomaly_sequences"])

        if anomaly_type in classes:
            return row, classes, sequences

    raise ValueError(f"No {anomaly_type} anomaly found!")

def plot_channel(row, classes, sequences, anomaly_type):

    channel = row["chan_id"]

    train = np.load(train_dir/ f"{channel}.npy")
    test = np.load(test_dir/ f"{channel}.npy")

    train_signal = train[:, 0]
    test_signal = test[:, 0]

    print(f"\nPlotting {channel}")
    print(f"Spacecraft: {row['spacecraft']}")
    print(f"Anomaly type: {anomaly_type}")

    fig, axes = plt.subplots(2, 1,
        figsize=(15, 9),
        gridspec_kw={"height_ratios": [1, 1.5]}
    )

    fig.suptitle(
        f"ORBY | {row['spacecraft']} | Channel {channel}",
        fontsize=16,
        fontweight="bold"
    )

    # training telemetry
    axes[0].plot(
        train_signal,
        color="#3B82F6",
        linewidth=0.8
    )

    axes[0].set_title("Training Telemetry (Normal Training Data)")
    axes[0].set_ylabel("Telemetry Value")
    axes[0].grid(alpha=0.2)

    # testing telemetry
    axes[1].plot(
        test_signal,
        color="#60A5FA",
        linewidth=0.8,
        label="Actual Telemetry"
    )

    # highlight anomalies of the selected type
    for i, (start, end) in enumerate(sequences):

        if i < len(classes) and classes[i] == anomaly_type:

            axes[1].axvspan(
                start,
                end,
                color="red",
                alpha=0.22,
                label="Labeled Anomaly" if not any(
                    text.get_label() == "Labeled Anomaly"
                    for text in axes[1].patches
                ) else ""
            )

    axes[1].set_title(
        f"Test Telemetry — {anomaly_type.capitalize()} Anomaly"
    )

    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Telemetry Value")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    plt.tight_layout()
    filename = output_dir / f"{anomaly_type}_{channel}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved visualization: {filename}")

def main():

    labels = pd.read_csv(labels_file)
    for anomaly_type in ["point", "contextual"]:
        row, classes, sequences = find_channel(
            labels,
            anomaly_type
        )
        plot_channel(
            row,
            classes,
            sequences,
            anomaly_type
        )

    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()
