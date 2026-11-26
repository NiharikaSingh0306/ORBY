
import ast
import csv
import json
from pathlib import Path
import numpy as np
import torch

from model import ORBYModel
from preprocessing import (
    WINDOW_SIZE,
    load_channel,
    preprocess_channel,
)

MODEL_PATH = Path("models/orby_best_model.pth")
THRESHOLDS_PATH = Path("models/orby_thresholds.json")
LABELS_PATH = Path("data/labeled_anomalies.csv")
OUTPUT_PATH = Path("models/orby_evaluation_results.csv")
DEVICE = torch.device("cpu")
BATCH_SIZE = 64

def load_labels():
    """Load NASA anomaly intervals for every channel."""
    labels = {}

    with open(LABELS_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            channel = row["chan_id"]
            intervals = ast.literal_eval(row["anomaly_sequences"])
            labels[channel] = intervals

    return labels

def reconstruct_model():
    """Reconstruct ORBY from the saved state dictionary."""
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    channel_dimensions = {}
    for key, tensor in checkpoint.items():
        if (
            key.startswith("projections.")
            and key.endswith(".projection.0.weight")
        ):
            channel = key.split(".")[1]
            channel_dimensions[channel] = tensor.shape[1]

    model = ORBYModel(channel_dimensions)
    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE)
    model.eval()

    return model, channel_dimensions

def predict_errors(model, channel, test_data):
    """Predict next-step telemetry values and calculate errors."""
    total_windows = len(test_data) - WINDOW_SIZE

    if total_windows <= 0:
        raise ValueError(
            f"{channel}: Not enough test observations."
        )
    errors = []

    with torch.no_grad():
        for start in range(0, total_windows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_windows)
            windows = np.stack([
                test_data[i:i + WINDOW_SIZE]
                for i in range(start, end)
            ])

            targets = test_data[
                start + WINDOW_SIZE:end + WINDOW_SIZE, 0
            ]
            x = torch.tensor(
                windows,
                dtype=torch.float32,
                device=DEVICE,
            )
            predictions = model(x, channel).cpu().numpy()
            batch_errors = np.abs(predictions - targets)
            errors.extend(batch_errors.tolist())

    return np.asarray(errors, dtype=np.float32)

def calculate_metrics(y_true, y_pred):
    """Calculate point-wise binary classification metrics."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }

def main():
    print("LOADING ORBY MODEL")
    model, channel_dimensions = reconstruct_model()

    print("Model channels:", len(channel_dimensions))
    with open(THRESHOLDS_PATH, "r") as file:
        thresholds = json.load(file)

    labels_by_channel = load_labels()
    results = []
    test_errors_by_channel={}

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    print("\nSTARTING TEST EVALUATION")
    print("-" * 90)

    for channel in sorted(channel_dimensions):
        train_data, test_data = load_channel(channel)

        split_index = int(len(train_data) * 0.8)
        fit_data = train_data[:split_index]

        train_processed, test_processed, params = (
            preprocess_channel(
                train_data,
                test_data,
                fit_data=fit_data,
            )
        )
        if params["input_features"] != channel_dimensions[channel]:
            raise ValueError(
                f"{channel}: Feature dimension mismatch."
            )

        errors = predict_errors(
            model,
            channel,
            test_processed,
        )

        test_errors_by_channel[channel]=errors
        threshold = thresholds[channel]["threshold"]
        predictions = (errors > threshold).astype(np.int32)

        # Build ground-truth labels for the complete test sequence.
        ground_truth = np.zeros(
            len(test_processed),
            dtype=np.int32,
        )
        if channel not in labels_by_channel:
            print(f"{channel:6} | SKIPPED — No NASA ground-truth labels")
            continue

        for start, end in labels_by_channel[channel]:
            start = int(start)
            end = int(end)
            ground_truth[start:end + 1] = 1

        # The first WINDOW_SIZE points are context only.
        y_true = ground_truth[WINDOW_SIZE:]
        if len(y_true) != len(predictions):
            raise ValueError(
                f"{channel}: Prediction/label alignment mismatch."
            )

        metrics = calculate_metrics(y_true, predictions)

        total_tp += metrics["TP"]
        total_fp += metrics["FP"]
        total_tn += metrics["TN"]
        total_fn += metrics["FN"]

        results.append({
            "channel": channel,
            "threshold": threshold,
            "samples": len(y_true),
            **metrics,
        })

        print(
            f"{channel:6} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"FPR={metrics['fpr']:.4f}"
        )
    test_errors_path = OUTPUT_PATH.parent / "orby_test_errors.npz"

    np.savez_compressed(
        test_errors_path,
        **test_errors_by_channel,
    )

    print("Test errors saved:", test_errors_path)
    overall = calculate_metrics(
        np.array(
            [1] * (total_tp + total_fn)
            + [0] * (total_tn + total_fp),
            dtype=np.int32,
        ),
        np.array(
            [1] * total_tp
            + [0] * total_fn
            + [1] * total_fp
            + [0] * total_tn,
            dtype=np.int32,
        ),
    )

    results.append({
        "channel": "OVERALL",
        "threshold": "",
        "samples": total_tp + total_fp + total_tn + total_fn,
        **overall,
    })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "channel",
        "threshold",
        "samples",
        "TP",
        "FP",
        "TN",
        "FN",
        "precision",
        "recall",
        "f1",
        "fpr",
    ]
    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 90)
    print("OVERALL TEST RESULTS")
    print("=" * 90)

    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1-score:  {overall['f1']:.4f}")
    print(f"FPR:       {overall['fpr']:.4f}")

    print("\nChannels evaluated:", len(channel_dimensions))
    print("D-12 excluded: True")
    print("Results saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
