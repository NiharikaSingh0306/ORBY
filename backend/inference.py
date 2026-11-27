
import json
from pathlib import Path
import numpy as np
import torch
from model import ORBYModel
from preprocessing import (
    WINDOW_SIZE,
    load_channel,
    preprocess_channel
)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "orby_best_model.pth"
THRESHOLDS_PATH = BASE_DIR / "models" / "orby_thresholds.json"

DEVICE = torch.device("cpu")
BATCH_SIZE = 64
VALIDATION_RATIO = 0.2


# Load trained model and thresholds
def load_artifacts():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
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

    with open(THRESHOLDS_PATH, "r") as file:
        thresholds = json.load(file)

    if set(channel_dimensions) != set(thresholds):
        raise ValueError(
            "Model channels and threshold channels do not match."
        )

    return model, channel_dimensions, thresholds


# Run anomaly prediction
def predict_channel(channel):

    model, channel_dimensions, thresholds = load_artifacts()

    if channel not in channel_dimensions:
        raise ValueError(
            f"Channel {channel} is not supported by the trained model."
        )

    # Load original NASA telemetry
    train_data, test_data = load_channel(channel)

    # Reproduce the original training preprocessing
    split_index = int(
        len(train_data) * (1 - VALIDATION_RATIO)
    )

    fit_data = train_data[:split_index]

    _, test_processed, params = preprocess_channel(
        train_data,
        test_data,
        fit_data=fit_data
    )

    if params["input_features"] != channel_dimensions[channel]:
        raise ValueError(
            f"Feature dimension mismatch for {channel}."
        )

    total_windows = len(test_processed) - WINDOW_SIZE

    if total_windows <= 0:
        raise ValueError("Insufficient telemetry observations.")

    errors = []

    # Batch inference
    with torch.no_grad():

        for start in range(0, total_windows, BATCH_SIZE):

            end = min(start + BATCH_SIZE, total_windows)

            windows = np.stack([
                test_processed[i:i + WINDOW_SIZE]
                for i in range(start, end)
            ])

            targets = test_processed[
                start + WINDOW_SIZE:end + WINDOW_SIZE
            ]

            x = torch.from_numpy(windows).float().to(DEVICE)

            predictions = model(x, channel)

            # Same scoring logic used during evaluation
            batch_errors = np.abs(
                predictions.cpu().numpy()[:, 0]
                - targets[:, 0]
            )

            errors.extend(batch_errors.tolist())

    errors = np.asarray(errors, dtype=np.float32)

    threshold = float(thresholds[channel]["threshold"])

    anomaly_flags = errors > threshold

    # Build prediction results
    results = []

    for index, (error, is_anomaly) in enumerate(
        zip(errors, anomaly_flags)
    ):

        results.append({
            "step": int(index + WINDOW_SIZE),
            "anomaly_score": float(error),
            "is_anomaly": bool(is_anomaly)
        })

    return {
        "channel": channel,
        "window_size": WINDOW_SIZE,
        "threshold": threshold,
        "total_predictions": len(results),
        "anomaly_count": int(np.sum(anomaly_flags)),
        "predictions": results
    }
