
import numpy as np
import torch

from model import ORBYModel
from preprocessing import (
    load_channel,
    preprocess_channel,
    TRAIN_DIR,
    WINDOW_SIZE
)

# Configuration
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
VALIDATION_RATIO = 0.2

CHECKPOINT_PATH = "models/orby_best_model.pth"
OUTPUT_PATH = "models/orby_validation_errors.npz"


def load_trained_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
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

    model = ORBYModel(channel_dimensions).to(DEVICE)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    return model, channel_dimensions


def calculate_errors(model, data, channel):
    total_windows = len(data) - WINDOW_SIZE

    if total_windows <= 0:
        raise ValueError(
            f"Not enough validation samples for {channel}"
        )

    errors = []

    with torch.no_grad():

        for start in range(0, total_windows, BATCH_SIZE):

            indices = np.arange(
                start,
                min(start + BATCH_SIZE, total_windows)
            )

            X_batch = np.stack([
                data[i:i + WINDOW_SIZE]
                for i in indices
            ])

            y_batch = data[
                indices + WINDOW_SIZE, 0
            ]

            X_tensor = torch.from_numpy(
                X_batch
            ).float().to(DEVICE)

            predictions = model(X_tensor, channel)

            batch_errors = np.abs(
                predictions.cpu().numpy() - y_batch
            )

            errors.extend(batch_errors.tolist())

    return np.asarray(errors, dtype=np.float32)


def main():
    model, channel_dimensions = load_trained_model()

    print("ORBY VALIDATION ERROR CALIBRATION")
    print("Channels:", len(channel_dimensions))
    print("Window size:", WINDOW_SIZE)

    validation_errors = {}

    for index, channel in enumerate(
        sorted(channel_dimensions),
        start=1
    ):

        train, _ = load_channel(channel)

        split_index = int(
            len(train) * (1 - VALIDATION_RATIO)
        )

        fit_data = train[:split_index]

        train_processed, _, _ = preprocess_channel(
            train,
            np.empty((0, train.shape[1]), dtype=np.float32),
            fit_data=fit_data
        )

        val_data = train_processed[
            split_index - WINDOW_SIZE:
        ]

        errors = calculate_errors(
            model,
            val_data,
            channel
        )

        validation_errors[channel] = errors

        median = np.median(errors)
        mad = np.median(np.abs(errors - median))

        print(
            f"[{index}/{len(channel_dimensions)}] "
            f"{channel}: "
            f"median={median:.5f}, "
            f"MAD={mad:.5f}, "
            f"p99={np.percentile(errors, 99):.5f}"
        )

    np.savez_compressed(
        OUTPUT_PATH,
        **validation_errors
    )

    print("\nVALIDATION CALIBRATION COMPLETE")
    print("Saved:", OUTPUT_PATH)
    print("Channels processed:", len(validation_errors))


if __name__ == "__main__":
    main()
