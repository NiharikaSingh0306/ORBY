
import numpy as np
import torch
import random
import time
import torch.nn as nn
import torch.optim as optim
from model import ORBYModel
from pathlib import Path
import copy

DEVICE=torch.device("cpu")

from preprocessing import (
    TRAIN_DIR,
    WINDOW_SIZE,
    load_channel,
    preprocess_channel,
    create_sliding_windows
)

from model import ORBYModel

# Reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Training configuration
DEVICE = torch.device("cpu")

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.2
MAX_TRAIN_BATCHES_PER_CHANNEL = 16
MAX_VALIDATION_BATCHES_PER_CHANNEL = None

def prepare_channel_data(channel):
    train, test = load_channel(channel)

    split_index = int(
        len(train) * (1 - VALIDATION_RATIO)
    )

    if split_index <= WINDOW_SIZE:
        raise ValueError(
            f"Not enough training samples for {channel}"
        )

    # Fit preprocessing only on the training portion
    fit_data = train[:split_index]

    train_processed, test_processed, params = preprocess_channel(
        train,
        test,
        fit_data=fit_data
    )

    train_data = train_processed[:split_index]

    # Keep historical context for validation windows
    val_data = train_processed[split_index - WINDOW_SIZE:]

    return train_data, val_data, params


def prepare_all_channels():
    channel_data = {}
    channel_dimensions = {}

    channel_files = sorted(TRAIN_DIR.glob("*.npy"))

    print(f"\nPreparing {len(channel_files)} channels...")

    for index, file_path in enumerate(channel_files, start=1):
        channel = file_path.stem

        try:
            train_data, val_data, params = prepare_channel_data(channel)
        except ValueError as error:
            print(f"Skipping {channel}: {error}")
            continue

        channel_data[channel] = (
            train_data,
            val_data,
            params
        )

        channel_dimensions[channel] = params["input_features"]

        print(
            f"[{index}/{len(channel_files)}] "
            f"{channel}: {params['input_features']} features"
        )

    return channel_data, channel_dimensions


def create_batches(X, y, batch_size=BATCH_SIZE, shuffle=True):
    """
    Generate small batches without converting the
    entire dataset into PyTorch tensors.
    """

    indices = np.arange(len(X))

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        X_batch = torch.from_numpy(
            X[batch_indices]
        ).float()

        y_batch = torch.from_numpy(
            y[batch_indices]
        ).float()

        yield X_batch, y_batch



def create_time_series_batches(
    data,
    batch_size=BATCH_SIZE,
    window_size=WINDOW_SIZE,
    shuffle=True,
    max_batches=None
):
    total_windows = len(data) - window_size

    if total_windows <= 0:
        raise ValueError("Not enough data to create windows.")

    indices = np.arange(total_windows)

    if shuffle:
        np.random.shuffle(indices)

    for batch_number, start in enumerate(
        range(0, total_windows, batch_size)
    ):
        if max_batches is not None and batch_number >= max_batches:
            break

        batch_indices = indices[start:start + batch_size]

        X_batch = np.stack([
            data[i:i + window_size]
            for i in batch_indices
        ])

        y_batch = data[batch_indices + window_size, 0]

        yield (
            torch.from_numpy(X_batch).float(),
            torch.from_numpy(y_batch).float()
        )


def train_one_epoch(model, optimizer, criterion, channel_data):
    model.train()

    iterators = {}

    for channel, (train_data, _, _) in channel_data.items():
        iterators[channel] = iter(
            create_time_series_batches(
                train_data,
                shuffle=True,
                max_batches=MAX_TRAIN_BATCHES_PER_CHANNEL
            )
        )

    active_channels = list(iterators.keys())

    total_loss = 0.0
    total_samples = 0

    # Round-robin training across channels
    while active_channels:
        random.shuffle(active_channels)
        remaining_channels = []

        for channel in active_channels:
            try:
                X_batch, y_batch = next(iterators[channel])
            except StopIteration:
                continue

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(X_batch, channel)
            loss = criterion(predictions, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = len(y_batch)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            remaining_channels.append(channel)

        active_channels = remaining_channels

    if total_samples == 0:
        raise ValueError("No training batches were generated.")

    return total_loss / total_samples


def validate(model, criterion, channel_data):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for channel, (_, val_data, _) in channel_data.items():

            for X_batch, y_batch in create_time_series_batches(
                val_data,
                shuffle=False,
                max_batches=MAX_VALIDATION_BATCHES_PER_CHANNEL
            ):
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                predictions = model(X_batch, channel)
                loss = criterion(predictions, y_batch)

                batch_size = len(y_batch)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

    if total_samples == 0:
        raise ValueError("No validation batches were generated.")

    return total_loss / total_samples


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model(model, channel_data):
    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    best_state = None
    history = []

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, optimizer, criterion, channel_data
        )

        val_loss = validate(
            model, criterion, channel_data
        )

        elapsed = time.time() - start_time

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time_seconds": elapsed
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {val_loss:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                best_state,
                MODEL_DIR / "orby_best_model.pth"
            )

            print("  Best model checkpoint saved.")

        else:
            patience_counter += 1
            print(f"  No improvement: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

print("ORBY TRAINING CONFIGURATION")
print("Device:", DEVICE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Window size:", WINDOW_SIZE)
if __name__ == "__main__":
    channel_data, channel_dimensions = prepare_all_channels()

    print("\nInitializing ORBY model...")

    model = ORBYModel(channel_dimensions).to(DEVICE)

    print("Model parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    print("\nStarting ORBY pilot training with full validation...")

    start_time = time.time()

    model, history = train_model(model, channel_data)

    total_time = time.time() - start_time

    print("\nPILOT TRAINING SUMMARY")

    for record in history:
        print(
            f"Epoch {record['epoch']}: "
            f"Train Loss = {record['train_loss']:.6f}, "
            f"Validation Loss = {record['val_loss']:.6f}"
        )

    print(f"\nTotal training time: {total_time:.2f} seconds")