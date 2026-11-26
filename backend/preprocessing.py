
import numpy as np
from pathlib import Path

# dataset configuration
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
WINDOW_SIZE = 250

def load_channel(channel):
    """
    Load training and testing telemetry for one channel.
    Validate shapes and numerical values.
    """
    train_path = TRAIN_DIR / f"{channel}.npy"
    test_path = TEST_DIR / f"{channel}.npy"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing file not found: {test_path}")

    train = np.load(train_path).astype(np.float32)
    test = np.load(test_path).astype(np.float32)

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("Expected 2D telemetry arrays.")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            f"Feature mismatch for {channel}: "
            f"train={train.shape[1]}, test={test.shape[1]}"
        )

    if not np.isfinite(train).all():
        raise ValueError(f"Invalid values in training data: {channel}")
    if not np.isfinite(test).all():
        raise ValueError(f"Invalid values in testing data: {channel}")

    return train, test



def preprocess_channel(train, test, fit_data=None):
    """
    Fit preprocessing parameters on training data only.
    Normalize all retained features and remove constant
    command features.
    """

    if fit_data is None:
        fit_data = train

    # Identify variable command features using training data only
    fit_commands = fit_data[:, 1:]
    command_std = np.std(fit_commands, axis=0)
    variable_mask = command_std > 1e-8

    # Keep telemetry and variable command features
    feature_mask = np.concatenate([
        np.array([True]),
        variable_mask
    ])

    train_filtered = train[:, feature_mask]
    test_filtered = test[:, feature_mask]
    fit_filtered = fit_data[:, feature_mask]

    # Calculate normalization parameters using training data only
    feature_mean = np.mean(fit_filtered, axis=0)
    feature_std = np.std(fit_filtered, axis=0)

    # Avoid division by zero
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    train_processed = (
        train_filtered - feature_mean
    ) / feature_std

    test_processed = (
        test_filtered - feature_mean
    ) / feature_std

    preprocessing_params = {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "variable_command_mask": variable_mask.tolist(),
        "input_features": int(train_processed.shape[1])
    }

    return (
        train_processed.astype(np.float32),
        test_processed.astype(np.float32),
        preprocessing_params
    )

def create_sliding_windows(data, window_size=WINDOW_SIZE):
    """
    Create sequences of consecutive time steps.

    X: Previous window_size rows of all features.
    y: Next-step telemetry value.
    """

    if len(data) <= window_size:
        raise ValueError(
            f"Not enough samples: {len(data)} for window {window_size}"
        )

    X = np.lib.stride_tricks.sliding_window_view(
        data, window_size, axis=0
    )

    # Shape: (windows, features, window_size)
    X = np.moveaxis(X, -1, 1)

    # Remove the final window because it has no next-step target
    X = X[:-1]

    # Target is the telemetry value immediately after each window
    y = data[window_size:, 0]

    return (
        np.ascontiguousarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )

def split_train_validation(X, y, validation_ratio=0.2):
    """
    Chronologically split sliding windows into training
    and validation sets.
    """

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    split_index = int(len(X) * (1 - validation_ratio))

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_val = X[split_index:]
    y_val = y[split_index:]

    return X_train, y_train, X_val, y_val



def inspect_all_channel_dimensions(validation_ratio=0.2):
    channel_dimensions = {}

    for file_path in sorted(TRAIN_DIR.glob("*.npy")):
        channel = file_path.stem
        data = np.load(file_path).astype(np.float32)

        # Use only the fitting portion to determine feature dimensions
        split_index = int(len(data) * (1 - validation_ratio))
        fit_data = data[:split_index]

        command_features = fit_data[:, 1:]
        command_std = np.std(command_features, axis=0)

        variable_count = int(np.sum(command_std > 0))

        channel_dimensions[channel] = {
            "original_features": int(data.shape[1]),
            "variable_commands": variable_count,
            "processed_features": variable_count + 1
        }

    return channel_dimensions

if __name__ == "__main__":
    train, test = load_channel("P-1")

    train_processed, test_processed, params = preprocess_channel(
        train, test
    )

    print("\nPREPROCESSING RESULTS")
    print("Original training shape:", train.shape)
    print("Processed training shape:", train_processed.shape)
    print("Original testing shape:", test.shape)
    print("Processed testing shape:", test_processed.shape)
    
    X, y = create_sliding_windows(train_processed)
    print("\nSLIDING WINDOW RESULTS")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Expected windows:", len(train_processed) - WINDOW_SIZE)  

    X_train, y_train, X_val, y_val = split_train_validation(X, y)

    print("\nTRAIN / VALIDATION SPLIT")
    print("Training windows:", X_train.shape)
    print("Training targets:", y_train.shape)
    print("Validation windows:", X_val.shape)
    print("Validation targets:", y_val.shape)

    print("\nNormalization parameters:")
    print("Mean:", params["telemetry_mean"])
    print("Std:", params["telemetry_std"])
    print("Input features:", params["input_features"])

    print("\nTraining telemetry mean after normalization:",
          np.mean(train_processed[:, 0]))

    print("Training telemetry std after normalization:",
          np.std(train_processed[:, 0]))

    print("\nValidation: PASSED")


    dimensions = inspect_all_channel_dimensions()

    print("\nALL CHANNEL INPUT DIMENSIONS")

    for channel, info in dimensions.items():
        print(
            f"{channel}: "
            f"original={info['original_features']}, "
            f"variable_commands={info['variable_commands']}, "
            f"processed={info['processed_features']}"
        )

    print("\nTotal channels:", len(dimensions))
    print(
        "Maximum processed features:",
        max(info["processed_features"] for info in dimensions.values())
    )

