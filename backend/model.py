import torch
import torch.nn as nn


class ChannelProjection(nn.Module):
    """
    Converts one channel's input features into a common
    latent representation.
    """

    def __init__(self, input_features):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_features, PROJECTION_SIZE),
            nn.ReLU()
        )

    def forward(self, x):
        return self.projection(x)


class ORBYModel(nn.Module):
    """
    Hybrid anomaly detection model:
    - Channel-specific input projections
    - Learned channel embeddings
    - Shared LSTM
    - Shared next-step prediction head
    """

    def __init__(self, channel_dimensions):
        super().__init__()

        self.channel_names = sorted(channel_dimensions.keys())

        # Assign a unique integer ID to every channel
        self.channel_to_id = {
            channel: index
            for index, channel in enumerate(self.channel_names)
        }

        # One independent projection for each channel
        self.projections = nn.ModuleDict({
            channel: ChannelProjection(input_features)
            for channel, input_features in channel_dimensions.items()
        })

        # Learnable identity representation for each channel
        self.channel_embeddings = nn.Embedding(
            len(self.channel_names),
            CHANNEL_EMBEDDING_SIZE
        )

        # Shared LSTM for all channels
        self.lstm = nn.LSTM(
            input_size=PROJECTION_SIZE + CHANNEL_EMBEDDING_SIZE,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )

        # Predict the next telemetry value
        self.prediction_head = nn.Linear(
            LSTM_HIDDEN_SIZE,
            1
        )

    def forward(self, x, channel):
        """
        x shape: (batch, window_size, input_features)
        channel: channel name, e.g. 'P-1'
        """

        # Project channel-specific input features
        projected = self.projections[channel](x)

        # Get the channel identity embedding
        channel_id = self.channel_to_id[channel]

        channel_tensor = torch.tensor(
            channel_id,
            dtype=torch.long,
            device=x.device
        )

        embedding = self.channel_embeddings(channel_tensor)

        # Repeat embedding across batch and time dimensions
        embedding = embedding.view(1, 1, -1)
        embedding = embedding.expand(
            x.size(0),
            x.size(1),
            -1
        )

        # Combine projected features and channel identity
        combined = torch.cat(
            [projected, embedding],
            dim=-1
        )

        # Pass through shared LSTM
        lstm_output, _ = self.lstm(combined)

        # Use the final time step for prediction
        final_output = lstm_output[:, -1, :]

        prediction = self.prediction_head(final_output)

        return prediction.squeeze(-1)

#model configurations
PROJECTION_SIZE = 32
CHANNEL_EMBEDDING_SIZE = 16
LSTM_HIDDEN_SIZE = 80
LSTM_LAYERS = 2
DROPOUT = 0.3


if __name__ == "__main__":
    from preprocessing import inspect_all_channel_dimensions

    # Get actual feature dimensions from the dataset
    dimensions_info = inspect_all_channel_dimensions()

    channel_dimensions = {
        channel: info["processed_features"]
        for channel, info in dimensions_info.items()
    }

    # Initialize ORBY model
    model = ORBYModel(channel_dimensions)

    print("\nORBY MODEL INITIALIZED")
    print("Total channels:", len(channel_dimensions))
    print("Total parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    # Test channels with different input dimensions
    test_channels = ["P-1", "A-1", "D-12"]

    model.eval()

    with torch.no_grad():
        for channel in test_channels:
            input_features = channel_dimensions[channel]

            # Batch of 4 sequences, each with 250 time steps
            sample_input = torch.randn(
                4,
                250,
                input_features
            )

            prediction = model(sample_input, channel)

            print(f"\nChannel: {channel}")
            print("Input shape:", sample_input.shape)
            print("Prediction shape:", prediction.shape)
            print("Predictions:", prediction)

    print("\nMODEL TEST: PASSED")
