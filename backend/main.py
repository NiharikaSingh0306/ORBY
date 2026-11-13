from pathlib import Path
import ast
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
import re

#configurations
base_dir=Path(__file__).resolve().parent
data_dir=base_dir/"data"
train_dir=data_dir/"train"
test_dir=data_dir/"test"
labels_file=data_dir/"labeled_anomalies.csv"

app=FastAPI(
    title="ORBY_API",
    description="Spacecraft telemetry and anomaly detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# load Labels
def load_labels():
    if not labels_file.exists():
        raise HTTPException(
            status_code=500,
            detail="NASA anomaly labels file not found."
        )
    return pd.read_csv(labels_file)

@app.get("/")
def home():
    return{
        "message":"This is ORBY API",
        "status":"online"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ORBY Backend"
    }

#mission info
@app.get("/missions")
def get_missions():
    return {
        "missions": [
            {
                "id": "SMAP",
                "name": "Soil Moisture Active Passive",
                "channels": 55
            },
            {
                "id": "MSL",
                "name": "Mars Science Laboratory",
                "channels": 27
            }
        ]
    }

#availabe channels
@app.get("/channels")
def get_channels(
    spacecraft: str = Query(..., pattern="^(SMAP|MSL)$")
):
    labels = load_labels()
    mission_labels = labels[
        labels["spacecraft"].str.upper() == spacecraft
    ]
    channels = []
    for _, row in mission_labels.iterrows():
        channel = row["chan_id"]
        train_exists = (train_dir/ f"{channel}.npy").exists()
        test_exists = (test_dir/ f"{channel}.npy").exists()

        if train_exists and test_exists:
            channels.append({
                "id": channel,
                "spacecraft": spacecraft,
                "anomaly_count": len(
                    ast.literal_eval(row["anomaly_sequences"])
                )
            })
    return {
        "spacecraft": spacecraft,
        "total_channels": len(channels),
        "channels": channels
    }

#telemetry data
@app.get("/telemetry/{spacecraft}/{channel}")
def get_telemetry(
    spacecraft: str,
    channel: str,
    split: str = Query(default="test", pattern="^(train|test)$")
):
    spacecraft = spacecraft.upper()
    if spacecraft not in ["SMAP", "MSL"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid spacecraft. Choose SMAP or MSL."
        )

    data_dir = train_dir if split == "train" else test_dir
    file_path = data_dir / f"{channel}.npy"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Channel {channel} not found."
        )

    labels = load_labels()
    channel_label = labels[
        (labels["chan_id"] == channel) &
        (labels["spacecraft"].str.upper() == spacecraft)
    ]

    if channel_label.empty:
        raise HTTPException(
            status_code=404,
            detail="Channel does not belong to the selected spacecraft."
        )
    try:
        data = np.load(file_path)
        if data.ndim != 2 or data.shape[1] < 1:
            raise ValueError("Unexpected telemetry array shape.")
        telemetry = data[:, 0]

        # labels refer to test-set indices.
        if split == "test":
            row = channel_label.iloc[0]
            sequences = ast.literal_eval(
                row["anomaly_sequences"]
            )
            classes = re.findall(r"point|contextual",str(row["class"]).lower())
            anomalies = [
                {
                    "start": int(start),
                    "end": int(end),
                    "type": str(anomaly_type)
                }
                for (start, end), anomaly_type
                in zip(sequences, classes)
            ]
        else:
            anomalies = []
        return {
            "spacecraft": spacecraft,
            "channel": channel,
            "split": split,
            "total_steps": len(telemetry),
            "total_features": int(data.shape[1]),
            "telemetry": [
                {
                    "step": int(i),
                    "value": float(value)
                }
                for i, value in enumerate(telemetry)
            ],
            "anomalies": anomalies
        }
    except (ValueError, SyntaxError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"could not process channel data: {error}"
        )



























