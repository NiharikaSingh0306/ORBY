from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI(
    title="ORBY_API",
    description="Spacecraft telemetry and anomaly detection API",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["https://localhost:5174"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

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