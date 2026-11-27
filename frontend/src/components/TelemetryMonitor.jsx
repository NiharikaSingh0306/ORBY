
import { useEffect, useState } from "react";
import {
    ResponsiveContainer,
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ReferenceArea,
} from "recharts";
import "./TelemetryMonitor.css";
const API = "http://127.0.0.1:8000";

const MODEL_CHANNELS = ["D-1", "D-2", "D-3", "D-4", "D-7"];

export default function TelemetryMonitor() {
    const [spacecraft, setSpacecraft] = useState("SMAP");
    const [channel, setChannel] = useState("P-1");
    const [split, setSplit] = useState("test");

    const [channels, setChannels] = useState([]);
    const [telemetry, setTelemetry] = useState([]);
    const [anomalies, setAnomalies] = useState([]);
    const [predictions, setPredictions] = useState([]);

    const [loading, setLoading] = useState(false);
    const [predictionStatus, setPredictionStatus] = useState("idle");
    const [error, setError] = useState("");

    // Fetch available channels
    useEffect(() => {
        async function fetchChannels() {
            try {
                const response = await fetch(
                    `${API}/channels?spacecraft=${spacecraft}`
                );

                if (!response.ok) {
                    throw new Error("Could not load channels");
                }

                const result = await response.json();

                setChannels(result.channels);

                setChannel((current) =>
                    result.channels.some((item) => item.id === current)
                        ? current
                        : result.channels[0]?.id || ""
                );
            } catch (err) {
                setError(err.message);
            }
        }

        fetchChannels();
    }, [spacecraft]);

    // Fetch NASA telemetry and ground-truth anomalies
    useEffect(() => {
        if (!channel) return;

        const controller = new AbortController();

        async function fetchTelemetry() {
            setLoading(true);
            setError("");

            try {
                const response = await fetch(
                    `${API}/telemetry/${spacecraft}/${channel}?split=${split}`,
                    { signal: controller.signal }
                );

                if (!response.ok) {
                    throw new Error("Failed to fetch telemetry");
                }

                const result = await response.json();

                setTelemetry(result.telemetry);
                setAnomalies(result.anomalies);
            } catch (err) {
                if (err.name !== "AbortError") {
                    setError(err.message);
                }
            } finally {
                if (!controller.signal.aborted) {
                    setLoading(false);
                }
            }
        }

        fetchTelemetry();

        return () => controller.abort();
    }, [spacecraft, channel, split]);

    // Fetch ORBY LSTM predictions
    useEffect(() => {
        setPredictions([]);

        if (split !== "test") {
            setPredictionStatus("test-only");
            return;
        }

        if (!MODEL_CHANNELS.includes(channel)) {
            setPredictionStatus("unsupported");
            return;
        }

        const controller = new AbortController();

        async function fetchPredictions() {
            setPredictionStatus("loading");

            try {
                const response = await fetch(
                    `${API}/predict/${spacecraft}/${channel}`,
                    { signal: controller.signal }
                );

                if (!response.ok) {
                    throw new Error("Model predictions unavailable");
                }
                const result = await response.json();

                setPredictions(
                    Array.isArray(result.predictions) ? result.predictions : []
                );

                setPredictionStatus("success");
            } catch (err) {
                if (err.name !== "AbortError") {
                    setPredictions([]);
                    setPredictionStatus("unavailable");
                }
            }
        }

        fetchPredictions();

        return () => controller.abort();
    }, [spacecraft, channel, split]);

    // Match LSTM predictions to telemetry timestamps
    const anomalySteps = new Set(
        predictions
            .filter((item) => item.is_anomaly)
            .map((item) => Number(item.step))
    );

    const chartData = telemetry.map((point) => ({
        ...point,
        modelAnomalyValue: anomalySteps.has(Number(point.step))
            ? point.value
            : null,
    }));

    const modelAnomalyCount = predictions.filter(
        (item) => item.is_anomaly
    ).length;

    return (
        <section className="telemetry-section">
            <div className="telemetry-heading">
                <div>
                    <p className="eyebrow">MISSION CONTROL</p>
                    <h2>Telemetry Monitor</h2>
                    <p className="telemetry-subtitle">
                        Explore spacecraft telemetry and inspect anomaly detection results.
                    </p>
                </div>

                <span className="live-status">
                    <span className="status-dot" />
                    NASA DATA
                </span>
            </div>

            <div className="telemetry-controls">
                <label>
                    Spacecraft
                    <select
                        value={spacecraft}
                        onChange={(e) => setSpacecraft(e.target.value)}
                    >
                        <option value="SMAP">SMAP</option>
                        <option value="MSL">MSL</option>
                    </select>
                </label>

                <label>
                    Channel
                    <select
                        value={channel}
                        onChange={(e) => setChannel(e.target.value)}
                    >
                        {channels.map((item) => (
                            <option key={item.id} value={item.id}>
                                {item.id}
                            </option>
                        ))}
                    </select>
                </label>

                <label>
                    Dataset
                    <select
                        value={split}
                        onChange={(e) => setSplit(e.target.value)}
                    >
                        <option value="test">Testing</option>
                        <option value="train">Training</option>
                    </select>
                </label>
            </div>

            {error && <p className="telemetry-error">{error}</p>}

            <div className="telemetry-chart-card">
                <div className="chart-heading">
                    <div>
                        <h3>{channel || "Select a channel"}</h3>
                        <p>
                            {spacecraft} / {split.toUpperCase()} TELEMETRY
                        </p>
                    </div>

                    <span className="anomaly-count">
                        {anomalies.length} NASA-labelled events
                    </span>
                </div>

                {loading ? (
                    <p className="chart-message">
                        Loading spacecraft telemetry...
                    </p>
                ) : telemetry.length > 0 ? (
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <CartesianGrid
                                    strokeDasharray="3 3"
                                    stroke="#24372c"
                                />

                                <XAxis
                                    dataKey="step"
                                    type="number"
                                    domain={["dataMin", "dataMax"]}
                                    tick={{ fill: "#899b90", fontSize: 11 }}
                                />

                                <YAxis
                                    tick={{ fill: "#899b90", fontSize: 11 }}
                                    width={45}
                                />

                                <Tooltip
                                    contentStyle={{
                                        background: "#101a15",
                                        border: "1px solid #24372c",
                                        borderRadius: "8px",
                                        color: "#e4eee8",
                                    }}
                                />

                                {anomalies.map((item, index) => (
                                    <ReferenceArea
                                        key={index}
                                        x1={item.start}
                                        x2={item.end}
                                        fill="#ef4444"
                                        fillOpacity={0.18}
                                        strokeOpacity={0}
                                    />
                                ))}

                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    name="Telemetry"
                                    stroke="#39d98a"
                                    strokeWidth={1.3}
                                    dot={false}
                                    isAnimationActive={false}
                                />

                                <Line
                                    type="linear"
                                    dataKey="modelAnomalyValue"
                                    name="ORBY detected anomaly"
                                    stroke="transparent"
                                    dot={{
                                        r: 4,
                                        fill: "#facc15",
                                        stroke: "#facc15",
                                        strokeWidth: 1,
                                    }}
                                    activeDot={{
                                        r: 6,
                                        fill: "#facc15",
                                    }}
                                    connectNulls={false}
                                    isAnimationActive={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                ) : (
                    <p className="chart-message">
                        No telemetry data available.
                    </p>
                )}

                <div className="chart-legend">
                    <span>
                        <i className="legend-blue" />
                        Actual telemetry
                    </span>

                    <span>
                        <i className="legend-red" />
                        NASA-labelled anomaly
                    </span>

                    <span>
                        <i className="legend-model" />
                        ORBY LSTM detection
                    </span>
                </div>
            </div>

            <div className="prediction-summary">
                <div>
                    <p className="prediction-label">ORBY MODEL</p>

                    <h3>
                        {predictionStatus === "loading"
                            ? "Analyzing..."
                            : predictionStatus === "success"
                                ? `${modelAnomalyCount} anomalies detected`
                                : predictionStatus === "unsupported"
                                    ? "Channel not supported"
                                    : predictionStatus === "test-only"
                                        ? "Testing dataset only"
                                        : "Predictions unavailable"}
                    </h3>

                    <p className="prediction-description">
                        {predictionStatus === "success"
                            ? "Anomalies identified independently by the trained LSTM model."
                            : predictionStatus === "unsupported"
                                ? "The trained model currently supports D-1, D-2, D-3, D-4 and D-7."
                                : predictionStatus === "test-only"
                                    ? "Switch to the testing dataset to view model predictions."
                                    : predictionStatus === "loading"
                                        ? "Running anomaly detection on the selected channel."
                                        : "Check that the backend is running and the selected channel is supported."}
                    </p>
                </div>
            </div>

            <p className="telemetry-footer">
                Source: NASA JPL SMAP & MSL Anomaly Detection Dataset
            </p>
        </section>
    );
}
