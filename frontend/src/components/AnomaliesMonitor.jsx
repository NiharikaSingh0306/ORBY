
import { useEffect, useMemo, useState } from "react";
import { AlertTriangle } from "lucide-react";
import "./AnomaliesMonitor.css";

const API = "http://127.0.0.1:8000";
const MODEL_CHANNELS = ["D-1", "D-2", "D-3", "D-4", "D-7"];

export default function AnomaliesMonitor() {
    const [spacecraft, setSpacecraft] = useState("SMAP");
    const [channels, setChannels] = useState([]);
    const [channel, setChannel] = useState("D-1");

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    // Fetch model-supported channels available for this spacecraft
    useEffect(() => {
        const controller = new AbortController();

        async function fetchChannels() {
            setChannels([]);
            setChannel("");
            setResult(null);
            setError("");

            try {
                const response = await fetch(
                    `${API}/channels?spacecraft=${spacecraft}`,
                    { signal: controller.signal }
                );

                if (!response.ok) {
                    throw new Error("Could not load spacecraft channels.");
                }

                const data = await response.json();

                const supported = data.channels
                    .map((item) => item.id)
                    .filter((id) => MODEL_CHANNELS.includes(id));

                setChannels(supported);
                setChannel(supported[0] || "");
            } catch (err) {
                if (err.name !== "AbortError") {
                    setError(err.message);
                }
            }
        }

        fetchChannels();

        return () => controller.abort();
    }, [spacecraft]);

    // Fetch model predictions
    useEffect(() => {
        if (!channel) {
            setResult(null);
            return;
        }

        const controller = new AbortController();

        async function fetchPredictions() {
            setLoading(true);
            setError("");
            setResult(null);

            try {
                const response = await fetch(
                    `${API}/predict/${spacecraft}/${channel}`,
                    { signal: controller.signal }
                );

                if (!response.ok) {
                    throw new Error("Could not retrieve model predictions.");
                }

                const data = await response.json();

                if (!Array.isArray(data.predictions)) {
                    throw new Error("Unexpected prediction response.");
                }

                setResult(data);
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

        fetchPredictions();

        return () => controller.abort();
    }, [spacecraft, channel]);

    // Show the 50 highest-scoring detected anomalies
    const detectedAnomalies = useMemo(() => {
        if (!result?.predictions) return [];

        return result.predictions
            .filter((item) => item.is_anomaly)
            .sort(
                (a, b) =>
                    b.anomaly_score - a.anomaly_score
            )
            .slice(0, 50);
    }, [result]);

    return (
        <section className="anomalies-section">
            <div className="anomalies-heading">
                <div>
                    <p className="eyebrow">MODEL ANALYSIS</p>
                    <h2>Anomaly Detection</h2>
                    <p className="anomalies-subtitle">
                        Investigate unusual spacecraft telemetry identified by ORBY.
                    </p>
                </div>

                <div className="anomaly-page-status">
                    <span className="status-dot" />
                    LSTM MODEL
                </div>
            </div>

            <div className="anomalies-controls">
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
                        disabled={channels.length === 0}
                    >
                        {channels.map((item) => (
                            <option key={item} value={item}>
                                {item}
                            </option>
                        ))}
                    </select>
                </label>
            </div>

            {error && (
                <p className="telemetry-error">{error}</p>
            )}

            {loading ? (
                <p className="chart-message">
                    Analyzing telemetry with ORBY...
                </p>
            ) : channels.length === 0 && !error ? (
                <p className="chart-message">
                    No trained model channels are available for this spacecraft.
                </p>
            ) : result && (
                <>
                    <div className="anomaly-summary">
                        <div>
                            <p className="prediction-label">
                                DETECTED ANOMALIES
                            </p>
                            <h3>{result.anomaly_count.toLocaleString()}</h3>
                            <p className="prediction-description">
                                Identified by the trained LSTM model
                            </p>
                        </div>

                        <div className="anomaly-summary-details">
                            <div>
                                <span>Total predictions</span>
                                <strong>
                                    {result.total_predictions.toLocaleString()}
                                </strong>
                            </div>
                            <div>
                                <span>Threshold</span>
                                <strong>
                                    {result.threshold.toFixed(4)}
                                </strong>
                            </div>
                            <div>
                                <span>Window size</span>
                                <strong>{result.window_size}</strong>
                            </div>
                        </div>
                    </div>

                    <div className="anomaly-table-section">
                        <div className="anomaly-table-heading">
                            <div>
                                <h3>
                                    <AlertTriangle size={17} />
                                    Detected events
                                </h3>
                                <p>
                                    Showing the 50 highest-scoring detections
                                </p>
                            </div>

                            <span className="anomaly-count">
                                {result.anomaly_count} detected
                            </span>
                        </div>

                        <div className="anomaly-table-wrapper">
                            <table className="anomaly-table">
                                <thead>
                                    <tr>
                                        <th>Spacecraft</th>
                                        <th>Channel</th>
                                        <th>Telemetry step</th>
                                        <th>Anomaly score</th>
                                        <th>Threshold</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>

                                <tbody>
                                    {detectedAnomalies.map((item, index) => (
                                        <tr key={`${item.step}-${index}`}>
                                            <td>{spacecraft}</td>
                                            <td>{channel}</td>
                                            <td>{item.step}</td>
                                            <td className="anomaly-score">
                                                {Number(item.anomaly_score).toFixed(4)}
                                            </td>
                                            <td>
                                                {result.threshold.toFixed(4)}
                                            </td>
                                            <td>
                                                <span className="detected-badge">
                                                    Detected
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            )}

            <p className="telemetry-footer">
                Source: NASA JPL SMAP & MSL Anomaly Detection Dataset
            </p>
        </section>
    );
}
