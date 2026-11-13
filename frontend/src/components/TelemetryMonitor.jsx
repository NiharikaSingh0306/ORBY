import { useEffect, useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceArea, } from "recharts";
import "./TelemetryMonitor.css"
const API = "http://127.0.0.1:8000";

export default function TelemetryMonitor() {
    const [spacecraft, setSpacecraft] = useState("SMAP");
    const [channel, setChannel] = useState("P-1");
    const [split, setSplit] = useState("test");

    const [channels, setChannels] = useState([]);
    const [telemetry, setTelemetry] = useState([]);
    const [anomalies, setAnomalies] = useState([]);

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    //fetch availabel channels
    useEffect(() => {
        async function fetchChannels() {
            try {
                const response = await fetch(
                    `${API}/channels?spacecraft=${spacecraft}`
                )
                if (!response.ok) throw new Error("Could not load channels");
                const result = await response.json();
                setChannels(result.channels);
                // keeping the selected channel if available
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

    //fetch telemetry for the selected channel
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

    return (
        <section className="telemetry-section">
            <div className="telemetry-heading">
                <div>
                    <p className="eyebrow">MISSION CONTROL</p>
                    <h2>Telemetry Monitor</h2>
                    <p className="telemetry-subtitle">
                        Monitor real spacecraft signals and anomaly events.
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
                        {anomalies.length} labeled events
                    </span>
                </div>
                {loading ? (
                    <p className="chart-message">Loading spacecraft telemetry...</p>
                ) : telemetry.length > 0 ? (
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={telemetry}>
                                <CartesianGrid
                                    strokeDasharray="3 3"
                                    stroke="#263244"
                                />
                                <XAxis
                                    dataKey="step"
                                    type="number"
                                    domain={["dataMin", "dataMax"]}
                                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                                />
                                <YAxis
                                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                                    width={45}
                                />
                                <Tooltip
                                    contentStyle={{
                                        background: "#111827",
                                        border: "1px solid #334155",
                                        borderRadius: "8px",
                                        color: "#fff",
                                    }}
                                />
                                {anomalies.map((item, index) => (
                                    <ReferenceArea
                                        key={index}
                                        x1={item.start}
                                        x2={item.end}
                                        fill="#ef4444"
                                        fillOpacity={0.22}
                                        strokeOpacity={0}
                                    />
                                ))}
                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#60a5fa"
                                    strokeWidth={1.3}
                                    dot={false}
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
                        Labeled anomaly
                    </span>
                </div>
            </div>
            <p className="telemetry-footer">
                Source: NASA JPL SMAP & MSL Anomaly Detection Dataset
            </p>
        </section>
    );
}



