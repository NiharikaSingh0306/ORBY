
import { useEffect, useState } from "react";
import { Activity, Target, TrendingUp, AlertTriangle } from "lucide-react";
import "./PerformanceMonitor.css";

const API = "http://127.0.0.1:8000";

function PerformanceMonitor() {
    const [evaluation, setEvaluation] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        const fetchEvaluation = async () => {
            try {
                const response = await fetch(`${API}/evaluation`);

                if (!response.ok) {
                    throw new Error("Unable to load model evaluation.");
                }

                const data = await response.json();
                setEvaluation(data.evaluation || []);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchEvaluation();
    }, []);

    if (loading) {
        return <div className="performance-loading">Loading evaluation results...</div>;
    }

    if (error) {
        return <div className="performance-error">{error}</div>;
    }

    const overall = evaluation.find(
        (item) => item.channel === "OVERALL"
    );

    const channels = evaluation.filter(
        (item) => item.channel !== "OVERALL"
    );

    if (!overall) {
        return (
            <div className="performance-error">
                Overall evaluation results are unavailable.
            </div>
        );
    }

    const metrics = [
        {
            label: "Precision",
            value: overall.precision,
            icon: Target,
            description: "Reliability of detected anomalies",
        },
        {
            label: "Recall",
            value: overall.recall,
            icon: Activity,
            description: "Actual anomalies successfully detected",
        },
        {
            label: "F1 Score",
            value: overall.f1,
            icon: TrendingUp,
            description: "Balance between precision and recall",
        },
        {
            label: "False Positive Rate",
            value: overall.fpr,
            icon: AlertTriangle,
            description: "Normal samples incorrectly flagged",
        },
    ];

    const percentage = (value) =>
        `${(Number(value) * 100).toFixed(2)}%`;

    return (
        <div className="performance-page">
            <div className="performance-heading">
                <div>
                    <h2>Model Performance</h2>
                    <p>Evaluation results from the trained ORBY LSTM model</p>
                </div>
                <span className="performance-status">EVALUATED</span>
            </div>

            <section className="performance-metrics">
                {metrics.map((metric) => {
                    const Icon = metric.icon;

                    return (
                        <div className="performance-card" key={metric.label}>
                            <div className="performance-card-top">
                                <span>{metric.label}</span>
                                <Icon size={19} />
                            </div>

                            <h3>{percentage(metric.value)}</h3>
                            <p>{metric.description}</p>
                        </div>
                    );
                })}
            </section>

            <section className="evaluation-section">
                <div className="section-heading">
                    <h3>Channel-wise Evaluation</h3>
                    <span>5 evaluated channels</span>
                </div>

                <div className="evaluation-table-wrapper">
                    <table className="evaluation-table">
                        <thead>
                            <tr>
                                <th>Channel</th>
                                <th>Samples</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>FPR</th>
                            </tr>
                        </thead>

                        <tbody>
                            {channels.map((item) => (
                                <tr key={item.channel}>
                                    <td className="channel-name">{item.channel}</td>
                                    <td>{Number(item.samples).toLocaleString()}</td>
                                    <td>{percentage(item.precision)}</td>
                                    <td>{percentage(item.recall)}</td>
                                    <td className="f1-value">{percentage(item.f1)}</td>
                                    <td>{percentage(item.fpr)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>

            <section className="confusion-section">
                <div className="section-heading">
                    <h3>Prediction Breakdown</h3>
                    <span>Overall test results</span>
                </div>

                <div className="confusion-grid">
                    <div className="confusion-item">
                        <span>True Positives</span>
                        <strong>{Number(overall.TP).toLocaleString()}</strong>
                    </div>
                    <div className="confusion-item">
                        <span>False Positives</span>
                        <strong>{Number(overall.FP).toLocaleString()}</strong>
                    </div>
                    <div className="confusion-item">
                        <span>True Negatives</span>
                        <strong>{Number(overall.TN).toLocaleString()}</strong>
                    </div>
                    <div className="confusion-item">
                        <span>False Negatives</span>
                        <strong>{Number(overall.FN).toLocaleString()}</strong>
                    </div>
                </div>

                <p className="evaluation-total">
                    Total evaluated samples:{" "}
                    <strong>{Number(overall.samples).toLocaleString()}</strong>
                </p>
            </section>
        </div>
    );
}

export default PerformanceMonitor;
