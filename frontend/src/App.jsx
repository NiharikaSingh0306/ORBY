
import { useState } from "react";
import {
  Activity,
  Orbit,
  LayoutDashboard,
  Radio,
  AlertTriangle,
  ChartNoAxesCombined,
  Settings,
  Bell,
  ChevronDown,
  Satellite,
  ArrowUpRight,
  CircleDot,
} from "lucide-react";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import TelemetryMonitor from "./components/TelemetryMonitor";
import "./index.css";

const telemetryData = [
  { time: "00:00", value: 42 },
  { time: "02:00", value: 44 },
  { time: "04:00", value: 43 },
  { time: "06:00", value: 46 },
  { time: "08:00", value: 45 },
  { time: "10:00", value: 48 },
  { time: "12:00", value: 47 },
  { time: "14:00", value: 45 },
  { time: "16:00", value: 49 },
  { time: "18:00", value: 47 },
  { time: "20:00", value: 50 },
  { time: "22:00", value: 48 },
];

const navigation = [
  { name: "Overview", icon: LayoutDashboard },
  { name: "Telemetry", icon: Radio },
  { name: "Anomalies", icon: AlertTriangle },
  { name: "Performance", icon: ChartNoAxesCombined },
];

function App() {
  const [spacecraft, setSpacecraft] = useState("SMAP");
  const [activePage, setActivePage] = useState("Overview");

  const channelCount = spacecraft === "SMAP" ? 55 : 27;

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Orbit size={23} />
          </div>
          <span>orby<span className="brand-dot">.</span></span>
        </div>

        <div className="nav-section">
          <p className="nav-label">WORKSPACE</p>

          {navigation.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.name}
                className={`nav-item ${activePage === item.name ? "active" : ""
                  }`}
                onClick={() => setActivePage(item.name)}
              >
                <Icon size={18} />
                <span>{item.name}</span>
              </button>
            );
          })}
        </div>

        <div className="sidebar-bottom">
          <button className="nav-item">
            <Settings size={18} />
            <span>Settings</span>
          </button>

          <div className="profile">
            <div className="avatar">N</div>
            <div>
              <p className="profile-name">Mission Operator</p>
              <p className="profile-role">Administrator</p>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="breadcrumb">WORKSPACE / {activePage.toUpperCase()}</p>
            <h1>{activePage}</h1>
          </div>

          <div className="topbar-right">
            <div className="system-status">
              <span className="status-dot"></span>
              System online
            </div>
            <button className="icon-button" aria-label="Notifications">
              <Bell size={19} />
            </button>
          </div>
        </header>

        <section className="welcome-section">
          <div>
            <p className="eyebrow">MISSION CONTROL</p>
            <h2>Spacecraft overview</h2>
            <p className="section-description">
              Monitor telemetry and identify unexpected spacecraft behavior.
            </p>
          </div>

          <div className="select-wrapper">
            <Satellite size={17} />
            <select
              value={spacecraft}
              onChange={(e) => setSpacecraft(e.target.value)}
            >
              <option value="SMAP">SMAP Satellite</option>
              <option value="MSL">MSL Curiosity Rover</option>
            </select>
            <ChevronDown size={15} />
          </div>
        </section>

        {activePage === "Overview" ? (
          <>
            <section className="stats-grid">
              <div className="stat-card">
                <div className="stat-heading">
                  <span>Spacecraft</span>
                  <Satellite size={18} />
                </div>
                <h3>{spacecraft}</h3>
                <p>{spacecraft === "SMAP" ? "Earth observation" : "Mars exploration"}</p>
              </div>

              <div className="stat-card">
                <div className="stat-heading">
                  <span>Telemetry channels</span>
                  <Activity size={18} />
                </div>
                <h3>{channelCount}</h3>
                <p>Monitored channels</p>
              </div>

              <div className="stat-card">
                <div className="stat-heading">
                  <span>Model</span>
                  <Orbit size={18} />
                </div>
                <h3>LSTM</h3>
                <p>Time-series prediction</p>
              </div>

              <div className="stat-card">
                <div className="stat-heading">
                  <span>Dataset anomalies</span>
                  <AlertTriangle size={18} />
                </div>
                <h3>105</h3>
                <p>Across both missions</p>
              </div>
            </section>

            <section className="chart-card">
              <div className="chart-header">
                <div>
                  <p className="eyebrow">TELEMETRY DATA</p>
                  <h3>Signal overview</h3>
                  <p className="chart-subtitle">
                    Sample telemetry variation over time
                  </p>
                </div>

                <div className="channel-tag">
                  <CircleDot size={13} />
                  Channel P-1
                </div>
              </div>

              <div className="chart-legend">
                <span><i className="legend-blue"></i> Telemetry value</span>
                <span className="demo-label">ILLUSTRATIVE DATA</span>
              </div>

              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={telemetryData}
                    margin={{ top: 10, right: 12, left: -18, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="telemetryGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#60A5FA" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="#60A5FA" stopOpacity={0} />
                      </linearGradient>
                    </defs>

                    <CartesianGrid
                      stroke="#263449"
                      strokeDasharray="4 5"
                      vertical={false}
                    />

                    <XAxis
                      dataKey="time"
                      stroke="#77869b"
                      tickLine={false}
                      axisLine={false}
                      tick={{ fontSize: 11 }}
                    />

                    <YAxis
                      stroke="#77869b"
                      tickLine={false}
                      axisLine={false}
                      tick={{ fontSize: 11 }}
                    />

                    <Tooltip
                      contentStyle={{
                        background: "#172235",
                        border: "1px solid #334155",
                        borderRadius: "8px",
                        color: "#fff",
                      }}
                    />

                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#60A5FA"
                      strokeWidth={2.5}
                      fill="url(#telemetryGradient)"
                      activeDot={{ r: 5, fill: "#60A5FA" }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="bottom-section">
              <div className="info-card">
                <div className="info-icon blue-icon">
                  <Activity size={19} />
                </div>
                <div>
                  <h4>Predictive monitoring</h4>
                  <p>
                    An LSTM model will learn normal telemetry patterns and
                    predict upcoming values.
                  </p>
                </div>
              </div>

              <div className="info-card">
                <div className="info-icon red-icon">
                  <AlertTriangle size={19} />
                </div>
                <div>
                  <h4>Anomaly detection</h4>
                  <p>
                    Unexpected deviations will be highlighted for further
                    investigation.
                  </p>
                </div>
              </div>
            </section>
          </>
        ) : activePage === "Telemetry" ? (
          <div id="monitor">
            <TelemetryMonitor />
          </div>
        ) : (
          <section className="placeholder-card">
            <div className="placeholder-icon">
              {activePage === "Anomalies" ? (
                <AlertTriangle size={27} />
              ) : (
                <ChartNoAxesCombined size={27} />
              )}
            </div>

            <h2>{activePage}</h2>

            <p>
              This module is coming soon. We'll connect it to the
              actual NASA telemetry dataset.
            </p>

            <button
              className="back-button"
              onClick={() => setActivePage("Overview")}
            >
              Back to overview <ArrowUpRight size={15} />
            </button>
          </section>
        )}

        <footer>
          <span>ORBY © 2026</span>
          <span>SPACECRAFT ANOMALY DETECTION SYSTEM</span>
        </footer>
      </main>
    </div>
  );
}
export default App;