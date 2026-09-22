import { FormEvent, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import Plotly from "plotly.js-dist-min";
import "./styles.css";

type Result = {
  channels: number[];
  phases: number[];
  accepted: boolean;
  gate_reason: string;
  confidence: number;
  ood_score: number;
  metrics: {
    throughput_mbps: number;
    mean_sinr_db: number;
    energy_j: number;
    spectral_efficiency_bps_hz: number;
  };
  simulation: { per_user_throughput_mbps: number[] };
  optimizer_metadata: {
    refinement_invoked?: boolean;
    pipeline?: string[];
    active_learning_buffer_size?: number;
  };
};

type FormValues = {
  snr_db: number;
  csi_error: number;
  mobility: number;
  num_elements: number;
  phase_resolution: number;
  num_users: number;
  environment: string;
};

const defaults: FormValues = {
  snr_db: 20,
  csi_error: 0,
  mobility: 0,
  num_elements: 16,
  phase_resolution: 2,
  num_users: 4,
  environment: "urban",
};

function App() {
  const [values, setValues] = useState(defaults);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runSimulation(event?: FormEvent) {
    event?.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/aurora/optimize", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...values, seed: 42 }),
      });
      if (!response.ok) throw new Error(`API request failed (${response.status})`);
      setResult(await response.json() as Result);
      window.location.hash = "output";
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!result) return;
    Plotly.newPlot("throughput-chart", [{
      x: result.simulation.per_user_throughput_mbps.map((_, index) => `UE ${index + 1}`),
      y: result.simulation.per_user_throughput_mbps,
      type: "bar",
      marker: { color: "#39d0c8" },
    }], {
      title: "Per-user throughput",
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#d9e8ff" },
      yaxis: { title: "Mbps", gridcolor: "#263b5d" },
      margin: { t: 45, r: 20, b: 45, l: 55 },
    }, { responsive: true, displayModeBar: false });
  }, [result]);

  const update = (key: keyof FormValues, value: string) => {
    setValues((current) => ({ ...current, [key]: key === "environment" ? value : Number(value) }));
  };

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">AURORA-RIS / DYNAMIC 6G</p>
          <h1>Adaptive RIS Optimization</h1>
          <p className="subtitle">Run a scenario, inspect the confidence gate, and view the measured simulation output on a separate result page.</p>
        </div>
        <a className="api-link" href="http://localhost:8000/docs" target="_blank">API docs ↗</a>
      </header>

      <section className="workspace">
        <form className="panel controls" onSubmit={runSimulation}>
          <h2>Simulation controls</h2>
          <label>SNR (dB)<input type="number" value={values.snr_db} onChange={(e) => update("snr_db", e.target.value)} /></label>
          <label>CSI error (0-1)<input type="number" min="0" max="1" step="0.05" value={values.csi_error} onChange={(e) => update("csi_error", e.target.value)} /></label>
          <label>Mobility<input type="number" min="0" step="0.1" value={values.mobility} onChange={(e) => update("mobility", e.target.value)} /></label>
          <label>RIS elements<input type="number" min="1" value={values.num_elements} onChange={(e) => update("num_elements", e.target.value)} /></label>
          <label>Phase resolution (bits)<input type="number" min="1" max="8" value={values.phase_resolution} onChange={(e) => update("phase_resolution", e.target.value)} /></label>
          <label>Users<input type="number" min="1" value={values.num_users} onChange={(e) => update("num_users", e.target.value)} /></label>
          <label>Environment<select value={values.environment} onChange={(e) => update("environment", e.target.value)}><option>urban</option><option>indoor</option><option>suburban</option><option>rural</option></select></label>
          <button disabled={loading}>{loading ? "Running..." : "Run optimization"}</button>
          {error && <p className="error">{error}. Start the API with <code>uvicorn backend.app.main:app --reload</code>.</p>}
        </form>

        <section className="panel output">
          <div className="output-heading">
            <div><p className="eyebrow">OUTPUT PAGE</p><h2>Experiment result</h2></div>
            {result && <span className={result.accepted ? "badge good" : "badge warn"}>{result.accepted ? "AI ACCEPTED" : "CLASSICAL REFINEMENT"}</span>}
          </div>
          {!result ? <div className="empty"><strong>No output yet</strong><span>Configure the scenario and click “Run optimization”. Results will appear here.</span></div> : (
            <>
              <div className="cards">
                <Metric label="Throughput" value={`${result.metrics.throughput_mbps.toFixed(3)} Mbps`} />
                <Metric label="Mean SINR" value={`${result.metrics.mean_sinr_db.toFixed(3)} dB`} />
                <Metric label="Confidence" value={result.confidence.toFixed(3)} />
                <Metric label="OOD score" value={result.ood_score.toFixed(3)} />
              </div>
              <div id="throughput-chart" className="chart" />
              <div className="details">
                <span><b>Gate:</b> {result.gate_reason}</span>
                <span><b>Channels:</b> {result.channels.join(", ")}</span>
                <span><b>Optimizer called:</b> {result.optimizer_metadata.refinement_invoked ? "Yes" : "No"}</span>
                <span><b>Pipeline:</b> {result.optimizer_metadata.pipeline?.join(" → ")}</span>
                <span><b>Energy:</b> {result.metrics.energy_j.toFixed(5)} J</span>
                <span><b>Active samples:</b> {result.optimizer_metadata.active_learning_buffer_size ?? 0}</span>
              </div>
            </>
          )}
        </section>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

createRoot(document.getElementById("root")!).render(<App />);
