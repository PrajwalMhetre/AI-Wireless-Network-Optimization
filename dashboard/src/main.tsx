import { FormEvent, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import Plotly from "plotly.js-dist-min";
import "./styles.css";

type NumericResult = {
  throughput_mbps?: number;
  mean_sinr_db?: number;
  energy_j?: number;
  spectral_efficiency_bps_hz?: number;
};

type Result = {
  channels: number[];
  phases: number[];
  accepted: boolean;
  gate_reason: string;
  confidence: number;
  ood_score: number;
  metrics: NumericResult & { [key: string]: number | undefined };
  simulation: { per_user_throughput_mbps?: number[]; [key: string]: unknown };
  optimizer_metadata: {
    refinement_invoked?: boolean;
    pipeline?: string[];
    active_learning_buffer_size?: number;
    runtime_ms?: number;
    objective_evaluations?: number;
    optimizer_iterations?: number;
  };
  decision?: {
    controller_decision?: string;
    optimizer_called?: boolean;
    reason?: string;
  };
  gate?: {
    passed?: boolean;
    confidence_threshold?: number;
    ood_threshold?: number;
    reason?: string;
  };
  optimizer?: {
    called?: boolean;
    iterations?: number;
    objective_evaluations?: number;
    runtime_ms?: number;
    model?: string;
  };
  ris?: {
    phases?: number[];
    phase_resolution_bits?: number;
  };
  comparison?: {
    before_refinement?: NumericResult;
    after_refinement?: NumericResult;
  };
  scenario?: {
    num_users?: number;
    environment?: string;
    phase_resolution?: number;
  };
  experiment_id?: string;
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

function formatNumber(value: number | undefined, suffix = "", digits = 3) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "NOT MEASURED";
  }
  return `${value.toFixed(digits)}${suffix}`;
}

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
      if (!response.ok) {
        throw new Error("Backend unavailable. Unable to run experiment. Retry.");
      }
      const payload = (await response.json()) as Result;
      setResult(payload);
    } catch (requestError) {
      const message = requestError instanceof Error
        ? requestError.message
        : "Backend unavailable. Unable to run experiment. Retry.";
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!result || !Array.isArray(result.simulation?.per_user_throughput_mbps)) return;
    const perUser = result.simulation.per_user_throughput_mbps as number[];
    Plotly.newPlot("throughput-chart", [{
      x: perUser.map((_, index) => `UE ${index + 1}`),
      y: perUser,
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

  const decisionLabel = result?.decision?.controller_decision || (result?.accepted ? "AI_ACCEPTED" : "CLASSICAL_REFINEMENT");
  const decisionClass = decisionLabel === "AI_ACCEPTED" ? "badge good" : "badge warn";
  const gateReason = result?.decision?.reason || result?.gate?.reason || result?.gate_reason || "NOT AVAILABLE";
  const optimizerCalled = typeof result?.decision?.optimizer_called === "boolean"
    ? result.decision.optimizer_called
    : Boolean(result?.optimizer_metadata?.refinement_invoked);

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">AURORA-RIS / DYNAMIC 6G</p>
          <h1>Adaptive RIS Optimization</h1>
          <p className="subtitle">Controller decision, gate thresholds, and live measurements are rendered only from the backend response.</p>
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
          {error && <p className="error">{error}</p>}
        </form>

        <section className="panel output">
          <div className="output-heading">
            <div><p className="eyebrow">OUTPUT PAGE</p><h2>Experiment result</h2></div>
            {result && <span className={decisionClass}>{decisionLabel}</span>}
          </div>
          {!result ? <div className="empty"><strong>No output yet</strong><span>Configure the scenario and click “Run optimization”. Results will appear here.</span></div> : (
            <>
              <div className="cards">
                <Metric label="Throughput" value={formatNumber(result.metrics?.throughput_mbps, " Mbps")} />
                <Metric label="Mean SINR" value={formatNumber(result.metrics?.mean_sinr_db, " dB")} />
                <Metric label="Confidence" value={formatNumber(result.confidence, "", 3)} />
                <Metric label="OOD score" value={formatNumber(result.ood_score, "", 3)} />
                <Metric label="Energy" value={formatNumber(result.metrics?.energy_j, " J")} />
                <Metric label="Model" value={result.optimizer?.model || result.optimizer_metadata?.pipeline?.join(" → ") || "NOT AVAILABLE"} />
              </div>
              <div id="throughput-chart" className="chart" />
              <div className="details">
                <span><b>Decision:</b> {decisionLabel}</span>
                <span><b>Reason:</b> {gateReason}</span>
                <span><b>Gate passed:</b> {String(result.gate?.passed ?? result.accepted)}</span>
                <span><b>Thresholds:</b> {formatNumber(result.gate?.confidence_threshold, "", 3)} / {formatNumber(result.gate?.ood_threshold, "", 3)}</span>
                <span><b>Optimizer called:</b> {optimizerCalled ? "Yes" : "No"}</span>
                <span><b>Runtime:</b> {formatNumber(result.optimizer?.runtime_ms ?? result.optimizer_metadata?.runtime_ms, " ms")}</span>
                <span><b>RIS phases:</b> {Array.isArray(result.ris?.phases) && result.ris.phases.length > 0 ? result.ris.phases.map((value) => value.toFixed(3)).join(", ") : "NOT AVAILABLE"}</span>
                <span><b>Before refinement throughput:</b> {formatNumber(result.comparison?.before_refinement?.throughput_mbps, " Mbps")}</span>
                <span><b>After refinement throughput:</b> {formatNumber(result.comparison?.after_refinement?.throughput_mbps, " Mbps")}</span>
                <span><b>Experiment ID:</b> {result.experiment_id || "NOT AVAILABLE"}</span>
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
