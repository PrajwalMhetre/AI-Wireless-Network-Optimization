import React, {useEffect, useState} from "react";
import {createRoot} from "react-dom/client";
import Plotly from "plotly.js-dist-min";

type Result = {metrics: {throughput_mbps:number}, simulation:{per_user_throughput_mbps:number[]}};
function Dashboard() {
  const [result, setResult] = useState<Result | null>(null);
  useEffect(() => { fetch("/optimize", {method:"POST", headers:{"content-type":"application/json"}, body:"{}"})
    .then(r => r.json()).then(setResult); }, []);
  useEffect(() => { if (result) Plotly.newPlot("plot", [{y:result.simulation.per_user_throughput_mbps, type:"bar"}], {title:"Per-user throughput (Mbps)"}); }, [result]);
  return <main><h1>AURORA-RIS dashboard</h1>{result ? <p>Throughput: {result.metrics.throughput_mbps.toFixed(3)} Mbps</p> : <p>Loading...</p>}<div id="plot"/></main>;
}
createRoot(document.getElementById("root")!).render(<Dashboard/>);
