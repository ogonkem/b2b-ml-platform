import { useState, useEffect } from "react";
import { apiGet } from "../api/client";

interface UsageResponse {
  tenant_id: string;
  month: string;
  used: number;
  limit: number;
}
interface PredictionRecord {
  application_id: number;
  probability: number;
  prediction: number;
  model_version: string;
  timestamp: string;
}

export default function Usage() {
  const [usage, setUsage] = useState<UsageResponse | null>(null);
  const [predictions, setPredictions] = useState<PredictionRecord[]>([]);

  useEffect(() => {
    apiGet<UsageResponse>("/v1/usage")
      .then(setUsage)
      .catch(() => {});
    apiGet<{ predictions: PredictionRecord[] }>("/v1/predictions/history")
      .then((r) => setPredictions(r.predictions))
      .catch(() => {});
  }, []);

  const pct = usage ? Math.min(100, (usage.used / usage.limit) * 100) : 0;

  return (
    <div className="page">
      <h1>Usage</h1>
      {usage && (
        <div className="quota-bar-wrap">
          <p>
            {usage.used} / {usage.limit} predictions used this month ({usage.month})
          </p>
          <div className="quota-bar">
            <div className="quota-bar-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}

      <h2>Recent predictions</h2>
      <table className="job-table">
        <thead>
          <tr>
            <th>Application ID</th>
            <th>Probability</th>
            <th>Decision</th>
            <th>Model version</th>
            <th>When</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((p, i) => (
            <tr key={i}>
              <td>{p.application_id}</td>
              <td>{(p.probability * 100).toFixed(2)}%</td>
              <td>{p.prediction === 1 ? "Default" : "Repaid"}</td>
              <td>{p.model_version}</td>
              <td>{p.timestamp}</td>
            </tr>
          ))}
          {predictions.length === 0 && (
            <tr>
              <td colSpan={5}>No predictions logged yet.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
