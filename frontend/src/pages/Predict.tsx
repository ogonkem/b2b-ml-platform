import { useState, type FormEvent } from "react";
import { apiPost, ApiError } from "../api/client";
import { REQUIRED_FIELDS, OPTIONAL_FIELDS } from "./predictFields";

interface PredictResponse {
  application_id: number;
  default_prediction: number;
  default_probability: number;
  status: string;
}

export default function Predict() {
  const [values, setValues] = useState<Record<string, string | number>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function update(name: string, type: "number" | "text", raw: string) {
    setValues((prev) => {
      // Leaving a field blank should omit it from the request (so optional
      // fields fall back to their server-side default) rather than coerce
      // an empty string to 0.
      if (raw === "") {
        const next = { ...prev };
        delete next[name];
        return next;
      }
      return { ...prev, [name]: type === "number" ? Number(raw) : raw };
    });
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const resp = await apiPost<PredictResponse>("/v1/predict", values);
      setResult(resp);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h1>Real-time prediction</h1>
      <form onSubmit={handleSubmit} className="predict-form">
        <div className="field-grid">
          {REQUIRED_FIELDS.map((f) => (
            <label key={f.name}>
              {f.label}
              <input
                type={f.type}
                value={values[f.name] ?? ""}
                onChange={(e) => update(f.name, f.type, e.target.value)}
                placeholder={String(f.default)}
                required
              />
            </label>
          ))}
        </div>
        <button type="button" className="link-btn" onClick={() => setShowAdvanced((s) => !s)}>
          {showAdvanced ? "Hide" : "Show"} advanced fields
        </button>
        {showAdvanced && (
          <div className="field-grid">
            {OPTIONAL_FIELDS.map((f) => (
              <label key={f.name}>
                {f.label}
                <input
                  type={f.type}
                  value={values[f.name] ?? ""}
                  onChange={(e) => update(f.name, f.type, e.target.value)}
                  placeholder={String(f.default)}
                />
              </label>
            ))}
          </div>
        )}
        <button type="submit" disabled={loading}>
          {loading ? "Scoring..." : "Score application"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className={`result-card ${result.default_prediction === 1 ? "risk-high" : "risk-low"}`}>
          <h2>{result.default_prediction === 1 ? "Likely default" : "Likely repaid"}</h2>
          <p>Probability of default: {(result.default_probability * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}
