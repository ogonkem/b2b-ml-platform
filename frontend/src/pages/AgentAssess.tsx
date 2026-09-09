import { useState, type FormEvent } from "react";
import { apiPost, ApiError, AGENT_BASE_URL } from "../api/client";
import { useAuth } from "../context/AuthContext";
import { REQUIRED_FIELDS, OPTIONAL_FIELDS } from "./predictFields";

interface StatisticalFactor {
  feature: string;
  value: number;
}
interface PolicyBasisEntry {
  rule: string;
  applicant_value: number;
  threshold: number;
  loan_type: string;
}
interface PolicyChunk {
  chunk_id: string;
  doc_id: string;
  doc_version: string;
  section_ref: string | null;
  chunk_text: string;
  score: number;
}
interface AssessResponse {
  decision: "approve" | "reject" | "refer";
  risk_score: number;
  statistical_factors: StatisticalFactor[];
  policy_basis: PolicyBasisEntry[];
  policy_chunks: PolicyChunk[];
  narrative: string;
  policy_aligned: boolean;
}

const DECISION_LABEL: Record<string, string> = {
  approve: "Approved",
  reject: "Rejected",
  refer: "Referred for manual review",
};

export default function AgentAssess() {
  const { user } = useAuth();
  const [values, setValues] = useState<Record<string, string | number>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loanType, setLoanType] = useState("");
  const [dti, setDti] = useState("");
  const [isFirstTime, setIsFirstTime] = useState(false);
  const [requestedAmount, setRequestedAmount] = useState("");
  const [result, setResult] = useState<AssessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function update(name: string, type: "number" | "text", raw: string) {
    setValues((prev) => {
      // Leaving a field blank should omit it from the request (so optional
      // fields fall back to their server-side default) rather than coerce
      // an empty string to 0 — same convention as pages/Predict.tsx.
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
      const applicant_profile: Record<string, unknown> = {};
      if (dti !== "") applicant_profile.dti = Number(dti);
      if (isFirstTime) applicant_profile.is_first_time_borrower = true;
      if (requestedAmount !== "") applicant_profile.requested_amount = Number(requestedAmount);

      const resp = await apiPost<AssessResponse>(
        "/v1/agent/assess",
        { tenant_id: user?.tenant_id, loan_type: loanType, applicant: values, applicant_profile },
        AGENT_BASE_URL,
      );
      setResult(resp);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Assessment failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h1>Agentic loan assessment</h1>
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

        <h2>Policy context</h2>
        <div className="field-grid">
          <label>
            Loan product (for policy matching)
            <input
              type="text"
              value={loanType}
              onChange={(e) => setLoanType(e.target.value)}
              placeholder="real_estate_payment_plan"
              required
            />
          </label>
          <label>
            Debt-to-income ratio (0-1)
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={dti}
              onChange={(e) => setDti(e.target.value)}
              placeholder="0.35"
            />
          </label>
          <label>
            Requested amount (first-time borrowers)
            <input
              type="number"
              value={requestedAmount}
              onChange={(e) => setRequestedAmount(e.target.value)}
              placeholder="50000"
            />
          </label>
          <label className="checkbox-label">
            <input type="checkbox" checked={isFirstTime} onChange={(e) => setIsFirstTime(e.target.checked)} />
            First-time borrower
          </label>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Assessing..." : "Run assessment"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="assess-result">
          <div
            className={`result-card ${
              result.decision === "reject" ? "risk-high" : result.decision === "approve" ? "risk-low" : ""
            }`}
          >
            <h2>{DECISION_LABEL[result.decision] ?? result.decision}</h2>
            <p>Risk score: {result.risk_score.toFixed(2)}</p>
            {!result.policy_aligned && (
              <p className="error">
                No policy documents were found for this tenant — this decision used a generic risk-score-only
                fallback, not your uploaded policy.
              </p>
            )}
            <p>{result.narrative}</p>
          </div>

          <div className="assess-columns">
            <div>
              <h2>Statistical factors</h2>
              <table className="job-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>SHAP contribution</th>
                  </tr>
                </thead>
                <tbody>
                  {result.statistical_factors.map((f) => (
                    <tr key={f.feature}>
                      <td>{f.feature}</td>
                      <td>{f.value.toFixed(4)}</td>
                    </tr>
                  ))}
                  {result.statistical_factors.length === 0 && (
                    <tr>
                      <td colSpan={2}>No statistical factors returned.</td>
                    </tr>
                  )}
                </tbody>
              </table>

              <h2>Policy thresholds applied</h2>
              <table className="job-table">
                <thead>
                  <tr>
                    <th>Rule</th>
                    <th>Applicant value</th>
                    <th>Threshold</th>
                  </tr>
                </thead>
                <tbody>
                  {result.policy_basis.map((t, i) => (
                    <tr key={`${t.rule}-${i}`}>
                      <td>{t.rule}</td>
                      <td>{t.applicant_value}</td>
                      <td>{t.threshold}</td>
                    </tr>
                  ))}
                  {result.policy_basis.length === 0 && (
                    <tr>
                      <td colSpan={3}>No threshold was breached.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div>
              <h2>Cited policy clauses</h2>
              {result.policy_chunks.length === 0 && <p>No policy text was cited for this decision.</p>}
              {result.policy_chunks.map((c) => (
                <div key={c.chunk_id} className="citation-card">
                  <div className="citation-header">
                    <span className="plan-tag">{c.section_ref ?? "unlabeled section"}</span>
                    <span className="citation-score">match {(c.score * 100).toFixed(0)}%</span>
                  </div>
                  <p>{c.chunk_text}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
