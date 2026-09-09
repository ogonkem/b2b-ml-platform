import { useState, useEffect, useCallback, type FormEvent } from "react";
import { apiGet, apiPost, apiUpload, ApiError, RAG_SERVICE_BASE_URL } from "../api/client";

interface DocumentSummary {
  id: string;
  filename: string;
  current_version: string;
  uploaded_at: string;
}
interface UploadResponse {
  doc_id: string;
  task_id: string;
  filename: string;
  status: string;
}
interface StatusResponse {
  doc_id: string;
  status: string;
  error: string | null;
}
interface UsageResponse {
  tenant_id: string;
  month: string;
  plan: string;
  ingestion: { used: number; limit: number };
  retrieval: { used: number; limit: number };
}
interface EvalMetrics {
  recall_at_k: number;
  mrr: number;
  total_queries: number;
  top_k: number;
}
interface EvalRun {
  doc_version: string;
  eval_set_version: string | null;
  status: string;
  metrics: EvalMetrics | null;
  lever_snapshot: Record<string, number> | null;
  error: string | null;
  created_at: string;
}
interface EvalReportResponse {
  doc_id: string;
  status: string;
  error: string | null;
  latest_run: EvalRun | null;
}

// Query-time levers (cheap — re-score current chunks only) vs chunking-time
// levers (expensive — trigger a real re-chunk/re-embed via reevaluate_document).
// See rag_service/ingest_task.py's module docstring for why the split matters.
const QUERY_TIME_LEVERS = [
  { key: "rrf_k", label: "RRF k" },
  { key: "candidate_limit", label: "Candidate limit" },
] as const;
const CHUNKING_TIME_LEVERS = [
  { key: "max_chunk_tokens", label: "Max chunk tokens" },
  { key: "min_chunk_tokens", label: "Min chunk tokens" },
  { key: "min_structural_chunks", label: "Min structural chunks" },
] as const;

export default function Policies() {
  const [file, setFile] = useState<File | null>(null);
  const [evalSetText, setEvalSetText] = useState("");
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [statuses, setStatuses] = useState<Record<string, StatusResponse>>({});
  const [evalReports, setEvalReports] = useState<Record<string, EvalReportResponse>>({});
  const [usage, setUsage] = useState<UsageResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [expandedDocId, setExpandedDocId] = useState<string | null>(null);
  const [leverInputs, setLeverInputs] = useState<Record<string, string>>({});
  const [reevaluating, setReevaluating] = useState<Record<string, boolean>>({});

  const loadDocs = useCallback(async () => {
    try {
      const resp = await apiGet<DocumentSummary[]>("/v1/documents", RAG_SERVICE_BASE_URL);
      setDocs(resp);
    } catch {
      // non-fatal — the table just won't refresh this tick
    }
  }, []);

  const loadUsage = useCallback(async () => {
    try {
      const resp = await apiGet<UsageResponse>("/v1/usage", RAG_SERVICE_BASE_URL);
      setUsage(resp);
    } catch {
      // non-fatal
    }
  }, []);

  const pollStatuses = useCallback(async (docList: DocumentSummary[]) => {
    const entries = await Promise.all(
      docList.map(async (d) => {
        try {
          const resp = await apiGet<StatusResponse>(`/v1/documents/status/${d.id}`, RAG_SERVICE_BASE_URL);
          return [d.id, resp] as const;
        } catch {
          return null;
        }
      }),
    );
    setStatuses((prev) => {
      const next = { ...prev };
      for (const entry of entries) {
        if (entry) next[entry[0]] = entry[1];
      }
      return next;
    });
  }, []);

  const pollEvalReports = useCallback(async (docList: DocumentSummary[]) => {
    const entries = await Promise.all(
      docList.map(async (d) => {
        try {
          const resp = await apiGet<EvalReportResponse>(`/v1/documents/${d.id}/eval`, RAG_SERVICE_BASE_URL);
          return [d.id, resp] as const;
        } catch {
          return null;
        }
      }),
    );
    setEvalReports((prev) => {
      const next = { ...prev };
      for (const entry of entries) {
        if (entry) next[entry[0]] = entry[1];
      }
      return next;
    });
  }, []);

  // Same polling shape as pages/Batch.tsx's job list: refresh on an
  // interval, unconditionally, for as long as the page is open.
  useEffect(() => {
    loadDocs();
    loadUsage();
    const interval = setInterval(() => {
      loadDocs();
      loadUsage();
    }, 5000);
    return () => clearInterval(interval);
  }, [loadDocs, loadUsage]);

  useEffect(() => {
    if (docs.length > 0) {
      pollStatuses(docs);
      pollEvalReports(docs);
    }
  }, [docs, pollStatuses, pollEvalReports]);

  async function handleUpload(e: FormEvent) {
    e.preventDefault();
    if (!file || !evalSetText.trim()) return;
    setError(null);
    setUploading(true);
    try {
      await apiUpload<UploadResponse>("/v1/documents", file, RAG_SERVICE_BASE_URL, { eval_set: evalSetText.trim() });
      setFile(null);
      setEvalSetText("");
      await loadDocs();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function setLever(docId: string, key: string, value: string) {
    setLeverInputs((prev) => ({ ...prev, [`${docId}:${key}`]: value }));
  }

  async function handleReeval(docId: string) {
    setError(null);
    setReevaluating((prev) => ({ ...prev, [docId]: true }));
    try {
      const body: Record<string, number> = {};
      for (const { key } of [...QUERY_TIME_LEVERS, ...CHUNKING_TIME_LEVERS]) {
        const raw = leverInputs[`${docId}:${key}`];
        if (raw !== undefined && raw.trim() !== "") {
          const parsed = Number(raw);
          if (!Number.isNaN(parsed)) body[key] = parsed;
        }
      }
      await apiPost(`/v1/documents/${docId}/eval`, body, RAG_SERVICE_BASE_URL);
      await pollEvalReports(docs);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Re-eval failed");
    } finally {
      setReevaluating((prev) => ({ ...prev, [docId]: false }));
    }
  }

  const ingestionPct = usage ? Math.min(100, (usage.ingestion.used / usage.ingestion.limit) * 100) : 0;
  const retrievalPct = usage ? Math.min(100, (usage.retrieval.used / usage.retrieval.limit) * 100) : 0;

  return (
    <div className="page">
      <h1>Policy documents</h1>
      <p>
        Upload your lending policy (PDF, DOCX, Markdown, or plain text) so the decision agent can cite the
        specific clause behind an automated decision.
      </p>

      <form onSubmit={handleUpload} className="upload-form">
        <input type="file" accept=".pdf,.docx,.md,.txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <div className="eval-set-input">
          <label>
            Ground-truth queries (required)
            <textarea
              required
              rows={4}
              placeholder='[{"query": "collateral requirements", "expected_text_substring": "collateral"}]'
              value={evalSetText}
              onChange={(e) => setEvalSetText(e.target.value)}
            />
          </label>
          <p>
            JSON array of <code>{"{query, expected_text_substring, expected_section_ref?}"}</code>. Every document
            needs this — it's what scores retrieval quality (recall@k, MRR) right after ingestion, and again any
            time you click Re-eval below.
          </p>
        </div>
        <button type="submit" disabled={!file || !evalSetText.trim() || uploading}>
          {uploading ? "Uploading..." : "Upload document"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}

      {usage && (
        <>
          <h2>RAG usage this month</h2>
          <div className="quota-bars">
            <div className="quota-bar-wrap">
              <p>
                <span className="plan-tag">{usage.plan} plan</span> — Ingestion: {usage.ingestion.used} /{" "}
                {usage.ingestion.limit} documents
              </p>
              <div className="quota-bar">
                <div className="quota-bar-fill" style={{ width: `${ingestionPct}%` }} />
              </div>
            </div>
            <div className="quota-bar-wrap">
              <p>Retrieval: {usage.retrieval.used} / {usage.retrieval.limit} queries</p>
              <div className="quota-bar">
                <div className="quota-bar-fill" style={{ width: `${retrievalPct}%` }} />
              </div>
            </div>
          </div>
        </>
      )}

      <h2>Your documents</h2>
      <table className="job-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Version</th>
            <th>Uploaded</th>
            <th>Ingestion status</th>
            <th>Retrieval quality</th>
          </tr>
        </thead>
        <tbody>
          {docs.map((d) => {
            const s = statuses[d.id];
            const evalReport = evalReports[d.id];
            const run = evalReport?.latest_run;
            const isExpanded = expandedDocId === d.id;
            return (
              <>
                <tr key={d.id}>
                  <td>{d.filename}</td>
                  <td>{d.current_version}</td>
                  <td>{new Date(d.uploaded_at).toLocaleString()}</td>
                  <td>
                    <div>
                      <span className={`status-badge status-${s?.status ?? "unknown"}`}>
                        {s?.status ?? "checking..."}
                      </span>
                    </div>
                    {s?.error && <p className="error status-error">{s.error}</p>}
                  </td>
                  <td>
                    {evalReport?.status === "no_eval_set" && <span className="muted">no eval_set uploaded</span>}
                    {run?.status === "complete" && run.metrics && (
                      <span>
                        recall@{run.metrics.top_k}: {(run.metrics.recall_at_k * 100).toFixed(0)}% · MRR:{" "}
                        {run.metrics.mrr.toFixed(2)}
                      </span>
                    )}
                    {run?.status === "failed" && <p className="error status-error">{run.error}</p>}
                    {(evalReport?.status === "queued" || evalReport?.status === "processing") && (
                      <span className={`status-badge status-${evalReport.status}`}>{evalReport.status}</span>
                    )}
                    {" "}
                    {run && (
                      <button type="button" className="link-button" onClick={() => setExpandedDocId(isExpanded ? null : d.id)}>
                        {isExpanded ? "Hide levers" : "Adjust levers"}
                      </button>
                    )}
                  </td>
                </tr>
                {isExpanded && (
                  <tr key={`${d.id}-levers`} className="lever-row">
                    <td colSpan={5}>
                      <div className="lever-panel">
                        <div className="lever-group">
                          <p className="lever-group-label">Query-time (cheap — re-scores current chunks)</p>
                          {QUERY_TIME_LEVERS.map(({ key, label }) => (
                            <label key={key}>
                              {label}
                              <input
                                type="number"
                                placeholder={run?.lever_snapshot?.[key]?.toString() ?? ""}
                                value={leverInputs[`${d.id}:${key}`] ?? ""}
                                onChange={(e) => setLever(d.id, key, e.target.value)}
                              />
                            </label>
                          ))}
                        </div>
                        <div className="lever-group">
                          <p className="lever-group-label">Chunking-time (expensive — re-chunks and re-embeds)</p>
                          {CHUNKING_TIME_LEVERS.map(({ key, label }) => (
                            <label key={key}>
                              {label}
                              <input
                                type="number"
                                placeholder={run?.lever_snapshot?.[key]?.toString() ?? ""}
                                value={leverInputs[`${d.id}:${key}`] ?? ""}
                                onChange={(e) => setLever(d.id, key, e.target.value)}
                              />
                            </label>
                          ))}
                        </div>
                        <button type="button" disabled={reevaluating[d.id]} onClick={() => handleReeval(d.id)}>
                          {reevaluating[d.id] ? "Re-evaluating..." : "Re-eval"}
                        </button>
                      </div>
                    </td>
                  </tr>
                )}
              </>
            );
          })}
          {docs.length === 0 && (
            <tr>
              <td colSpan={5}>No policy documents uploaded yet.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
