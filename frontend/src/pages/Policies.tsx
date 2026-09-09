import { useState, useEffect, useCallback, type FormEvent } from "react";
import { apiGet, apiUpload, ApiError, RAG_SERVICE_BASE_URL } from "../api/client";

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

export default function Policies() {
  const [file, setFile] = useState<File | null>(null);
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [statuses, setStatuses] = useState<Record<string, StatusResponse>>({});
  const [usage, setUsage] = useState<UsageResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

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
    if (docs.length > 0) pollStatuses(docs);
  }, [docs, pollStatuses]);

  async function handleUpload(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setUploading(true);
    try {
      await apiUpload<UploadResponse>("/v1/documents", file, RAG_SERVICE_BASE_URL);
      setFile(null);
      await loadDocs();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setUploading(false);
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
        <button type="submit" disabled={!file || uploading}>
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
          </tr>
        </thead>
        <tbody>
          {docs.map((d) => {
            const s = statuses[d.id];
            return (
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
              </tr>
            );
          })}
          {docs.length === 0 && (
            <tr>
              <td colSpan={4}>No policy documents uploaded yet.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
