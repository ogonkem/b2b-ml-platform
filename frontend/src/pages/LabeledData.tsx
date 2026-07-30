import { useState, useEffect, useCallback, type FormEvent } from "react";
import { apiGet, apiUpload, ApiError } from "../api/client";

interface UploadResponse {
  object: string;
  tenant_id: string;
  rows_received: number;
  status: string;
}
interface HistoryRow {
  object_name: string;
  rows_received: number;
  timestamp: string;
}

export default function LabeledData() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [history, setHistory] = useState<HistoryRow[]>([]);

  const loadHistory = useCallback(async () => {
    try {
      const resp = await apiGet<{ uploads: HistoryRow[] }>("/v1/labeled-data/history");
      setHistory(resp.uploads);
    } catch {
      // non-fatal — the table just won't refresh this tick
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  async function handleUpload(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setResult(null);
    setUploading(true);
    try {
      const resp = await apiUpload<UploadResponse>("/v1/labeled-data", file);
      setResult(resp);
      setFile(null);
      await loadHistory();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="page">
      <h1>Labeled data upload</h1>
      <p>
        Upload a CSV of actual loan outcomes (with an <code>actual_outcome</code> or <code>Status</code> column)
        for drift detection against the training baseline.
      </p>
      <form onSubmit={handleUpload} className="upload-form">
        <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <button type="submit" disabled={!file || uploading}>
          {uploading ? "Uploading..." : "Upload CSV"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="result-card">
          <p>
            Stored {result.rows_received} rows &rarr; {result.object}
          </p>
        </div>
      )}

      <h2>Upload history</h2>
      <table className="job-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Rows</th>
            <th>Uploaded</th>
          </tr>
        </thead>
        <tbody>
          {history.map((h) => (
            <tr key={h.object_name}>
              <td>{h.object_name}</td>
              <td>{h.rows_received}</td>
              <td>{new Date(h.timestamp).toLocaleString()}</td>
            </tr>
          ))}
          {history.length === 0 && (
            <tr>
              <td colSpan={3}>No labeled-data uploads yet.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
