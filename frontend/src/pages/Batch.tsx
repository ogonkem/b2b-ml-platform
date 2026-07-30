import { useState, useEffect, useCallback, type FormEvent } from "react";
import { apiGet, apiUpload, ApiError } from "../api/client";

interface UploadResponse {
  job_id: string;
  tenant_id: string;
  rows_received: number;
  object: string;
  status: string;
}
interface JobSummary {
  job_id: string;
  status: string;
}
interface JobResult {
  job_id: string;
  status: string;
  rows_scored?: number;
  download_url?: string;
  error?: string;
}

export default function Batch() {
  const [file, setFile] = useState<File | null>(null);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [details, setDetails] = useState<Record<string, JobResult>>({});
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const loadJobs = useCallback(async () => {
    try {
      const resp = await apiGet<{ jobs: JobSummary[] }>("/v1/batch/jobs");
      setJobs(resp.jobs);
    } catch {
      // non-fatal — the table just won't refresh this tick
    }
  }, []);

  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 5000);
    return () => clearInterval(interval);
  }, [loadJobs]);

  async function handleUpload(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setUploading(true);
    try {
      await apiUpload<UploadResponse>("/v1/batch/upload", file);
      setFile(null);
      await loadJobs();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function checkDetails(jobId: string) {
    try {
      const resp = await apiGet<JobResult>(`/v1/batch/results/${jobId}`);
      setDetails((prev) => ({ ...prev, [jobId]: resp }));
    } catch {
      // non-fatal
    }
  }

  return (
    <div className="page">
      <h1>Batch upload</h1>
      <form onSubmit={handleUpload} className="upload-form">
        <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <button type="submit" disabled={!file || uploading}>
          {uploading ? "Uploading..." : "Upload CSV"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}

      <h2>Job history</h2>
      <table className="job-table">
        <thead>
          <tr>
            <th>Job ID</th>
            <th>Status</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((j) => {
            const detail = details[j.job_id];
            return (
              <tr key={j.job_id}>
                <td>{j.job_id}</td>
                <td>{detail?.status ?? j.status}</td>
                <td>
                  <button type="button" onClick={() => checkDetails(j.job_id)}>
                    Refresh
                  </button>
                  {detail?.download_url && (
                    <a href={detail.download_url} target="_blank" rel="noreferrer">
                      Download
                    </a>
                  )}
                  {detail?.error && <span className="error">{detail.error}</span>}
                </td>
              </tr>
            );
          })}
          {jobs.length === 0 && (
            <tr>
              <td colSpan={3}>No batch jobs yet.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
