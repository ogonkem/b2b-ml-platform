import { useState, useEffect } from "react";
import { apiGet, apiPost, ApiError } from "../api/client";

interface MeResponse {
  email: string;
  tenant_id: string;
  role: string;
}
interface ApiKeyResponse {
  api_key: string;
}

export default function Account() {
  const [me, setMe] = useState<MeResponse | null>(null);
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    apiGet<MeResponse>("/auth/me")
      .then(setMe)
      .catch(() => {});
  }, []);

  async function handleGenerate() {
    setError(null);
    setGenerating(true);
    try {
      const resp = await apiPost<ApiKeyResponse>("/auth/api-key");
      setApiKey(resp.api_key);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to generate key");
    } finally {
      setGenerating(false);
    }
  }

  return (
    <div className="page">
      <h1>Account</h1>
      {me && (
        <div className="account-info">
          <p>
            <strong>Email:</strong> {me.email}
          </p>
          <p>
            <strong>Tenant ID:</strong> {me.tenant_id}
          </p>
          <p>
            <strong>Role:</strong> {me.role}
          </p>
        </div>
      )}

      <h2>Programmatic API key</h2>
      <p>
        Generate a persistent key for scripts and integrations, separate from your browser session.
        Regenerating invalidates any previous key.
      </p>
      <button onClick={handleGenerate} disabled={generating}>
        {generating ? "Generating..." : "Generate new API key"}
      </button>
      {error && <p className="error">{error}</p>}
      {apiKey && (
        <div className="result-card">
          <p>Copy this now — it won't be shown again:</p>
          <code className="invite-code">{apiKey}</code>
        </div>
      )}
    </div>
  );
}
