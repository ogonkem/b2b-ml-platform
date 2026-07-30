import { useEffect, useState } from "react";
import { apiGet } from "../api/client";

interface TenantSummary {
  tenant_id: string;
  name: string;
  created_at: string;
  user_count: number;
  quota_used: number;
  quota_limit: number;
}

export default function Admin() {
  const [tenants, setTenants] = useState<TenantSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<{ tenants: TenantSummary[] }>("/admin/tenants")
      .then((r) => setTenants(r.tenants))
      .catch(() => setError("Failed to load tenants"));
  }, []);

  return (
    <div className="page">
      <h1>Admin — all tenants</h1>
      {error && <p className="error">{error}</p>}
      <table className="job-table">
        <thead>
          <tr>
            <th>Organization</th>
            <th>Tenant ID</th>
            <th>Users</th>
            <th>Quota used</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {tenants.map((t) => (
            <tr key={t.tenant_id}>
              <td>{t.name}</td>
              <td>{t.tenant_id}</td>
              <td>{t.user_count}</td>
              <td>
                {t.quota_used} / {t.quota_limit}
              </td>
              <td>{new Date(t.created_at).toLocaleString()}</td>
            </tr>
          ))}
          {tenants.length === 0 && (
            <tr>
              <td colSpan={5}>No tenants yet.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
