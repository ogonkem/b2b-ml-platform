import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { apiGet } from "../api/client";

interface UsageResponse {
  used: number;
  limit: number;
  month: string;
}

export default function Dashboard() {
  const { user } = useAuth();
  const [usage, setUsage] = useState<UsageResponse | null>(null);

  useEffect(() => {
    apiGet<UsageResponse>("/v1/usage")
      .then(setUsage)
      .catch(() => {});
  }, []);

  return (
    <div className="page">
      <h1>Dashboard</h1>
      <p>
        Tenant: <strong>{user?.tenant_id}</strong>
      </p>
      {usage && (
        <p>
          {usage.used} / {usage.limit} predictions used this month ({usage.month})
        </p>
      )}
      <div className="quick-links">
        <Link to="/predict" className="card-link">
          Score a loan application
        </Link>
        <Link to="/batch" className="card-link">
          Upload a batch CSV
        </Link>
        <Link to="/labeled-data" className="card-link">
          Upload labeled outcomes
        </Link>
        <Link to="/usage" className="card-link">
          View usage &amp; history
        </Link>
        <Link to="/account" className="card-link">
          Account settings
        </Link>
        {user?.role === "admin" && (
          <Link to="/admin" className="card-link">
            Admin: cross-tenant metrics
          </Link>
        )}
      </div>
    </div>
  );
}
