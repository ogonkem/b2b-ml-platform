import { useState, type FormEvent } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { ApiError } from "../api/client";

export default function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState<"create" | "join">("create");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [tenantName, setTenantName] = useState("");
  const [inviteCode, setInviteCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [createdInviteCode, setCreatedInviteCode] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const resp = await register({
        email,
        password,
        ...(mode === "create" ? { tenant_name: tenantName } : { invite_code: inviteCode }),
      });
      if (resp.invite_code) {
        setCreatedInviteCode(resp.invite_code);
      } else {
        navigate("/dashboard");
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Registration failed");
    } finally {
      setLoading(false);
    }
  }

  if (createdInviteCode) {
    return (
      <div className="auth-page">
        <div className="auth-card">
          <h1>Organization created</h1>
          <p>Share this invite code with your team so they can join it:</p>
          <code className="invite-code">{createdInviteCode}</code>
          <button onClick={() => navigate("/dashboard")}>Continue to dashboard</button>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-page">
      <form className="auth-card" onSubmit={handleSubmit}>
        <h1>Register</h1>
        {error && <p className="error">{error}</p>}
        <div className="toggle">
          <button type="button" className={mode === "create" ? "active" : ""} onClick={() => setMode("create")}>
            Create new organization
          </button>
          <button type="button" className={mode === "join" ? "active" : ""} onClick={() => setMode("join")}>
            Join existing organization
          </button>
        </div>
        <label>
          Email
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            minLength={8}
            maxLength={72}
            required
          />
        </label>
        {mode === "create" ? (
          <label>
            Organization name
            <input type="text" value={tenantName} onChange={(e) => setTenantName(e.target.value)} required />
          </label>
        ) : (
          <label>
            Invite code
            <input type="text" value={inviteCode} onChange={(e) => setInviteCode(e.target.value)} required />
          </label>
        )}
        <button type="submit" disabled={loading}>
          {loading ? "Registering..." : "Register"}
        </button>
        <p>
          Already have an account? <Link to="/login">Log in</Link>
        </p>
      </form>
    </div>
  );
}
