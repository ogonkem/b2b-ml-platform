import { Link } from "react-router-dom";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function ApiDocs() {
  return (
    <div className="page docs-page">
      <h1>API documentation</h1>
      <p>
        Everything on this page works against the same base URL your browser is already talking to:{" "}
        <code>{BASE_URL}</code>. For the full request/response schema and a live try-it-out console, see{" "}
        <a href={`${BASE_URL}/docs`} target="_blank" rel="noreferrer">
          Swagger UI
        </a>
        .
      </p>

      <h2>Authentication</h2>
      <p>
        Every request needs <code>Authorization: Bearer &lt;token&gt;</code>. Three kinds of token work, and
        they all resolve to the same tenant:
      </p>
      <table>
        <thead>
          <tr>
            <th>Token type</th>
            <th>Where it comes from</th>
            <th>When to use it</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Static token</td>
            <td>A pre-shared value from your <code>API_TOKENS</code> allowlist</td>
            <td>Existing integrations, CI, scripts that predate user accounts</td>
          </tr>
          <tr>
            <td>JWT</td>
            <td><code>POST /auth/login</code> or <code>POST /auth/register</code></td>
            <td>Browser sessions — short-lived, not meant for long-running scripts</td>
          </tr>
          <tr>
            <td>API key (<code>sk_...</code>)</td>
            <td><code>POST /auth/api-key</code> (see the Account page)</td>
            <td>Scripts and server-to-server integrations tied to your account</td>
          </tr>
        </tbody>
      </table>

      <h2>Endpoints</h2>
      <table>
        <thead>
          <tr>
            <th>Method &amp; path</th>
            <th>Purpose</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>POST /v1/predict</code></td>
            <td>Score a single loan application in real time</td>
          </tr>
          <tr>
            <td><code>POST /v1/batch/upload</code></td>
            <td>Upload a CSV of applications for async bulk scoring</td>
          </tr>
          <tr>
            <td><code>GET /v1/batch/jobs</code></td>
            <td>List your recent batch job IDs and statuses</td>
          </tr>
          <tr>
            <td><code>GET /v1/batch/results/&#123;job_id&#125;</code></td>
            <td>Poll a job; returns a download URL once complete</td>
          </tr>
          <tr>
            <td><code>POST /v1/labeled-data</code></td>
            <td>Upload actual outcomes for drift detection and retraining</td>
          </tr>
          <tr>
            <td><code>GET /v1/labeled-data/history</code></td>
            <td>List your past labeled-data uploads</td>
          </tr>
          <tr>
            <td><code>GET /v1/usage</code></td>
            <td>Current month's quota used / limit / plan</td>
          </tr>
          <tr>
            <td><code>GET /v1/predictions/history</code></td>
            <td>Your recent scored applications</td>
          </tr>
          <tr>
            <td><code>GET /v1/plans</code></td>
            <td>List available subscription tiers</td>
          </tr>
          <tr>
            <td><code>POST /v1/tenant/plan</code></td>
            <td>Switch your tenant's subscription tier</td>
          </tr>
        </tbody>
      </table>

      <h2>Examples</h2>

      <p>Score one application:</p>
      <pre>
        <code>{`curl -X POST ${BASE_URL}/v1/predict \\
  -H "Authorization: Bearer <token>" \\
  -H "Content-Type: application/json" \\
  -d '{
    "ID": 1,
    "year": 2023,
    "loan_amount": 250000,
    "property_value": 320000,
    "income": 6000,
    "Credit_Score": 720
  }'`}</code>
      </pre>

      <p>Upload a batch CSV:</p>
      <pre>
        <code>{`curl -X POST ${BASE_URL}/v1/batch/upload \\
  -H "Authorization: Bearer <token>" \\
  -F "file=@applications.csv;type=text/csv"`}</code>
      </pre>

      <p>Check this month's usage:</p>
      <pre>
        <code>{`curl ${BASE_URL}/v1/usage \\
  -H "Authorization: Bearer <token>"`}</code>
      </pre>

      <h2>Quota and rate limits</h2>
      <p>
        Each tenant has a monthly prediction quota tied to its subscription plan — a batch upload consumes
        one unit of quota per row, same as a real-time prediction. Exceeding it returns{" "}
        <code>429 Too Many Requests</code> without charging the request against your quota. See{" "}
        <Link to="/plans">Plans</Link> for current tier limits, or check your own usage anytime at{" "}
        <code>GET /v1/usage</code>.
      </p>
    </div>
  );
}
