const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
// agent/ and rag_harness/ are separate services/deployments — each gets its
// own browser-facing base URL, same VITE_*_BASE_URL-baked-at-build-time
// pattern as VITE_API_BASE_URL.
export const AGENT_BASE_URL = import.meta.env.VITE_AGENT_BASE_URL || "http://localhost:8003";
export const RAG_HARNESS_BASE_URL = import.meta.env.VITE_RAG_HARNESS_BASE_URL || "http://localhost:8001";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function getToken(): string | null {
  return localStorage.getItem("token");
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit & { baseUrl?: string } = {},
): Promise<T> {
  const { baseUrl = BASE_URL, ...fetchOptions } = options;
  const token = getToken();
  const headers: Record<string, string> = {
    ...(fetchOptions.body && !(fetchOptions.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
    ...(fetchOptions.headers as Record<string, string> | undefined),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // Every response here is scoped to whichever tenant's token is attached —
  // the browser's HTTP cache keys on URL, not on the Authorization header,
  // so without this a request made as one tenant could be served back to a
  // different tenant hitting the same path later in the same session.
  const res = await fetch(`${baseUrl}${path}`, { ...fetchOptions, headers, cache: "no-store" });

  // A 401 from /auth/login or /auth/register just means "wrong credentials" —
  // there's no session to have expired yet. Only treat a 401 from an
  // authenticated call as session expiry (bad/expired JWT on an existing
  // session), so login/register can show the backend's real error message
  // instead of forcing a redirect back to the page the user is already on.
  const isAuthEndpoint = path === "/auth/login" || path === "/auth/register";
  if (res.status === 401 && !isAuthEndpoint) {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    if (window.location.pathname !== "/login") {
      window.location.href = "/login";
    }
    throw new ApiError(401, "Session expired");
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        // FastAPI/Pydantic request-validation errors: detail is a list of
        // {loc, msg, type} objects, not a string — Error's own message
        // coercion would otherwise render this as the useless "[object
        // Object]" wherever a page just does {error} in JSX.
        detail =
          body.detail
            .map((d: { msg?: string }) => d.msg)
            .filter(Boolean)
            .join("; ") || JSON.stringify(body.detail);
      } else if (body.detail) {
        detail = JSON.stringify(body.detail);
      } else {
        detail = JSON.stringify(body);
      }
    } catch {
      // response body wasn't JSON — fall back to statusText
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) {
    return undefined as T;
  }
  return res.json();
}

export function apiGet<T>(path: string, baseUrl?: string): Promise<T> {
  return apiFetch<T>(path, { method: "GET", baseUrl });
}

export function apiPost<T>(path: string, body?: unknown, baseUrl?: string): Promise<T> {
  return apiFetch<T>(path, {
    method: "POST",
    body: body instanceof FormData ? body : JSON.stringify(body ?? {}),
    baseUrl,
  });
}

export function apiUpload<T>(path: string, file: File, baseUrl?: string): Promise<T> {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch<T>(path, { method: "POST", body: formData, baseUrl });
}
