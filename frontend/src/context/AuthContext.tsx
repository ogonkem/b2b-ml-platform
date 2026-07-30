import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import { apiPost } from "../api/client";

export interface User {
  tenant_id: string;
  role: "user" | "admin";
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  tenant_id: string;
  role: "user" | "admin";
  invite_code?: string | null;
}

export interface RegisterParams {
  email: string;
  password: string;
  tenant_name?: string;
  invite_code?: string;
}

interface AuthContextValue {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  register: (params: RegisterParams) => Promise<AuthResponse>;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function loadStoredUser(): User | null {
  const raw = localStorage.getItem("user");
  return raw ? (JSON.parse(raw) as User) : null;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(localStorage.getItem("token"));
  const [user, setUser] = useState<User | null>(loadStoredUser());

  const persist = useCallback((resp: AuthResponse) => {
    localStorage.setItem("token", resp.access_token);
    const nextUser: User = { tenant_id: resp.tenant_id, role: resp.role };
    localStorage.setItem("user", JSON.stringify(nextUser));
    setToken(resp.access_token);
    setUser(nextUser);
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      const resp = await apiPost<AuthResponse>("/auth/login", { email, password });
      persist(resp);
    },
    [persist],
  );

  const register = useCallback(
    async (params: RegisterParams) => {
      const resp = await apiPost<AuthResponse>("/auth/register", params);
      persist(resp);
      return resp;
    },
    [persist],
  );

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setToken(null);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, login, register, logout, isAuthenticated: !!token }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
