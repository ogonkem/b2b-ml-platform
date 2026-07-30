import { useState, type ReactNode } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import logoMark from "../assets/logo-mark.png";

const COLLAPSE_KEY = "sidebar-collapsed";

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="nav-icon"
    >
      {children}
    </svg>
  );
}

const icons = {
  dashboard: (
    <Icon>
      <rect x="3.5" y="3.5" width="7.5" height="7.5" rx="1.5" />
      <rect x="13" y="3.5" width="7.5" height="4.5" rx="1.5" />
      <rect x="13" y="10.5" width="7.5" height="10" rx="1.5" />
      <rect x="3.5" y="13.5" width="7.5" height="7" rx="1.5" />
    </Icon>
  ),
  predict: (
    <Icon>
      <path d="M12 2 4 13.5h6.5L11 22l8.5-11.5H13z" />
    </Icon>
  ),
  batch: (
    <Icon>
      <path d="M12 15V4M7.5 8.5 12 4l4.5 4.5" />
      <path d="M4.5 15v3a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-3" />
    </Icon>
  ),
  labeled: (
    <Icon>
      <path d="M11.2 3.5h5.3a2 2 0 0 1 2 2v5.3a2 2 0 0 1-.6 1.4l-8 8a2 2 0 0 1-2.8 0l-5.3-5.3a2 2 0 0 1 0-2.8l8-8a2 2 0 0 1 1.4-.6Z" />
      <circle cx="15" cy="9" r="1.4" />
    </Icon>
  ),
  usage: (
    <Icon>
      <path d="M4 20V10M11 20V4M18 20v-7" />
    </Icon>
  ),
  account: (
    <Icon>
      <circle cx="12" cy="8" r="3.5" />
      <path d="M4.5 20c1.2-4 4-6 7.5-6s6.3 2 7.5 6" />
    </Icon>
  ),
  admin: (
    <Icon>
      <path d="M12 3 5 5.5v5.2c0 4.6 3 8.4 7 9.8 4-1.4 7-5.2 7-9.8V5.5Z" />
    </Icon>
  ),
  collapse: (
    <Icon>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2" />
      <path d="M9.5 4.5v15" />
      <path d="M14 10l-2 2 2 2" />
    </Icon>
  ),
  expand: (
    <Icon>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2" />
      <path d="M9.5 4.5v15" />
      <path d="M12.5 10l2 2-2 2" />
    </Icon>
  ),
  logout: (
    <Icon>
      <path d="M9 21H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3" />
      <path d="M16 17l5-5-5-5M21 12H9" />
    </Icon>
  ),
  sun: (
    <Icon>
      <circle cx="12" cy="12" r="4.2" />
      <path d="M12 2.5v2.3M12 19.2v2.3M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6M2.5 12h2.3M19.2 12h2.3M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6" />
    </Icon>
  ),
  moon: (
    <Icon>
      <path d="M20.2 14.6A8.5 8.5 0 1 1 9.4 3.8a7 7 0 0 0 10.8 10.8Z" />
    </Icon>
  ),
};

const NAV_ITEMS: { to: string; label: string; icon: keyof typeof icons }[] = [
  { to: "/dashboard", label: "Dashboard", icon: "dashboard" },
  { to: "/predict", label: "Predict", icon: "predict" },
  { to: "/batch", label: "Batch Upload", icon: "batch" },
  { to: "/labeled-data", label: "Labeled Data", icon: "labeled" },
  { to: "/usage", label: "Usage", icon: "usage" },
  { to: "/account", label: "Account", icon: "account" },
];

export function Layout() {
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem(COLLAPSE_KEY) === "1");

  function toggleCollapsed() {
    setCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem(COLLAPSE_KEY, next ? "1" : "0");
      return next;
    });
  }

  return (
    <div className="app-shell">
      <nav className={`sidebar${collapsed ? " collapsed" : ""}`}>
        <div className="sidebar-top">
          <div className="brand">
            <img src={logoMark} alt="" className="brand-mark" />
            {!collapsed && (
              <span className="brand-text">
                <span className="brand-sela">sela</span>
                <span className="brand-stone">stone</span>
              </span>
            )}
          </div>
          <button
            type="button"
            className="collapse-btn"
            onClick={toggleCollapsed}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {icons[collapsed ? "expand" : "collapse"]}
          </button>
        </div>

        {NAV_ITEMS.map((item) => (
          <NavLink key={item.to} to={item.to} title={collapsed ? item.label : undefined}>
            {icons[item.icon]}
            {!collapsed && <span className="label">{item.label}</span>}
          </NavLink>
        ))}
        {user?.role === "admin" && (
          <NavLink to="/admin" title={collapsed ? "Admin" : undefined}>
            {icons.admin}
            {!collapsed && <span className="label">Admin</span>}
          </NavLink>
        )}

        <div className="sidebar-spacer" />
        {!collapsed && <div className="tenant-tag">{user?.tenant_id}</div>}
        <button
          className="logout-btn"
          onClick={toggleTheme}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {icons[theme === "dark" ? "sun" : "moon"]}
          {!collapsed && <span className="label">{theme === "dark" ? "Light mode" : "Dark mode"}</span>}
        </button>
        <button
          className="logout-btn"
          onClick={logout}
          title={collapsed ? "Log out" : undefined}
        >
          {icons.logout}
          {!collapsed && <span className="label">Log out</span>}
        </button>
      </nav>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
