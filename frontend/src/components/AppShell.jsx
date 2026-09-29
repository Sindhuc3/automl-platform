import { NavLink, Outlet, useLocation } from "react-router-dom";

function AppShell() {
  const location = useLocation();

  const pageTitle =
    location.pathname === "/"
      ? "Home"
      : location.pathname.startsWith("/datasets")
        ? "Datasets"
        : location.pathname.startsWith("/runs")
          ? "Runs"
          : "Settings";

  return (
    <div className="studio-shell">
      <aside className="studio-sidebar">
        <div className="studio-brand">
          <span className="brand-primary">AutoML</span>
          <span className="brand-secondary">Studio</span>
        </div>

        <nav className="studio-nav" aria-label="Primary navigation">
          <NavLink to="/" end className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            Home
          </NavLink>
          <NavLink to="/datasets" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            Datasets
          </NavLink>
          <NavLink to="/runs" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            Runs
          </NavLink>
        </nav>

        <div className="sidebar-bottom">
          <NavLink to="/settings" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            Settings
          </NavLink>
        </div>
      </aside>

      <div className="studio-main">
        <header className="studio-header">
          <div className="header-page-title">{pageTitle}</div>
          <div className="guest-status">Guest</div>
        </header>

        <main className="studio-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default AppShell;
