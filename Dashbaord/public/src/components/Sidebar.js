import React from "react";

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="nav-label">MISSION CONTROL</div>
      <nav>
        <a className="active"><span>▦</span> Overview</a>
        <a><span>◉</span> Camera Feed</a>
        <a><span>✓</span> Procedures</a>
        <a><span>≡</span> Event Logs</a>
      </nav>

      <div className="sidebar-bottom">
        <div className="connection">
          <i />
          <div>
            <strong>EDGE MODE</strong>
            <span>LOCAL PROCESSING</span>
          </div>
        </div>
        <div className="version">BAS AI • v1.0 DEMO</div>
      </div>
    </aside>
  );
}