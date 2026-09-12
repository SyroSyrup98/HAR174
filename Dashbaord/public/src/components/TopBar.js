import React from "react";

export default function TopBar({ status }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">B</div>
        <div>
          <strong>BAS</strong>
          <span>ONBOARD AI SYSTEM</span>
        </div>
      </div>

      <div className="mission">
        <span>MISSION</span>
        <strong>SIH • BAS EXPERIMENT 01</strong>
      </div>

      <div className="top-status">
        <span className="system-online"><i /> SYSTEM ONLINE</span>
        <div className="utc">
          <span>LOCAL</span>
          <strong>{status.updated.toLocaleTimeString()}</strong>
        </div>
      </div>
    </header>
  );
}