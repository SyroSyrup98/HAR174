import React from "react";

export default function Summary({ status }) {
  const deviation = status.state === "DEVIATION";

  return (
    <div className="summary-grid">
      <Card label="CURRENT STEP" value={`${status.step} / ${status.total}`} sub={`Expected: ${status.expected}`} />
      <Card label="AI CONFIDENCE" value={`${Math.round(status.confidence * 100)}%`} sub="Object detection" />
      <Card label="INTERACTION" value={status.interaction} sub="Hand → object" />
      <Card
        label="LAST EVENT"
        value={status.event}
        sub={deviation ? "Sequence deviation" : "Sequence valid"}
        danger={deviation}
      />
    </div>
  );
}

function Card({ label, value, sub, danger }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong className={danger ? "red-text" : ""}>{value}</strong>
      <small>{sub}</small>
    </div>
  );
}