import React from "react";

export default function EventLog({ events }) {
  return (
    <div className="event-table">
      <div className="event-head">
        <span>TIME</span>
        <span>EVENT</span>
        <span>OBJECT</span>
        <span>RESULT</span>
      </div>
      {events.map((e, i) => (
        <div className="event-row" key={i}>
          <span className="mono">{e.time}</span>
          <span className="event-name">{e.event}</span>
          <span>{e.object}</span>
          <span className={`result ${e.status === "PASS" ? "pass" : "deviation"}`}>
            {e.status}
          </span>
        </div>
      ))}
    </div>
  );
}