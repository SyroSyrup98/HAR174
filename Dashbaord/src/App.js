import React, { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard";

export default function App() {
  const [status, setStatus] = useState({
    step: 1,
    total: 3,
    expected: "bottle",
    object: "bottle",
    confidence: 0.94,
    hand: "DETECTED",
    interaction: "YES",
    motion: 18.4,
    event: "PICK_UP",
    state: "PASS",
    camera: true,
    ai: true,
    updated: new Date()
  });

  const [events, setEvents] = useState([
    { time: "15:32:01", event: "PICK_UP", object: "BOTTLE", status: "PASS" },
    { time: "15:32:08", event: "PICK_UP", object: "CUP", status: "DEVIATION" },
    { time: "15:32:14", event: "PICK_UP", object: "BOOK", status: "PASS" }
  ]);

  // Ready for the Python engine:
  // If a Flask/FastAPI backend is added later, expose GET /api/status
  // and replace this demo state with fetch("/api/status").
  useEffect(() => {
    const timer = setInterval(() => {
      setStatus(s => ({ ...s, updated: new Date() }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return <Dashboard status={status} events={events} />;
}