import React, { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard";

const DEFAULT_STATUS = {
  step: 1,
  total: 3,
  expected: "bottle",
  object: "none",
  confidence: 0,
  hand: "NOT DETECTED",
  interaction: "NO",
  motion: 0,
  event: "WAITING",
  state: "WAITING",
  camera: true,
  ai: true,
  updated: new Date(),
};

export default function App() {
  const [status, setStatus] = useState(DEFAULT_STATUS);

  const [events] = useState([]);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch("http://localhost:5000/api/status");

        const data = await response.json();

        setStatus({
          ...data,
          camera: true,
          ai: true,
          updated: new Date(),
        });
      } catch (error) {
        console.log("AI server not connected");
      }
    };

    fetchStatus();

    const interval = setInterval(fetchStatus, 500);

    return () => clearInterval(interval);
  }, []);

  return <Dashboard status={status} events={events} />;
}
