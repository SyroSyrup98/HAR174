import React from "react";
import TopBar from "./TopBar";
import Sidebar from "./Sidebar";
import Summary from "./Summary";
import EventLog from "./EventLog";
import CameraFeed from "./Camerafeed";

export default function Dashboard({ status, events }) {
  return (
    <div className="app-shell">
      <TopBar status={status} />

      <div className="body-layout">
        <Sidebar />

        <main className="main-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">ONBOARD EXPERIMENT MONITORING</div>

              <h1>Human Activity Recognition</h1>

              <p>AI-assisted procedure validation for payload experiments</p>
            </div>

            <div className="live-pill">
              <span /> LIVE MONITORING
            </div>
          </div>

          <Summary status={status} />

          {/* =====================================================
              MAIN GRID
          ====================================================== */}

          <div className="grid-main">
            {/* ===================================================
                CAMERA
            ==================================================== */}

            <section className="panel camera-panel">
              <div className="panel-head">
                <div>
                  <span className="panel-kicker">CAMERA FEED</span>

                  <h2>Payload Camera 01</h2>
                </div>

                <span className="status-badge green">ONLINE</span>
              </div>

              <div className="camera-frame">
                <CameraFeed />
              </div>

              {/* Detection information */}

              <div className="detection-row">
                <div>
                  <span>DETECTED OBJECT</span>

                  <strong>{status.object.toUpperCase()}</strong>
                </div>

                <div>
                  <span>CONFIDENCE</span>

                  <strong>{Math.round(status.confidence * 100)}%</strong>
                </div>

                <div>
                  <span>HAND</span>

                  <strong className="green-text">{status.hand}</strong>
                </div>

                <div>
                  <span>MOTION</span>

                  <strong>{status.motion.toFixed(1)} px</strong>
                </div>
              </div>
            </section>

            {/* ===================================================
                PROCEDURE
            ==================================================== */}

            <section className="panel procedure-panel">
              <div className="panel-head">
                <div>
                  <span className="panel-kicker">EXPERIMENT PROCEDURE</span>

                  <h2>Sequence Validation</h2>
                </div>

                <span
                  className={`status-badge ${
                    status.state === "PASS" ? "green" : "red"
                  }`}
                >
                  {status.state}
                </span>
              </div>

              <div className="step-count">
                <strong>{status.step}</strong>

                <span>/ {status.total} STEPS</span>
              </div>

              <div className="progress-track">
                <div
                  style={{
                    width: `${((status.step - 1) / status.total) * 100}%`,
                  }}
                />
              </div>

              <div className="steps">
                {["bottle", "book", "cup"].map((item, i) => {
                  const done = i < status.step - 1;

                  const current = i === status.step - 1;

                  return (
                    <div
                      className={`procedure-step ${done ? "done" : ""} ${
                        current ? "current" : ""
                      }`}
                      key={item}
                    >
                      <div className="step-dot">{done ? "✓" : i + 1}</div>

                      <div>
                        <span>STEP {i + 1}</span>

                        <strong>{item.toUpperCase()}</strong>
                      </div>

                      {current && <em>CURRENT</em>}
                    </div>
                  );
                })}
              </div>

              <div className="next-action">
                <span>NEXT ACTION</span>

                <strong>PICK UP {status.expected.toUpperCase()}</strong>
              </div>
            </section>
          </div>

          {/* =====================================================
              BOTTOM GRID
          ====================================================== */}

          <div className="grid-bottom">
            {/* Event log */}

            <section className="panel event-panel">
              <div className="panel-head">
                <div>
                  <span className="panel-kicker">ACTIVITY STREAM</span>

                  <h2>Event Log</h2>
                </div>

                <span className="muted">CSV: experiment_log.csv</span>
              </div>

              <EventLog events={events} />
            </section>

            {/* System health */}

            <section className="panel system-panel">
              <div className="panel-head">
                <div>
                  <span className="panel-kicker">SYSTEM HEALTH</span>

                  <h2>AI Engine</h2>
                </div>
              </div>

              <Health
                name="Camera Input"
                ok={status.camera}
                detail="DroidCam / Browser"
              />

              <Health name="Object Detection" ok={status.ai} detail="YOLO26n" />

              <Health name="Hand & Pose" ok={status.ai} detail="MediaPipe" />

              <Health
                name="Sequence Validator"
                ok={status.ai}
                detail="Procedure state machine"
              />
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}

/* ===============================================================
   HEALTH COMPONENT
================================================================ */

function Health({ name, ok, detail }) {
  return (
    <div className="health-row">
      <div className="health-dot">{ok ? "✓" : "!"}</div>

      <div>
        <strong>{name}</strong>

        <span>{detail}</span>
      </div>

      <b>{ok ? "ONLINE" : "ERROR"}</b>
    </div>
  );
}
