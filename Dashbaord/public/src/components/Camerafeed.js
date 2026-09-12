import React from "react";

export default function CameraFeed() {
  return (
    <div className="camera-live">

      <img
        src="http://localhost:5000/video_feed"
        alt="Live AI Camera"
      />

      <div className="camera-overlay">
        <span className="rec">● REC</span>
        <span>LIVE AI</span>
      </div>

    </div>
  );
}