# BAS AI React Dashboard

React 18 dashboard for the SIH BAS Human Activity Recognition prototype.

## Run

```bash
npm install
npm start
```

Then open the local React URL shown by Create React App.

## Current state

The UI is a polished demo dashboard with demo telemetry.

The Python AI engine remains separate. The intended integration endpoint is:

- `GET /api/status` — current experiment state
- `GET /video_feed` — MJPEG camera stream

The dashboard can then replace the demo state and camera placeholder with the live Python engine.

## Important

The sequence `bottle -> book -> cup` is only a temporary prototype sequence and is not the official ISRO experiment sequence.
