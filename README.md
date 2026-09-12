# BAS Onboard AI Prototype

Computer-vision prototypes for human activity recognition and object-hand interaction. The project contains a Python AI engine, data collection and model-training scripts, and a separate React dashboard.

> **Project status:** This is a research/demo prototype. The dashboard currently displays demo telemetry and is not yet connected to the Flask APIs. Camera indices, model filenames, and experiment sequences are hard-coded in several scripts.

## What This Project Does

The repository contains three related workflows:

1. **Live object interaction experiment**
   - YOLO detects objects such as bottles, books, and mice.
   - MediaPipe detects hands and body pose.
   - A pickup is inferred when a detected fingertip stays near an object and the object moves.
   - The current server sequence is `bottle -> book -> mouse`.

2. **Human activity recognition**
   - MediaPipe Pose extracts 33 body landmarks per frame.
   - A Random Forest classifier is trained from landmark coordinates.
   - The trained model can classify activity from a video.

3. **Hand gesture collection**
   - MediaPipe Hands extracts 21 hand landmarks.
   - Samples are saved to `gesture_data.csv` for future gesture-model training.
   - The currently listed gestures are `OK`, `STOP`, `YES`, `NO`, `HELLO`, `HELP`, `HOLD`, and `NEXT`.

## Repository Layout

```text
.
|-- server.py                 Flask live AI server and API
|-- main_prototype.py         Standalone live object-interaction experiment
|-- sequence.py               Small configurable sequence validator
|-- collect_gestures.py      Collects hand-landmark samples
|-- train_model.py            Trains the pose activity classifier
|-- predict.py                Runs activity predictions on a video
|-- main.py                   Extracts pose landmarks and movement labels
|-- interaction_test.py       YOLO plus fingertip/object interaction test
|-- hand_test.py              MediaPipe hand tracking test
|-- Phone_Camera.py           Camera preview test
|-- phone_yolo.py             YOLO camera test
|-- yolo.py                   YOLO video test
|-- voices.py                 Lists text-to-speech voices
|-- yolo11n.pt                YOLO model weights used by basic YOLO tests
|-- yolo26n.pt                YOLO model weights used by the live experiment
|-- gesture_data.csv          Hand-landmark dataset
|-- landmarks_clean.csv       Pose dataset used by train_model.py
|-- experiment_log.csv        Event log used by main_prototype.py
|-- data/                     Input videos and other media
|-- activity_model.pkl        Existing serialized activity model, if present
|-- Dashbaord/                React dashboard (directory name is intentionally spelled this way)
```

## Requirements

### Python

- Windows, macOS, or Linux
- Python 3.10+ recommended
- A working webcam, or a video file for offline tests
- A terminal opened at the repository root

The Python scripts use these packages:

```text
opencv-python
mediapipe
ultralytics
flask
flask-cors
pandas
scikit-learn
joblib
pyttsx3
```

Install them with:

```bash
python -m pip install opencv-python mediapipe ultralytics flask flask-cors pandas scikit-learn joblib pyttsx3
```

The YOLO weight files are already present in the repository. If a weight file is missing, Ultralytics may attempt to download it when the model is initialized.

### Node.js dashboard

- Node.js 18 or newer recommended
- npm

The dashboard uses React 18 and Create React App. Its package configuration is in `Dashbaord/package.json`.

## Quick Start: Live AI Server

From the repository root:

```bash
python server.py
```

The server starts on Flask's default address, normally `http://127.0.0.1:5000`.

The server opens camera index `1`, uses a 640x480 frame size, and loads `yolo26n.pt`. Press `q` in the OpenCV window to stop the camera stream. The available endpoints are:

| Endpoint          | Purpose                                                          |
| ----------------- | ---------------------------------------------------------------- |
| `GET /api/health` | Reports server, camera, AI, model, pose, and hand status         |
| `GET /api/status` | Returns the current experiment step and most recent pickup event |
| `GET /video_feed` | Returns the annotated camera stream as MJPEG                     |

Example health check:

```bash
curl http://127.0.0.1:5000/api/health
```

Example status response shape:

```json
{
  "step": 1,
  "total": 3,
  "expected": "bottle",
  "object": "none",
  "confidence": 0,
  "hand": "NOT DETECTED",
  "interaction": "NO",
  "motion": 0,
  "event": "WAITING",
  "state": "WAITING",
  "timestamp": "2026-09-12T12:00:00"
}
```

## Quick Start: React Dashboard

Open a second terminal:

```bash
cd Dashbaord
npm install
npm start
```

Then open the local URL printed by Create React App, normally `http://localhost:3000`.

The current dashboard is a polished demo UI. Its data is local/demo data; it does not currently fetch `/api/status` or `/video_feed`. To make it live, the frontend must be updated to call the Flask server and the two processes must run at the same time.

Create a production build with:

```bash
cd Dashbaord
npm run build
```

## Activity Recognition Workflow

### 1. Collect pose landmarks

Run:

```bash
python main.py
```

Enter an activity label when prompted. The script records MediaPipe's 33 pose landmarks and a movement label to `landmarks.csv`. It opens camera index `1`; press `q` to stop.

The activity rule in this prototype is based on the left shoulder, left hip, and left knee angle:

- angle greater than 140 degrees: `STANDING`
- otherwise: `SITTING`

### 2. Prepare and train the model

`train_model.py` expects a file named `landmarks_clean.csv` with an `activity` column and columns beginning with `landmark_`.

```bash
python train_model.py
```

The script splits the data into training and test sets, trains a 200-tree `RandomForestClassifier`, prints accuracy and a classification report, and writes `activity_model.pkl`.

### 3. Run predictions

`predict.py` loads `activity_model.pkl` and reads `./data/crazy.mp4`:

```bash
python predict.py
```

It extracts the same 99 pose features used during training and uses a 10-prediction majority history to stabilize the displayed label. Change the `VIDEO` constant in `predict.py` to use another input file.

## Hand Gesture Data Collection

Run:

```bash
python collect_gestures.py
```

Choose a gesture number, show one hand to the camera, and press **Space** to save a sample. Press **Q** to quit. Samples are appended to `gesture_data.csv`; the file contains one label plus 63 values (`x`, `y`, and `z` for each of 21 hand landmarks).

This repository currently contains the collector and data, but no gesture training or gesture prediction script.

## Standalone Tests and Utilities

| Script                | Description                                                                        |
| --------------------- | ---------------------------------------------------------------------------------- |
| `main_prototype.py`   | Runs the live object sequence with camera overlays and writes `experiment_log.csv` |
| `interaction_test.py` | Tests YOLO detections and fingertip/object overlap without sequence validation     |
| `hand_test.py`        | Tests MediaPipe hand landmark tracking                                             |
| `Phone_Camera.py`     | Displays camera index `1` without AI processing                                    |
| `phone_yolo.py`       | Runs YOLO on camera index `1` using `yolo11n.pt`                                   |
| `yolo.py`             | Runs YOLO on `./data/video.mp4` using `yolo11n.pt`                                 |
| `sequence.py`         | Demonstrates `PASS`, `DEVIATION`, and `COMPLETE` sequence states                   |
| `voices.py`           | Prints available `pyttsx3` voices                                                  |

Most OpenCV windows close with `q`. `hand_test.py` uses `x` instead.

## Camera Configuration

Camera numbers are hard-coded and differ between scripts:

- `0` is used by the gesture collector, standalone prototype, and interaction test.
- `1` is used by the Flask server and several camera tests.

If the wrong camera opens, change the `cv2.VideoCapture(...)` value in the script you are running. Close other applications that may already be using the camera.

## Troubleshooting

### Camera cannot be opened

Try camera index `0` instead of `1`, verify Windows camera permissions, and close other camera applications. The Flask server raises an error immediately when its camera cannot be opened.

### `AttributeError: module 'cv2' has no attribute 'Videocapture'`

Some scripts currently use `cv2.Videocapture` with a lowercase `c`. The correct OpenCV constructor is:

```python
cv2.VideoCapture(0)
```

Update that spelling in the script before running it. This affects multiple prototype and test scripts.

### Model or dataset not found

Run commands from the repository root so relative paths resolve correctly. Confirm that `yolo26n.pt`, `yolo11n.pt`, `landmarks_clean.csv`, and `activity_model.pkl` exist when the selected workflow requires them.

### Low or unstable detection

Use good lighting, keep the whole hand/body or target object in view, reduce background clutter, and keep the camera stable. YOLO confidence and interaction thresholds are hard-coded prototype values in `server.py` and `main_prototype.py`.

### Dashboard does not show live server data

This is expected in the current version. The React components still use demo telemetry. Start the Flask server separately, then implement frontend requests to `/api/health`, `/api/status`, and `/video_feed` to complete the integration.

## Important Prototype Notes

- The sequence in `server.py` and `main_prototype.py` is `bottle -> book -> mouse`.
- `sequence.py` uses a different demonstration sequence: `bottle -> book -> cup`.
- These sequences are explicitly prototypes and are not the official ISRO experiment sequence.
- The server stores experiment state in process memory. Restarting the server resets the current step.
- The default Flask app enables CORS for all origins; restrict this before deploying outside a local development environment.
- There is no automated test suite or dependency lockfile yet. Reproducibility depends on the installed Python and Node package versions.

## Suggested Development Order

1. Add a Python `requirements.txt` with pinned versions.
2. Replace hard-coded camera indices and input paths with command-line arguments or environment variables.
3. Fix the `VideoCapture` spelling errors across the prototype scripts.
4. Add a shared configuration for the experiment sequence and model path.
5. Connect the React dashboard to the Flask status and video endpoints.
6. Add tests for sequence validation, pickup detection, API responses, and dataset shape validation.
