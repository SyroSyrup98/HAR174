from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import datetime
from ultralytics import YOLO
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# ============================================================
# MODEL SETUP
# ============================================================

model = YOLO("yolo26n.pt")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


# ============================================================
# EXPERIMENT
# ============================================================

EXPERIMENT_SEQUENCE = [
    "bottle",
    "book",
    "mouse"
]

current_step = 0

interaction_frames = {}
previous_centers = {}
pickup_state = {}

last_status = {
    "step": 1,
    "total": len(EXPERIMENT_SEQUENCE),
    "expected": EXPERIMENT_SEQUENCE[0],
    "object": "none",
    "confidence": 0,
    "hand": "NOT DETECTED",
    "interaction": "NO",
    "motion": 0,
    "event": "WAITING",
    "state": "WAITING"
}


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("Could not open camera")


# ============================================================
# AI FRAME PROCESSING
# ============================================================

def generate_frames():

    global current_step
    global last_status

    frame_count = 0
    last_results = None

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame_count += 1

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # ----------------------------------------------------
        # EXPECTED OBJECT
        # ----------------------------------------------------

        if current_step < len(EXPERIMENT_SEQUENCE):
            expected_object = EXPERIMENT_SEQUENCE[current_step]
        else:
            expected_object = "DONE"


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        if frame_count % 3 == 0:

            results = model(
                frame,
                verbose=False,
                conf=0.5
            )

            last_results = results


        # ----------------------------------------------------
        # HAND DETECTION
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        hand_results = hands.process(rgb_frame)

        hand_points = []

        if hand_results.multi_hand_landmarks:

            for hand_landmarks in hand_results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                fingertip = hand_landmarks.landmark[8]

                hx = int(fingertip.x * width)
                hy = int(fingertip.y * height)

                hand_points.append((hx, hy))

                cv2.circle(
                    frame,
                    (hx, hy),
                    7,
                    (255, 0, 255),
                    -1
                )


        # ----------------------------------------------------
        # POSE
        # ----------------------------------------------------

        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:

            mp_draw.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )


        # ----------------------------------------------------
        # PROCESS OBJECTS
        # ----------------------------------------------------

        detected_object = "none"
        detected_confidence = 0
        detected_motion = 0
        interaction = False

        if last_results is not None:

            for result in last_results:

                for box in result.boxes:

                    confidence = float(box.conf[0])

                    if confidence < 0.60:
                        continue

                    class_id = int(box.cls[0])

                    object_name = model.names[class_id]

                    if object_name == "person":
                        continue


                    # Bounding box

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )


                    # Center

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2


                    # Movement

                    motion = 0

                    if object_name in previous_centers:

                        prev_cx, prev_cy = previous_centers[
                            object_name
                        ]

                        motion = (
                            (cx - prev_cx) ** 2 +
                            (cy - prev_cy) ** 2
                        ) ** 0.5

                    previous_centers[object_name] = (
                        cx,
                        cy
                    )


                    # Hand interaction

                    interacting = False

                    for hx, hy in hand_points:

                        if (
                            x1 - 40 <= hx <= x2 + 40
                            and
                            y1 - 40 <= hy <= y2 + 40
                        ):

                            interacting = True

                            cv2.line(
                                frame,
                                (hx, hy),
                                (cx, cy),
                                (255, 0, 255),
                                2
                            )

                            break


                    if object_name not in interaction_frames:
                        interaction_frames[object_name] = 0

                    if interacting:
                        interaction_frames[object_name] += 1
                    else:
                        interaction_frames[object_name] = max(
                            0,
                            interaction_frames[object_name] - 1
                        )


                    # ------------------------------------------------
                    # DRAW OBJECT
                    # ------------------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 200, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{object_name} {confidence:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 200, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"MOTION: {motion:.1f}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )


                    # ------------------------------------------------
                    # PICK UP
                    # ------------------------------------------------

                    if object_name not in pickup_state:
                        pickup_state[object_name] = False


                    pickup_detected = (
                        confidence >= 0.70
                        and
                        interaction_frames[object_name] >= 5
                        and
                        motion > 15
                        and
                        not pickup_state[object_name]
                    )


                    if pickup_detected:

                        pickup_state[object_name] = True

                        detected_object = object_name
                        detected_confidence = confidence
                        detected_motion = motion
                        interaction = True


                        cv2.putText(
                            frame,
                            f"PICK UP: {object_name.upper()}",
                            (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 0),
                            2
                        )


                        if expected_object == object_name:

                            status = "PASS"

                            cv2.putText(
                                frame,
                                "STATUS: PASS",
                                (20, 195),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2
                            )

                            current_step += 1

                        else:

                            status = "DEVIATION"

                            cv2.putText(
                                frame,
                                "STATUS: DEVIATION",
                                (20, 195),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )

                            pickup_state[object_name] = False


                        interaction_frames[object_name] = 0


                        # Update expected object

                        if current_step < len(EXPERIMENT_SEQUENCE):
                            new_expected = EXPERIMENT_SEQUENCE[current_step]
                        else:
                            new_expected = "DONE"


                        last_status = {
                            "step": min(
                                current_step + 1,
                                len(EXPERIMENT_SEQUENCE)
                            ),
                            "total": len(EXPERIMENT_SEQUENCE),
                            "expected": new_expected,
                            "object": object_name,
                            "confidence": confidence,
                            "hand": "DETECTED",
                            "interaction": "YES",
                            "motion": motion,
                            "event": "PICK_UP",
                            "state": status
                        }


        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 90),
            (35, 35, 35),
            -1
        )

        cv2.putText(
            frame,
            f"STEP {min(current_step + 1, len(EXPERIMENT_SEQUENCE))}/{len(EXPERIMENT_SEQUENCE)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"EXPECTED: {expected_object.upper()}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        if current_step >= len(EXPERIMENT_SEQUENCE):

            cv2.putText(
                frame,
                "EXPERIMENT COMPLETE",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # NEXT ACTION
        # ----------------------------------------------------

        if current_step < len(EXPERIMENT_SEQUENCE):

            cv2.putText(
                frame,
                f"NEXT: PICK UP {EXPERIMENT_SEQUENCE[current_step].upper()}",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )


        # ----------------------------------------------------
        # ENCODE FRAME
        # ----------------------------------------------------

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# VIDEO API
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# STATUS API
# ============================================================

@app.route("/api/status")
def status():

    return jsonify({
        **last_status,
        "timestamp": datetime.datetime.now().isoformat()
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({
        "status": "online",
        "camera": camera.isOpened(),
        "ai": True,
        "model": "YOLO26n",
        "pose": "MediaPipe",
        "hands": "MediaPipe"
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" BAS ONBOARD AI SERVER")
    print("==========================================")
    print()
    print("Camera: ONLINE")
    print("YOLO26n: ONLINE")
    print("MediaPipe: ONLINE")
    print()
    print("Video:")
    print("http://localhost:5000/video_feed")
    print()
    print("Status:")
    print("http://localhost:5000/api/status")
    print()
    print("==========================================")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )