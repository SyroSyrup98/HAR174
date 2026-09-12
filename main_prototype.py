import joblib
import cv2
import csv
import os
import datetime
from ultralytics import YOLO
import mediapipe as mp

model = YOLO("yolo26n.pt")
gesture_model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

EXPERIMENT_SEQUENCE = [
    "bottle",
    "book",
    "laptop"
]

current_step = 0

interaction_frames = {}
previous_centers = {}
pickup_state = {}

last_results = None
frame_count = 0


# Evengt Logs

LOG_FILE = "experiment_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "step",
            "expected",
            "detected",
            "event",
            "status"
        ])


def log_event(step, expected, detected, event, status):

    timestamp = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            timestamp,
            step,
            expected,
            detected,
            event,
            status
        ])


camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

# Main

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    frame_count += 1

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # EXPECTED STEP
    
    if current_step < len(EXPERIMENT_SEQUENCE):

        expected_object = EXPERIMENT_SEQUENCE[current_step]

    else:

        expected_object = "DONE"


    # HEADER
    
    cv2.rectangle(
        frame,
        (0, 0),
        (width, 100),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        f"STEP: {current_step + 1}/{len(EXPERIMENT_SEQUENCE)}",
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


    # EXPERIMENT COMPLETE
    
    if current_step >= len(EXPERIMENT_SEQUENCE):

        cv2.putText(
            frame,
            "EXPERIMENT COMPLETE",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


    # YOLO Object detection
    
    if frame_count % 3 == 0:

        results = model(
            frame,
            verbose=False,
            conf=0.5
        )

        last_results = results


    # HAND DETECTION + GESTURE RECOGNITION

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    hand_results = hands.process(rgb_frame)

    hand_points = []

    gesture_name = "NONE"
    gesture_confidence = 0.0

    if hand_results.multi_hand_landmarks:

        # Use first detected hand
        hand_landmarks = hand_results.multi_hand_landmarks[0]

        # Draw hand skeleton
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # =========================
        # GESTURE RECOGNITION
        # =========================

        landmarks = []

        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # Predict gesture
        gesture_name = gesture_model.predict(
            [landmarks]
        )[0]

        # Confidence
        probabilities = gesture_model.predict_proba(
            [landmarks]
        )[0]

        gesture_confidence = max(probabilities) * 100

        # =========================
        # INDEX FINGER TIP
        # =========================

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

        # =========================
        # DISPLAY GESTURE
        # =========================

        cv2.putText(
            frame,
            f"HAND SIGN: {gesture_name}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"SIGN CONFIDENCE: {gesture_confidence:.1f}%",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "HAND SIGN: NONE",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


    # BODY POSE
    
    pose_results = pose.process(rgb_frame)

    if pose_results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        cv2.putText(
            frame,
            "POSE: DETECTED",
            (width - 210, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "POSE: NOT DETECTED",
            (width - 250, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )


    # PROCESS OBJECT DETECTIONS
    
    if last_results is not None:

        for result in last_results:

            boxes = result.boxes

            for box in boxes:

                confidence = float(box.conf[0])

                if confidence < 0.5:
                    continue

                class_id = int(box.cls[0])

                object_name = model.names[class_id]

                if object_name == "person":
                    continue


                # BOUNDING BOX
                
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

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


                # OBJECT CENTER
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (255, 255, 255),
                    -1
                )


                # OBJECT MOVEMENT
                
                motion = 0

                if object_name in previous_centers:

                    prev_cx, prev_cy = previous_centers[
                        object_name
                    ]

                    motion = (
                        (cx - prev_cx) ** 2 +
                        (cy - prev_cy) ** 2
                    ) ** 0.5

                previous_centers[object_name] = (cx, cy)


                cv2.putText(
                    frame,
                    f"MOTION: {motion:.1f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )


                # INTERACTION
                
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


                # SHOW INTERACTION

                if interaction_frames[object_name] >= 3:

                    cv2.putText(
                        frame,
                        f"HAND INTERACTION: {object_name}",
                        (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 0, 255),
                        2
                    )


                # PICK-UP DETECTION
                
                if object_name not in pickup_state:

                    pickup_state[object_name] = False


                pickup_detected = (
                    confidence >= 0.5
                    and interaction_frames[object_name] >= 3
                    and motion > 5
                    and not pickup_state[object_name]
                )


                if pickup_detected:

                    pickup_state[object_name] = True


                    # PICK UP DISPLAY

                    cv2.putText(
                        frame,
                        f"PICK UP: {object_name.upper()}",
                        (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2
                    )


                    # CORRECT OBJECT

                    if expected_object == object_name:

                        cv2.putText(
                            frame,
                            "STATUS: PASS",
                            (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        log_event(
                            current_step + 1,
                            expected_object,
                            object_name,
                            "PICK_UP",
                            "PASS"
                        )

                        current_step += 1

                    # WRONG OBJECT
                    
                    else:

                        cv2.putText(
                            frame,
                            "STATUS: DEVIATION",
                            (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        log_event(
                            current_step + 1,
                            expected_object,
                            object_name,
                            "PICK_UP",
                            "DEVIATION"
                        )

                        # Allow the same object to be tried again
                        pickup_state[object_name] = False


                    # RESET INTERACTION
                    
                    interaction_frames[object_name] = 0

    # NEXT STEP DISPLAY

    if current_step < len(EXPERIMENT_SEQUENCE):

        next_object = EXPERIMENT_SEQUENCE[current_step]

        cv2.putText(
            frame,
            f"NEXT: PICK UP {next_object.upper()}",
            (20, height - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "NEXT: NONE",
            (20, height - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


    # CONTROLS

    cv2.putText(
        frame,
        "Press Q to quit",
        (width - 180, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1
    )


    # # DISPLAY
    
    # cv2.imshow(
    #     "Experiment Window",
    #     frame
    # )

    # key = cv2.waitKey(1) & 0xFF

    # if key == ord("q"):
    #     break
    
    # Show video
    display_frame = cv2.resize(frame, (1000, 800))
    
    cv2.imshow(
    "Landmark Extraction",
    display_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
hands.close()
pose.close()
cv2.destroyAllWindows()
