from ultralytics import YOLO
import cv2
import mediapipe as mp
import pyttsx3
from datetime import datetime

# =========================
# YOLO
# =========================

model = YOLO("yolo26n.pt")


# =========================
# MediaPipe Hands
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# # =========================
# # Voice Alert
# # =========================

# engine = pyttsx3.init()

# voices = engine.getProperty("voices")

# # Microsoft Zira
# engine.setProperty("voice", voices[1].id)

# engine.setProperty("rate", 165)
# engine.setProperty("volume", 1.0)


# def voice_alert(message):

#     engine.say(message)
#     engine.runAndWait()


# =========================
# Event Log
# =========================

log_file = open(
    "experiment_log.txt",
    "a",
    encoding="utf-8"
)


def log_event(event):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    log_file.write(
        f"{timestamp} | {event}\n"
    )

    log_file.flush()

    print(
        f"[{timestamp}] {event}"
    )

# =========================
# Experiment Sequence
# =========================

EXPERIMENT_SEQUENCE = [
    "bottle",
    "book",
    "cup"
]

current_step = 0

# Number of consecutive frames required
# to confirm an interaction
interaction_frames = {}


# =========================
# Camera
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# =========================
# YOLO frame skipping
# =========================

frame_count = 0
last_results = []


# =========================
# Main Loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break


    # =========================
    # YOLO object detection
    # =========================

    frame_count += 1

    # Run YOLO every 3rd frame
    if frame_count % 3 == 0:
        last_results = model(
            frame,
            verbose=False
        )

    results = last_results


    # =========================
    # MediaPipe hand detection
    # =========================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    hand_results = hands.process(rgb)

    hand_points = []

    if hand_results.multi_hand_landmarks:

        for hand in hand_results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            # Index fingertip
            fingertip = hand.landmark[8]

            hx = int(fingertip.x * w)
            hy = int(fingertip.y * h)

            hand_points.append(
                (hx, hy)
            )

            cv2.circle(
                frame,
                (hx, hy),
                8,
                (255, 0, 255),
                -1
            )


    # =========================
    # Current experiment step
    # =========================

    if current_step < len(EXPERIMENT_SEQUENCE):

        expected_object = EXPERIMENT_SEQUENCE[current_step]

        cv2.putText(
            frame,
            f"STEP {current_step + 1}/{len(EXPERIMENT_SEQUENCE)}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"EXPECTED: {expected_object}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    else:

        expected_object = None

        cv2.putText(
            frame,
            "EXPERIMENT COMPLETE",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # =========================
    # Check object interaction
    # =========================

    for result in results:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            # Ignore low-confidence detections
            if confidence < 0.6:
                continue


            # Object coordinates

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Object class

            class_id = int(
                box.cls[0]
            )

            object_name = model.names[class_id]


            # Completely ignore person
            if object_name == "person":
                continue


            # =========================
            # Draw detected object
            # =========================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{object_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # =========================
            # Check hand-object interaction
            # =========================

            interacting = False

            for hx, hy in hand_points:

                if (
                    x1 <= hx <= x2
                    and
                    y1 <= hy <= y2
                ):

                    interacting = True
                    break


            # =========================
            # Temporal confirmation
            # =========================

            if object_name not in interaction_frames:

                interaction_frames[object_name] = 0


            if interacting:

                interaction_frames[object_name] += 1

            else:

                interaction_frames[object_name] = 0


            # =========================
            # Confirm interaction
            # =========================

            if interaction_frames[object_name] >= 3:

                cv2.putText(
                    frame,
                    f"INTERACTION: {object_name}",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )


                # =========================
                # Sequence validation
                # =========================

                if expected_object == object_name:

                    cv2.putText(
                        frame,
                        "STATUS: PASS",
                        (30, 145),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    log_event(
                        f"STEP {current_step + 1} PASS | {object_name} interacted"
                    )
                    
                    # voice_alert(
                    #     f"{object_name} step completed"
                    # )

                    current_step += 1

                    # Reset interaction counter
                    # so the same interaction
                    # doesn't repeatedly advance

                    interaction_frames[object_name] = 0

                else:

                    cv2.putText(
                        frame,
                        "STATUS: DEVIATION",
                        (30, 145),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    log_event(
                        f"DEVIATION | Expected {expected_object}, detected {object_name}"
                    )
                    
                    voice_alert(
                        f"Procedure deviation. Expected {expected_object}"
                    )
                    
                    interaction_frames[object_name] = 0


    # =========================
    # Display
    # =========================

    cv2.imshow(
        "ASTRA-HAR Prototype",
        frame
    )


    # =========================
    # Quit
    # =========================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()
hands.close()
log_file.close()