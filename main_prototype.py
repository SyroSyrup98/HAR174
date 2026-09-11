from ultralytics import YOLO
import cv2
import mediapipe as mp
# import pyttsx3
from datetime import datetime
import csv
import os

model = YOLO("yolo26n.pt")

# _____________________________________________________________________________________________________________________
# MediaPipe Hands
# _____________________________________________________________________________________________________________________

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# _____________________________________________________________________________________________________________________
# Voice Alert
# _____________________________________________________________________________________________________________________

# engine = pyttsx3.init()

# voices = engine.getProperty("voices")

# # Microsoft Zira
# engine.setProperty("voice", voices[1].id)

# engine.setProperty("rate", 165)
# engine.setProperty("volume", 1.0)


# def voice_alert(message):

#     engine.say(message)
#     engine.runAndWait()


# _____________________________________________________________________________________________________________________
# MEDIAPIPE FULL BODY POSE
# _____________________________________________________________________________________________________________________

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# _____________________________________________________________________________________________________________________
# Event Log
# _____________________________________________________________________________________________________________________

LOG_FILE = "experiment_log.csv"

if not os.path.exists(LOG_FILE):
    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "step",
            "expected",
            "detected",
            "event",
            "status"
        ])
    
def log_event(
    step,
    expected,
    detected,
    event,
    status
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            step,
            expected,
            detected,
            event,
            status
        ])

    print(
        f"[{timestamp}] "
        f"STEP {step} | "
        f"{event} | "
        f"{status}"
    )
# _____________________________________________________________________________________________________________________
# Experiment Sequence
# _____________________________________________________________________________________________________________________

EXPERIMENT_SEQUENCE = [
    "bottle",
    "book",
    "cup"
]

current_step = 0

interaction_frames = {}


camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


frame_count = 0
last_results = []


# _____________________________________________________________________________________________________________________
# Main Loop
# _____________________________________________________________________________________________________________________

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera error")
        break

    frame_count += 1

    # Run YOLO every 3rd frame
    if frame_count % 3 == 0:
        last_results = model(
            frame,
            verbose=False
        )

    results = last_results

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # _____________________________________________________________________________________________________________________
    # MediaPipe hand detection
    # _____________________________________________________________________________________________________________________

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

            hx = int(
                fingertip.x * w
            )

            hy = int(
                fingertip.y * h
            )

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

    # _____________________________________________________________________________________________________________________
    # FULL BODY POSE
    # _____________________________________________________________________________________________________________________

    pose_results = pose.process(rgb)


    if pose_results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        cv2.putText(
            frame,
            "BODY POSE: DETECTED",
            (30, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "BODY POSE: NOT DETECTED",
            (30, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    # _____________________________________________________________________________________________________________________
    # Current experiment step
    # _____________________________________________________________________________________________________________________

    if current_step < len(EXPERIMENT_SEQUENCE):

        expected_object = (
            EXPERIMENT_SEQUENCE[current_step]
        )

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


    # _____________________________________________________________________________________________________________________
    # Object interaction
    # _____________________________________________________________________________________________________________________

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

            if object_name == "person":
                continue


            # _____________________________________________________________________________________________________________________
            # Draw detected object
            # _____________________________________________________________________________________________________________________

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


            # _____________________________________________________________________________________________________________________
            # Hand-object interaction
            # _____________________________________________________________________________________________________________________

            interacting = False

            for hx, hy in hand_points:

                if (
                    x1 <= hx <= x2
                    and
                    y1 <= hy <= y2
                ):

                    interacting = True
                    break


            # _____________________________________________________________________________________________________________________
            # Temporal confirmation
            # _____________________________________________________________________________________________________________________

            if object_name not in interaction_frames:

                interaction_frames[object_name] = 0


            if interacting:

                interaction_frames[object_name] += 1

            else:

                interaction_frames[object_name] = 0


            # _____________________________________________________________________________________________________________________
            # Confirm interaction
            # _____________________________________________________________________________________________________________________

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


                # _____________________________________________________________________________________________________________________
                # Sequence validation
                # _____________________________________________________________________________________________________________________

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
                        current_step + 1,
                        expected_object,
                        object_name,
                        "INTERACTION",
                        "PASS"
                    )
                    
                    # voice_alert(
                    #     f"{object_name} step completed"
                    # )

                    current_step += 1

                    interaction_frames[
                        object_name
                    ] = 0

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
                        current_step + 1,
                        expected_object,
                        object_name,
                        "INTERACTION",
                        "DEVIATION"
                    )

                    
                    # voice_alert(
                    #     f"Procedure deviation. Expected {expected_object}"
                    # )
                    
                    interaction_frames[
                        object_name
                    ] = 0


    cv2.imshow(
        "Code Nova Prototype",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
hands.close()
pose.close()