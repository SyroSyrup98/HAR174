from ultralytics import YOLO
import cv2
import mediapipe as mp
import math

# ---------- YOLO ----------
model = YOLO("yolo26n.pt")

# ---------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Use the same camera index that worked for your phone
cap = cv2.VideoCapture(0)

interaction_frames = {}

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    # ---------- YOLO ----------
    results = model(frame, verbose=False)

    # ---------- MediaPipe ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)

    # Store hand landmark positions
    hand_points = []

    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            # Index fingertip = landmark 8
            fingertip = hand.landmark[8]

            x = int(fingertip.x * w)
            y = int(fingertip.y * h)

            hand_points.append((x, y))

            cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)

    # ---------- Check hand/object interaction ----------
    for result in results:

        boxes = result.boxes

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            # Only look at useful objects for our prototype
            useful_objects = [
                "bottle",
                "cup",
                "book",
                "laptop",
                "cell phone"
            ]

            if object_name not in useful_objects:
                continue

            # Draw object box
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

            # Check whether fingertip is inside object box
            interacting = False

            for hx, hy in hand_points:

                if x1 <= hx <= x2 and y1 <= hy <= y2:
                    interacting = True

            if object_name not in interaction_frames:
                interaction_frames[object_name] = 0
                
            if interacting:
                interaction_frames[object_name] += 1
            else:
                interaction_frames[object_name] = 0
                
            if interaction_frames[object_name] >= 3:
                
                cv2.putText(
                    frame,
                    f"INTERACTION CONFIRMED: {object_name}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

    cv2.imshow("YOLO + Hand Interaction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()