import cv2
import mediapipe as mp
import csv
import os

GESTURES = [
    "OK",
    "STOP",
    "YES",
    "NO",
    "HELLO",
    "HELP",
    "HOLD",
    "NEXT"
]

DATA_FILE = "gesture_data.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("\nGesture Collector")
print("-----------------")
for i, gesture in enumerate(GESTURES):
    print(f"{i + 1}. {gesture}")

choice = int(input("\nEnter gesture number: "))
gesture_name = GESTURES[choice - 1]

print(f"\nCollecting: {gesture_name}")
print("Press SPACE to save a sample")
print("Press Q to quit\n")

file_exists = os.path.exists(DATA_FILE)

with open(DATA_FILE, "a", newline="") as f:

    writer = csv.writer(f)

    if not file_exists:
        header = ["label"]

        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]

        writer.writerow(header)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for landmark in hand.landmark:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            cv2.putText(
                frame,
                f"GESTURE: {gesture_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "SPACE = SAVE SAMPLE",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):

                writer.writerow(
                    [gesture_name] + landmarks
                )

                print(f"Saved {gesture_name}")

        else:

            cv2.putText(
                frame,
                "Show your hand",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            key = cv2.waitKey(1) & 0xFF

        cv2.imshow("Gesture Collector", frame)

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
hands.close()