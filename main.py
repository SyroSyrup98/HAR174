import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import csv
import math

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(
            c.y - b.y,
            c.x - b.x
        )
        -
        math.atan2(
            a.y - b.y,
            a.x - b.x
        )
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

# Ask what activity this video contains
ACTIVITY = input("Enter activity: ").strip().upper()

# Open video
# camera = cv2.Videocapture(0)
camera = cv2.Videocapture(1)

# Create CSV file

file_exists = os.path.exists("landmarks.csv")

csv_file = open("landmarks.csv", "a", newline="")
writer = csv.writer(csv_file)

# Create CSV header only if file doesn't exist

if not file_exists:
    header = [
        "frame",
        "activity",
        "movement"
    ]

    for i in range(33):
        header.extend([
            f"landmark_{i}_x",
            f"landmark_{i}_y",
            f"landmark_{i}_z"
        ])

    writer.writerow(header)

frame_number = 0
previous_landmarks = None

while True:

    success, frame = camera.read()

    if not success:
        break

    frame_number += 1

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect pose
    results = pose.process(rgb_frame)

    # Only continue if a pose is detected
    if results.pose_landmarks:

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark
        
        # Calculate movement compared with previous frame
        movement = 0
        
        if previous_landmarks is not None:
            for i in range(33):
                dx = landmarks[i].x - previous_landmarks[i][0]
                dy = landmarks[i].y - previous_landmarks[i][1]
                
                movement += math.sqrt(dx**2 + dy**2)
                
        # Save current landmarks for next frame
        previous_landmarks = [
            (landmark.x, landmark.y)
            for landmark in landmarks
        ]

        if movement < 0.1:
            movement_label = "STILL"
        elif movement < 0.5:
            movement_label = "MOVING"
        else:
            movement_label = "HIGH MOVEMENT"

        left_shoulder = landmarks[11]
        left_hip = landmarks[23]
        left_knee = landmarks[25]
        
        angle = calculate_angle(
            left_shoulder,
            left_hip,
            left_knee
        )
        
        if angle > 140:
            detected_activity = "STANDING"
        else:
            detected_activity = "SITTING"

        cv2.putText(
            frame,
            f"Detected: {detected_activity}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
                frame,
                f"Movement: {movement_label}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
        )

        # Row for this frame
        row = [
            frame_number,
            ACTIVITY,
            movement_label
        ]

        # Extract all 33 landmarks
        for landmark in results.pose_landmarks.landmark:

            row.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # Save row to CSV
        writer.writerow(row)

    # Show video
    display_frame = cv2.resize(frame, (800, 800))
    
    cv2.imshow(
    "Landmark Extraction",
    display_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
camera.release()
cv2.destroyAllWindows()
csv_file.close()