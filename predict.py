import cv2
import mediapipe as mp
import joblib
import pandas as pd

# Load trained model
model = joblib.load("activity_model.pkl")

# MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Choose video
VIDEO = "./data/crazy.mp4"

camera = cv2.VideoCapture(VIDEO)

if not camera.isOpened():
    print("ERROR: Could not open video")
    exit()

prediction_history = []

while True:
    success, frame = camera.read()

    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # Extract the same 99 landmark features used during training
        features = []

        for landmark in landmarks:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # Convert to DataFrame
        X = pd.DataFrame(
            [features],
            columns=[
                f"landmark_{i}_{axis}"
                for i in range(33)
                for axis in ["x", "y", "z"]
            ]
        )

        # Predict activity
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = max(probabilities) * 100
        
        prediction_history.append(prediction)
        
        if len(prediction_history) > 10:
            prediction_history.pop(0)
            
        stable_prediction = max(
            set(prediction_history),
            key=prediction_history.count
        )

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Display prediction
        cv2.putText(
            frame,
            f"Activity: {stable_prediction} ({confidence:.1f}%)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    display_frame = cv2.resize(frame, (800, 800))
    cv2.imshow("AI Human Activity Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()