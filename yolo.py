from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

camera = cv2.Videocapture("./data/video.mp4")

while True:
    ret, frame = camera.read()

    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()