import cv2

camera = cv2.Videocapture(1)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to read camera")
        break

    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()