from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('best.pt')

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Run detection - imgsz=416 keeps it faster on CPU
    results = model(frame, conf=0.5, imgsz=416)
    annotated = results[0].plot()

    cv2.imshow('Weapon Detection Demo', annotated)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()