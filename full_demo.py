from ultralytics import YOLO
import cv2

# Load both models once
coco_model = YOLO('yolov8n.pt')      # pretrained - person + vehicles
weapon_model = YOLO('best.pt')        # your custom - weapons

# COCO class indices: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
coco_classes = [0, 1, 2, 3, 5, 7]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run person + vehicle detection
    coco_results = coco_model(frame, conf=0.5, imgsz=320, classes=coco_classes)

    # Run weapon detection
    weapon_results = weapon_model(frame, conf=0.5, imgsz=320)

    # Draw person/vehicle boxes first
    annotated = coco_results[0].plot()

    # Draw weapon boxes on top of the same frame
    annotated = weapon_results[0].plot(img=annotated)

    cv2.imshow('Person + Vehicle + Weapon Detection', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()