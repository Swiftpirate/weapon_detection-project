from ultralytics import YOLO
import cv2

# Load both models
person_model = YOLO('yolov8n.pt')      # pretrained COCO model - detects person
weapon_model = YOLO('best.pt')          # your custom trained model - detects weapons

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run person detection (class 0 = person in COCO)
    person_results = person_model(frame, conf=0.5, imgsz=320, classes=[0])
    
    # Run weapon detection (your 9 classes)
    weapon_results = weapon_model(frame, conf=0.5, imgsz=320)

    # Draw person boxes first
    annotated = person_results[0].plot()
    
    # Draw weapon boxes on top of that same frame
    annotated = weapon_results[0].plot(img=annotated)

    cv2.imshow('Person + Weapon Detection', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()