import cv2 
from ultralytics import YOLO
import numpy as np

#load pre-trained model
model = YOLO("yolov8n.pt")


#set reference width
KNOWN_WIDTH_MM = 76.2
pixels_per_metric = None

#load image 
img = cv2.imread("MachineLearningTestImage.jpg")
results = model(img)   #where the L part of MLK occurs

for r in results:
    boxes = r.boxes
    for box in boxes:
    # Get coordinates of the bounds
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        pixel_width = x2-x1
        #get name of object
        cls = int(box.cls[0])
        label = model.names[cls]

        #use first object detected as reference
        if pixels_per_metric is None:
            pixels_per_metric = pixel_width/KNOWN_WIDTH_MM
            print(f"Scale using {label}")

        actual_size = pixel_width/pixels_per_metric

        #draw on imave
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,2),2 )
        cv2.putText(img, f"{label}: {actual_size:.1f}mm", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("ML Dimension Detection", img)
cv2.waitKey(0)


