import sys
import os

sys.path.append(os.path.dirname(__file__))

import cv2
from detector import YOLODetector

# -------------------------
# Hard-coded params
# -------------------------
CAR_CLASS_ID = 2
CONF_THRESHOLD = 0.5

# -------------------------
# Load image
# -------------------------
image_path = "data/prototype/images/test3.png"
output_path = "data/prototype/results/annotated/test_annotated.jpg"

img = cv2.imread(image_path)
if img is None:
    print("IMAGE NOT FOUND")
    exit()

# -------------------------
# Run detector
# -------------------------
detector = YOLODetector()
detections = detector.detect(img)

print("Raw detections:")
print(detections)

# -------------------------
# Draw detections
# -------------------------
annotated_img = img.copy()

for det in detections:
    class_id = det["class_id"]
    confidence = det["confidence"]
    x1, y1, x2, y2 = det["bbox"]

    if class_id != CAR_CLASS_ID:
        continue
    if confidence < CONF_THRESHOLD:
        continue

    cv2.rectangle(
        annotated_img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"CAR {confidence:.2f}"

    cv2.putText(
        annotated_img,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# -------------------------
# Save result
# -------------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, annotated_img)

print(f"Annotated image saved to: {output_path}")
