import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np

# Initialize OCR with detection + classification + recognition
ocr = PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv4')

# Open default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for PaddleOCR
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # OCR expects a PIL.Image or ndarray (RGB)
    result = ocr.ocr(img_rgb, cls=True)

    # Extract and draw results
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    img_ocr = draw_ocr(img_rgb, boxes, txts, scores, font_path='path/to/font.ttf')

    # Convert back to BGR for OpenCV display
    img_bgr = cv2.cvtColor(np.array(img_ocr), cv2.COLOR_RGB2BGR)

    # Show result
    cv2.imshow("PaddleOCR Live", img_bgr)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
