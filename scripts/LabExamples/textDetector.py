from PIL import Image

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

import cv2

img_cv = cv2.imread("./completed_images/IMG_2313.jpeg")

img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)


recognized_text = pytesseract.image_to_string(img_rgb)
d = pytesseract.image_to_data(img_rgb, config ="oem 1 psm 7", output_type=pytesseract.Output.DICT)
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 0:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img_rgb = cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
img_with_text = img_rgb.copy()       
cv2.putText(img_with_text, recognized_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', 600, 400)
cv2.imshow("Result",img_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows() 