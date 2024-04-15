# coding: utf-8
# =====================================================================
#  Filename:    text_recognition.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Recognizes regions of text in a given image
#
#  Usage: python text_recognition.py --east frozen_east_text_detection.pb --image test.png
#         or
#         python text_recognition.py --east frozen_east_text_detection.pb --image test.png --padding 0.25
#
#  Note: Requires opencv 3.4.2 or later
#        Requires tesseract 4.0 or later
# =====================================================================

### --- Code based on the text recog from OpenCV EAST --- ###

import pytesseract
import numpy as np
import argparse
import cv2
import sys 
import os 
sys.path.insert(0,"C:/Users/Ossia/Documents/SCCUIF/OpenCV-Tesseract-EAST-Text-Detector")
from utils import forward_passer, box_extractor
from text_detection import resize_image
from imutils.object_detection import non_max_suppression


def txt_recog(img, min_confidence):
    # East Text Detector
    detector= "C:/Users/Ossia/Documents/SCCUIF/OpenCV-Tesseract-EAST-Text-Detector/frozen_east_text_detection.pb"
    # Read the image
    image = cv2.imread(img,cv2.COLOR_RGB2GRAY)
    image = cv2.medianBlur(image, 5)

    # Define the threshold value
    threshold_value = 127  # Commonly used value, but you might need to adjust it for your specific image

    # Apply binary thresholding
    ret, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)


    # Make a copy
    orig_image = image.copy()
    orig_h, orig_w = orig_image.shape[:2]

    # Resize the image for the CNN
    image, ratio_w, ratio_h = resize_image(image,  3840, 2176)

    # layers used for ROI recognition
    layer_names = ['feature_fusion/Conv_7/Sigmoid',
                   'feature_fusion/concat_3']

    # pre-loading the frozen graph
    print("[INFO] loading the detector...")
    net = cv2.dnn.readNet(detector)

    # getting results from the model
    scores, geometry = forward_passer(net, image, layers=layer_names)

    # decoding results from the model
    rectangles, confidences = box_extractor(scores, geometry, min_confidence)

    # applying non-max suppression to get boxes depicting text regions
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    results = []

    # text recognition main loop
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * ratio_w)
        start_y = int(start_y * ratio_h)
        end_x = int(end_x * ratio_w)
        end_y = int(end_y * ratio_h)

        dx = int((end_x - start_x))
        dy = int((end_y - start_y))

        start_x = max(0, start_x - dx)
        start_y = max(0, start_y - dy)
        end_x = min(orig_w, end_x + (dx*2))
        end_y = min(orig_h, end_y + (dy*2))

        # ROI to be recognized
        roi = orig_image[start_y:end_y, start_x:end_x]

        # Get whole words
        config = '-l eng --oem 1 --psm 7 '
        text = pytesseract.image_to_string(roi, config=config)

        # collating results
        results.append(((start_x, start_y, end_x, end_y), text))

    # sorting results top to bottom
    results.sort(key=lambda r: r[0][1])

    return results

    