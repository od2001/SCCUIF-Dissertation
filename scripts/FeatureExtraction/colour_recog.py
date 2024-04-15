import cv2
import numpy as np
import matplotlib.pyplot as plt

class HistColour:
    def __init__ (self,image_path):
        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #seperating colour channels
        H1 = image[:,:,0] #blue layer
        S1 = image[:,:,1] #green layer
        V1 = image[:,:,2] #red layer

        #calculating histograms for each channel
        self.hue = cv2.calcHist([image],[0], None, [255], [0,255])
        self.sat = cv2.calcHist([image],[1], None, [255], [0,255])
        self.value = cv2.calcHist([image],[2], None, [255], [0,255])

def colour_recog(img):
    histClassImage = HistColour(img)

    histImage = [histClassImage.hue,histClassImage.sat,histClassImage.value]

    return histImage
    
    