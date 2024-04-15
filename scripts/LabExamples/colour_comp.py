import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from skimage import io
from skimage.color import rgb2lab,deltaE_ciede94,deltaE_cie76,deltaE_ciede2000,deltaE_cmc

class HistColour:
    def __init__ (self,image_path):
        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #calculating histograms for each channel
        self.hue = cv2.calcHist([image],[0], None, [255], [0,255])
        self.sat = cv2.calcHist([image],[1], None, [255], [0,255])
        self.value = cv2.calcHist([image],[2], None, [255], [0,255])

class HistColourPercentage:
    def __init__ (self,image_path):
        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calculate the total number of pixels
        total_pixels = image.shape[0] * image.shape[1]

        hue_value = cv2.calcHist([image],[0], None, [255], [0,255])
        sat_value = cv2.calcHist([image],[1], None, [255], [0,255])
        val_value = cv2.calcHist([image],[2], None, [255], [0,255])


        #calculating histograms for each channel
        self.hue = (hue_value/total_pixels) * 100
        self.sat = (sat_value/total_pixels) * 100
        self.value = (val_value/total_pixels) * 100


# Distance hue
def get_distance(image1Name, image2Name):
    hist1 = HistColourPercentage(image1Name)
    hist2 = HistColourPercentage(image2Name)

    #Difference between all the hue points
    hue = np.subtract(hist1.hue, hist2.hue)

    distances = hue

    # Feature Scalling
    scaled_distance = (np.average(distances) - np.min(distances)) / (np.max(distances) - np.min(distances))

    return scaled_distance




def get_distance_all(image1Name, image2Name):
    hist1 = HistColourPercentage(image1Name)
    hist2 = HistColourPercentage(image2Name)

    #Difference between all the hue points
    hue = np.subtract(hist1.hue, hist2.hue)
    sat = np.subtract(hist1.sat, hist2.sat)
    val = np.subtract(hist1.value, hist2.value)

    distances = hue + sat + val

    # Feature Scalling
    scaled_distance = (np.average(distances) - np.min(distances)) / (np.max(distances) - np.min(distances))

    return scaled_distance
    


# Distance hue
def get_distance_no_norm(image1Name, image2Name):
    hist1 = HistColour(image1Name)
    hist2 = HistColour(image2Name)

    #Difference between all the hue points
    hue = np.subtract(hist1.hue, hist2.hue)

    distances = hue

    # Feature Scalling
    scaled_distance = (np.average(distances) - np.min(distances)) / (np.max(distances) - np.min(distances))

    return np.average(scaled_distance)





def hist_color_intersect(image1Name, image2Name):
    img1 = cv2.imread(image1Name)
    img2 = cv2.imread(image2Name)
    # Convert images to HSV color space
    image1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    image2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Calculate histograms
    hist1 = cv2.calcHist([image1_hsv],[0], None, [255], [0,255])
    hist2 = cv2.calcHist([image2_hsv],[0], None, [255], [0,255])

    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Compute histogram intersection
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

    # Normalize the intersection by dividing it by the sum of all bins in the smaller histogram
    smaller_hist_sum = min(sum(hist1), sum(hist2))
    normalized_intersection = intersection / smaller_hist_sum if smaller_hist_sum > 0 else 0

    return normalized_intersection




def lab_space_comp(img1Name,img2Name):
    lab_data = []
    img1 = rgb2lab(io.imread(img1Name))
    img2 = rgb2lab(io.imread(img2Name))

    if img1.shape != img2.shape:
        img2 = cv2.resize(rgb2lab(io.imread(img2Name)), (img1.shape[:2][1],img1.shape[:2][0]))

    #Choose function based on label 
    # 1: Car
    if img1Name.split('_')[0] == "car" and img2Name.split('_')[0] == "car":
        #Show this works better on cars
        lab_data = deltaE_ciede2000(img1,img2)
    # 2: Person
    elif img1Name.split('_')[0] == "person" and img2Name.split('_')[0] == "person":
        #Show this works better on clothes
        lab_data = deltaE_cmc(img1,img2)
    # 3: Default
    else:
        lab_data = deltaE_ciede94(img1,img2)

    return np.average(lab_data)

def lab_space_comp_percentage(img1Name,img2Name):
    lab_data = []
    #Read images
    img1 = rgb2lab(io.imread(img1Name))
    img2 = rgb2lab(io.imread(img2Name))

    #Resize images if they arent equal
    if img1.shape != img2.shape:
        img2 = cv2.resize(rgb2lab(io.imread(img2Name)), (img1.shape[:2][1],img1.shape[:2][0]))

    #Normailse the images
    img1 = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #Choose function based on label 
    # 1: Car
    if img1Name.split('_')[0] == "car" and img2Name.split('_')[0] == "car":
        lab_data = deltaE_ciede2000(img1,img2)
    # 2: Person
    elif img1Name.split('_')[0] == "person" and img2Name.split('_')[0] == "person":
        lab_data = deltaE_cmc(img1,img2,kL=2,kC=1)
    # 3: Default
    else:
        lab_data = deltaE_cie76(img1,img2)

    return np.average(lab_data)

    

def img_2_img(src,ogImg1Name,ogImg2Name):
    img1Name = src + ogImg1Name
    img2Name = src + ogImg2Name
    img_2_img_comp_data = {
       #----Metric 1: Basic Distance 
        "comp" : ("src= " + src+ " : " + img1Name + "-" + img2Name),
        "basic_distance" :  str(get_distance(img1Name,img2Name)) + "%",
        "hist_intersection" :hist_color_intersect(img1Name,img2Name),
        "lab_space" : lab_space_comp(img1Name,img2Name)
    }
    
    return img_2_img_comp_data


def img_2_all(src,img1Name,imgList):
    img2_all_comp_data = {
        "overall_comp" : (src + ":" + img1Name),
        "list_comp" :[]
    }
    list_img_comp = []
    for i in imgList:
        list_img_comp.appennd(img_2_img(src,img1Name,i))

    img2_all_comp_data["list_comp"] = list_img_comp

    return img2_all_comp_data
    

# # os.chdir("././test_images/Lab")
# os.chdir("scripts/Comparisions")

# print("\n")
# print("\n")
# print("Base 1 person_1.png; ")
# print(img_2_img("","person_1.png","person_1.png")) 
# print("\n")
# print("Base 2 person_3.png; ")
# print(img_2_img("","person_3.png","person_3.png")) 
# print("\n")
# print("Comparision:")
# print(img_2_img("","person_3.png","person_1.png")) 
# print("\n")
# print("\n")

# print("Base 1 Blured_upper_body.jpg; ")
# print(img_2_img("","Blured_upper_body.jpg","Blured_upper_body.jpg")) 
# print("\n")
# print("Base 2 person_3_upper_body.jpg; ")
# print(img_2_img("","person_3_upper_body.jpg","person_3_upper_body.jpg")) 
# print("\n")
# print("Comparision:")
# print(img_2_img("","person_3_upper_body.jpg","Blured_upper_body.jpg")) 
# print("\n")
# print("\n")

# print("Base 1 Blured_lower_body.jpg; ")
# print(img_2_img("","Blured_lower_body.jpg","Blured_lower_body.jpg")) 
# print("\n")
# print("Base 2 person_3_lower_body.jpg; ")
# print(img_2_img("","person_3_lower_body.jpg","person_3_lower_body.jpg")) 
# print("\n")
# print("Comparision:")
# print(img_2_img("","Blured_lower_body.jpg","person_3_lower_body.jpg")) 
# print("\n")
# print("\n")




