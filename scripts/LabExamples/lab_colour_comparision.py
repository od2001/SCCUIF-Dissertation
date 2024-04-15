import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


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

# Distance measuring d= ((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 )^1/2
def get_distance(hist1,hist2):
    hue = np.subtract(hist1.hue,hist2.hue)
    sat = np.subtract(hist1.sat,hist2.sat)
    value = np.subtract(hist1.value,hist2.value)

    hueSquare = np.square(hue)
    satSquare = np.square(sat)
    valueSquare = np.square(value)

    totalAdd = np.add(np.add(hueSquare,satSquare),valueSquare)

    return np.average(np.sqrt(totalAdd))

def get_distance_hue(hist1,hist2):
    hue = np.subtract(hist1.hue,hist2.hue)
    return np.average(hue)

def get_distance_hue_sat(hist1,hist2):
    hue = np.subtract(hist1.hue,hist2.hue)
    sat = np.subtract(hist1.sat,hist2.sat)

    hueSquare = np.square(hue)
    satSquare = np.square(sat)

    totalAdd = np.add(hueSquare,satSquare)
    return np.average(np.sqrt(totalAdd))





list_of_apple_hist = []
list_of_orange_hist = []

list_of_apple_images = os.listdir("././test_images/Apple")


list_of_orange_images = os.listdir("././test_images/Orange")


os.chdir("././test_images/Apple")
for apple_image in list_of_apple_images:
    image_hist = HistColour(apple_image)
    list_of_apple_hist.append(image_hist)


os.chdir("../Orange")  
print(os.getcwd()) 
for orange_image in list_of_orange_images:
    image_hist = HistColour(orange_image)
    list_of_orange_hist.append(image_hist)


list_of_lab_images = os.listdir("../Lab")
list_of_lab_hist = []
os.chdir("../Lab")  
print(os.getcwd()) 
for lab_image in list_of_lab_images:
    image_hist = HistColour(lab_image)
    list_of_lab_hist.append(image_hist)

for i in range(0,len(list_of_lab_images)):
    print(str(i) + " " + str(list_of_lab_images[i]))

# # Visualizing histograms 2D
# plt.subplot(2, 2, 1)
# plt.plot(list_of_apple_hist[0].hue, color='blue')  # Using magenta for Hue
# cool_patch = mpatches.Patch(color = 'blue', label='Cool White')
# plt.subplot(2, 2, 1)
# plt.plot(list_of_apple_hist[1].hue, color='red')  # Using magenta for Hue
# nat_patch = mpatches.Patch(color = 'red', label='Natural')
# plt.subplot(2, 2, 1)
# plt.plot(list_of_apple_hist[2].hue, color='green')  # Using magenta for Hue
# shadow_patch = mpatches.Patch(color = 'green', label='Shadow')
# plt.subplot(2, 2, 1)
# plt.plot(list_of_apple_hist[3].hue, color='yellow')  # Using magenta for Hue
# warm_patch = mpatches.Patch(color = 'yellow', label='Warm White')
# plt.legend(handles=[cool_patch,nat_patch,shadow_patch,warm_patch])

# plt.title('Hue of Apple under different lighting conditions')
# plt.show()


# 3D ploting
    

#ANGLES IN THE CARS BACK OF THE CAR ISNNT PRESENT IN THE FIRST 
# print(get_distance_hue_sat(list_of_lab_hist[5],list_of_lab_hist[4]))
    
# --------------------------------------------------------- distance is a bit meh

def fuzzy_color_match(image1, image2):
    # Convert images to HSV color space
    image1_hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    image2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # # Calculate histograms
    # hist1 = cv2.calcHist([image1_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # hist2 = cv2.calcHist([image2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # Calculate histograms
    hist1 = cv2.calcHist([image1_hsv],[0], None, [255], [0,255])
    hist2 = cv2.calcHist([image2_hsv],[0], None, [255], [0,255])

    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Compute histogram intersection
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

    return intersection
    
car1Img = cv2.imread("car_1.png")
car3Img = cv2.imread("car_3.png")
or1Image = cv2.imread("O1.jpeg")
or2Image = cv2.imread("O2.jpeg")
angl1 = cv2.imread("Angle1.jpeg")
angl2 = cv2.imread("Angle2.jpeg")
angl3 = cv2.imread("Angle3.jpeg")

print(fuzzy_color_match(or1Image,angl1))


# ---------------------- Fuzzy a little bit tempromental


from skimage import io

from skimage.color import rgb2lab,deltaE_ciede94,deltaE_ciede2000,deltaE_cmc

car1LabImg = cv2.resize(rgb2lab(io.imread("car_1.png")), (5712, 3213))
car3LabImg = rgb2lab(io.imread("car_3.png"))




# This used for cars check the function def
print(np.average(deltaE_ciede2000(car1LabImg,car3LabImg)))


#Clothing colour useage
#deltaE_cmc

