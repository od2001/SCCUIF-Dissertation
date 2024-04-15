import cv2
import json
import os 


def main(jsonfile):
  
    jsonfile = sorted(jsonfile,key=lambda item: item['bbox'][0])

    counter = 0
    for i in jsonfile:
        if i['bbox'][0] == -1:
            i["idNo"] = -1
        else:
            i["idNo"] = counter
            counter += 1
    
    return jsonfile

def getLR(jsonfile,x):

    if(x == 0):
        print("Main" + str(jsonfile[x]))
        print("Right" + str(jsonfile[x+1]))
    elif (x == len(jsonfile)):
        print("Left" + str(jsonfile[x-2]))
        print("Main" + str(jsonfile[x-1]))
    else:
        print("Left" + str(jsonfile[x-2]))
        print("Main" + str(jsonfile[x-1]))
        print("Right" + str(jsonfile[x]))


print(os.getcwd())

os.chdir("scripts/LabExamples")

#jsonfileMain = [{"label": "person", "object_image": "person_1.png", "bbox": [3633, 1679, 4091, 3202]}, {"label": "car", "object_image": "car_2.png", "bbox": [1576, 1783, 1715, 1886]}, {"label": "bicycle", "object_image": "bicycle_3.png", "bbox": [5282, 2023, 5707, 2623]}, {"label": "person", "object_image": "person_4.png", "bbox": [1345, 1768, 1402, 1955]}, {"label": "person", "object_image": "person_5.png", "bbox": [1525, 1765, 1582, 1914]}, {"label": "bicycle", "object_image": "bicycle_6.png", "bbox": [4263, 2005, 4717, 2432]}, {"label": "bicycle", "object_image": "bicycle_7.png", "bbox": [4808, 2136, 5067, 2531]}, {"label": "bicycle", "object_image": "bicycle_8.png", "bbox": [5248, 2032, 5340, 2087]}, {"label": "bicycle", "object_image": "bicycle_9.png", "bbox": [4603, 2078, 4889, 2466]}, {"label": "window", "object_image": "window_10.png", "bbox": [-1]}, {"label": "tree", "object_image": "tree_11.png", "bbox": [-1]}, {"label": "sky", "object_image": "sky_12.png", "bbox": [-1]}, {"label": "pavement", "object_image": "pavement_13.png", "bbox": [-1]}, {"label": "building", "object_image": "building_14.png", "bbox": [-1]}]


with open("panoptic_predictions.json", 'r') as f:
                jsonfileMain = json.load(f)
 

jsonfileMain = main(jsonfileMain)

for i in jsonfileMain:
    print(i)

print(" ")

getLR(jsonfileMain,0)

jsonfileRev = sorted(jsonfileMain,reversed)