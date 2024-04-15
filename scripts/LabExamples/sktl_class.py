import json
import os
from PIL import Image
import numpy as np

class Skeleton:
    image = ""
    upperbody = {
        "Nose": {"x": -1, "y": -1}, 
        "Neck": {"x": -1, "y": -1}, 
        "LEye": {"x": -1, "y": -1}, 
        "REye": {"x": -1, "y": -1}, 
        "LEar": {"x": -1, "y": -1}, 
        "REar": {"x": -1, "y": -1}, 
        "LShoulder": {"x": -1, "y": -1}, 
        "RShoulder": {"x": -1, "y": -1}, 
        "LElbow": {"x": -1, "y": -1}, 
        "RElbow": {"x": -1, "y": -1}, 
        "LWrist": {"x": -1, "y": -1}, 
        "RWrist": {"x": -1, "y": -1}, 
        "LHip": {"x": -1, "y": -1}, 
        "RHip": {"x": -1, "y": -1}
    }
    lowerbody={
        "LKnee": {"x": -1, "y": -1},
        "RKnee": {"x": -1, "y": -1}, 
        "LAnkle": {"x": -1, "y": -1}, 
        "RAnkle": {"x": -1, "y": -1}
    }


    #search through skeltal data dict

    def __init__(self,skelton_data):
        self.image = skelton_data['image']
        if 'body_parts' in skelton_data['skeltal_data']:
            for x in skelton_data['skeltal_data']['body_parts']:
                    
                        if x in self.upperbody:
                            if skelton_data['skeltal_data']['body_parts'][x]:
                                self.upperbody[x] = skelton_data['skeltal_data']['body_parts'][x]
                            else: 
                                self.upperbody[x] ={"x": -1, "y": -1}
                        else:
                            if skelton_data['skeltal_data']['body_parts'][x]:
                                self.lowerbody[x] = skelton_data['skeltal_data']['body_parts'][x]
                            else: 
                                self.lowerbody[x] ={"x": -1, "y": -1}
        else:
             self.upperbody={}
             self.lowerbody={}

    def get_pose(self):
        # -1 for err, 0 for stand, 1 for sit
        if self.upperbody == {}:
            print("No upperbody to compare")
            return -1
        elif self.lowerbody == {}:
            print("No lowerbody to compare")
            return -1
        else:
            upx=[]
            lowx =[]
            for x in self.upperbody:
                if(self.upperbody[x]["x"] != -1):
                    upx.append(self.upperbody[x]["x"])
            for x in self.lowerbody:
                if(self.lowerbody[x]["x"] != -1):
                    lowx.append(self.lowerbody[x]["x"])

            #Un normalised
            print("LowMin:" + str(np.min(lowx)))
            print("UpMin:" + str(np.min(upx)))
            print("LowAvg:" + str(np.average(lowx)))
            print("UpAvg:" + str(np.average(upx)))
            print("LowMax:" + str(np.max(lowx)))
            print("UpMax:" + str(np.max(upx)))
            print("Diff:" + str(np.average(lowx) -np.average(upx)))

            #Normailsed
            lowXN = [(x - np.min(lowx)) / (np.max(lowx) - np.min(lowx)) for x in lowx]
            upXN = [(x - np.min(upx)) / (np.max(upx) - np.min(upx)) for x in upx]

            print("!Normailsed!")
            print("LowAvg:" + str(np.average(lowXN)))
            print("UpAvg:" + str(np.average(upXN)))
            #Negative for sitting?
            print("Diff:" + str(np.average(lowXN) -np.average(upXN)))

    def get_upper_cloth(self):
        #
        #Max y from neck
        if self.upperbody["Neck"]['y'] == -1:
            print("No Neck found")
            return None
        else:
            max_y = self.upperbody["Neck"]['y']
        #Min y from waist
        if self.upperbody["LHip"]['y'] == -1 and self.upperbody["RHip"]['y'] == -1:
            print("No hips found")
            return None
        else:
            if self.upperbody["LHip"]['y'] != -1:
                min_y = self.upperbody["LHip"]['y']
            else:
                min_y = self.upperbody["RHip"]['y']
        #Min x i.e left side from wrist
        if self.upperbody["LWrist"]['x'] == -1:
            if self.upperbody["LElbow"]['x'] == -1:
                if self.upperbody["LShoulder"]['x'] == -1:
                    print("No left arms found defaulting to left hip")
                    if self.upperbody["LHip"]['x'] != -1:
                        min_x = self.upperbody["LHip"]['x']
                    else:
                        #Could use offest using distance between x neck and rhip to gain undetected left side back
                        #However this may not work if person is not fully captured
                        print("no left hip found")
                        return None
                else:
                    min_x = self.upperbody["LShoulder"]['x']
            else:
                 min_x = self.upperbody["LElbow"]['x']
        else:
            min_x = self.upperbody["LWrist"]['x']
        #Max x i.e right side from wrist 
        if self.upperbody["RWrist"]['x'] == -1:
            if self.upperbody["RElbow"]['x'] == -1:
                if self.upperbody["RShoulder"]['x'] == -1:
                    print("No right arms found defaulting to right hip")
                    if self.upperbody["RHip"]['x'] != -1:
                        max_x = self.upperbody["RHip"]['x']
                    else:
                        #Could use offest using distance between x neck and rhip to gain undetected left side back
                        #However this may not work if person is not fully captured
                        print("no left hip found")
                        return None
                else:
                    max_x = self.upperbody["RShoulder"]['x']
            else:
                 max_x = self.upperbody["RElbow"]['x']
        else:
            max_x = self.upperbody["RWrist"]['x']

        ogimg = Image.open(self.image)
        
        #Openpose handles left to right relative to person not camera
        if max_x < min_x:
            min_x, max_x = max_x, min_x
        if max_y < min_y:
            min_y, max_y = max_y, min_y

        bbox = (min_x,min_y,max_x,max_y)
        print(bbox)
        cropped_img = ogimg.crop(bbox)
        return cropped_img

    
    def get_lower_cloth(self):
        #Max y from hips
        if self.upperbody["LHip"]['y'] == -1 and self.upperbody["RHip"]['y'] == -1:
            print("No hips found")
            return None
        else:
            if self.upperbody["LHip"]['y'] != -1:
                max_y = self.upperbody["LHip"]['y']
            else:
                max_y = self.upperbody["RHip"]['y']

        #Min elements
        
        #trying ankles
        if self.lowerbody["LAnkle"]['y'] == -1 and self.lowerbody["RAnkle"]['y'] == -1:
            print("No ankles found, trying knees")
            #Assuming lower clothing is uniform in colour
            if self.lowerbody["LKnee"]['y'] == -1 and self.lowerbody["RKnee"]['y'] == -1:
                print("no lower body found")
                return None
            else:
                #Trying Knees
                #Both Knees
                if self.lowerbody["LKnee"]['y'] != -1 and self.lowerbody["RKnee"]['y'] != -1:
                    min_y = self.lowerbody["LKnee"]['y']
                    min_x = self.lowerbody["LKnee"]['x']
                    max_x = self.lowerbody["RKnee"]['x']
                #Left Knee
                elif self.lowerbody["LKnee"]['y'] != -1:
                    min_y = self.lowerbody["LKnee"]['y']
                    min_x = self.lowerbody["LKnee"]['x']
                    if self.upperbody["RHip"]['x'] != -1:
                        max_x = self.upperbody["RHip"]['x']
                    else:
                        print("No right side found")
                #Right Knee
                else:
                    min_y = self.lowerbody["RKnee"]['y']
                    max_x = self.lowerbody["RKnee"]['x']
                    if self.upperbody["LHip"]['x'] != -1:
                        min_x = self.upperbody["LHip"]['x']
                    else:
                        print("No left side found")
        else:
            #If both ankles are present
            if self.lowerbody["LAnkle"]['y'] != -1 and self.lowerbody["RAnkle"]['y'] != -1:
                min_y = self.lowerbody["LAnkle"]['y']
                min_x = self.lowerbody["LAnkle"]['x']
                max_x = self.lowerbody["RAnkle"]['x']
            else:
                #Only left ankle is present
                if self.lowerbody["LAnkle"]['y'] != -1:
                    min_y = self.lowerbody["LAnkle"]['y']
                    min_x = self.lowerbody["LAnkle"]['x']
                    #Get right side from either knee or hips
                    if self.lowerbody["RKnee"]['x'] != -1:
                        max_x = self.lowerbody["RKnee"]['x']
                    else:
                        if self.upperbody["RHip"]['x'] != -1:
                            max_x = self.upperbody["RHip"]['x']
                        else:
                            print("No right side found")

                #Only right ankle is present
                else:
                    min_y = self.lowerbody["RAnkle"]['y']
                    max_x = self.lowerbody["RAnkle"]['x']
                    #Get left side from either knee or hips
                    if self.lowerbody["LKnee"]['x'] != -1:
                        min_x = self.lowerbody["LKnee"]['x']
                    else:
                        if self.upperbody["LHip"]['x'] != -1:
                            min_x = self.upperbody["LHip"]['x']
                        else:
                            print("No left side found")
        

        ogimg = Image.open(self.image)
        #Openpose handles left to right relative to person not camera
        if max_x < min_x:
            min_x, max_x = max_x, min_x
        if max_y < min_y:
            min_y, max_y = max_y, min_y

        bbox = (min_x,min_y,max_x,max_y)
        print(bbox)
        cropped_img = ogimg.crop(bbox)
        return cropped_img



        
    def get_bdp_img(self,body_part1,body_part2):
        if body_part1 in self.upperbody and body_part2 in self.upperbody  :
            skeltal_image = Image.open(self.image)
            x1 = self.upperbody[body_part1]["x"]
            x2 = self.upperbody[body_part2]["x"]
            y1 = self.upperbody[body_part1]["y"]
            y2 = self.upperbody[body_part2]["y"]
            bbox = (x2,y1,x1,y2)
            cropped_img = skeltal_image.crop(bbox)
            cropped_img.show()
        else:
            print("body part could not be found in image")
            

    def get_stand_sit(self):
        #Feasiblity Check
        isLHip = self.upperbody["LHip"]["y"] != -1
        isRHip = self.upperbody["RHip"]["y"] != -1
        isLKnee = self.lowerbody["LKnee"]["y"] != -1
        isLAnkle = self.lowerbody["LAnkle"]["y"] != -1
        isRKnee = self.lowerbody["RKnee"]["y"] != -1
        isRAnkle = self.lowerbody["RAnkle"]["y"] != -1

        HipToKneeLength = 0
        KneeToAnkleLength = 0


        if isLHip and isLKnee and isLAnkle or isRHip and isRKnee and isRAnkle:
            if isLHip and isLKnee and isLAnkle:
                KneeToAnkleLength = self.upperbody['LHip']["y"] - self.lowerbody["LKnee"]["y"]
                HipToKneeLength = self.lowerbody["LKnee"]["y"] - self.lowerbody["LAnkle"]["y"]
            else:
                KneeToAnkleLength = self.upperbody['RHip']["y"] - self.lowerbody["RKnee"]["y"]
                HipToKneeLength = self.lowerbody["RKnee"]["y"] - self.lowerbody["rAnkle"]["y"]
        else:
            print("No Usable lowerbody")
            return None
        # If the height of the hips to the knees is greater than
        # the height of the knees to the ankles then return stand else return sit
        return True if HipToKneeLength >= KneeToAnkleLength  else False