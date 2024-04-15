
import json
import pandas as pd
import os
import colour_comp as cp
import sktl_class as sk
from PIL import Image

def check_dimensions_within_bounds(ref, new):
    # Get dimensions of both images
    width_ref, height_ref = ref.size
    width_new, height_new = new.size

    # Check if image b's dimensions are within the bounds of image a
    if width_new <= width_ref and height_new <= height_ref:
        return True
    else:
        return False
    
def get_obj_placement(df, x):
    if 0 < x < len(df) - 1:
        return (df.iloc[x-1]['label'], df.iloc[x]['label'], df.iloc[x+1]['label'])
    return None  



def comp_img_2_img(imgdir1,imgdir2):
    comp_score = 0
    occurences_score = 0
    placement_score = 0
    colour_score = 0
    person_score = 0

    comp_weight = 1
    occurences_weight = 2
    placement_weight =1
    colour_weight = 3
    person_weight = 2 

    colour_distance_thres = 0.2
    colour_hist_thres = 0.8
    colour_lab_thres = 0.2


    img1Df = pd.read_json(imgdir1 + "/" + imgdir1 + "_feature_df.json")
    img2Df = pd.read_json(imgdir2 + "/" + imgdir2 + "_feature_df.json")

    img1DfOrderedLablesAsc = img1Df.sort_values(by='label', ascending=True)['label']
    img2DfOrderedLablesAsc = img2Df.sort_values(by='label', ascending=True)['label']

    img1LabelAscCounts = img1DfOrderedLablesAsc.value_counts()
    img2LabelAscCounts = img2DfOrderedLablesAsc.value_counts()

    common_lables = img1LabelAscCounts.index.intersection(img2LabelAscCounts.index)

    for value in common_lables:
        count1 = img1LabelAscCounts[value]
        count2 = img2LabelAscCounts[value]
        if count1 == count2:
            occurences_score +=1 * occurences_weight
            comp_score += 1

    

    # Why is wouldn't wokr
    for i in range(1, len(img1Df) - 1):
        if img1Df.iloc[i]['idNo'] != -1:
            img1_placement = get_obj_placement(img1Df, i)
            if img1_placement:
                for j in range(1, len(img2Df) - 1):
                    if img2Df.iloc[j]['idNo'] != -1:
                        img2_placement = get_obj_placement(img2Df, j)
                        if img1_placement == img2_placement and img1_placement != None and img2_placement !=None:
                            placement_score += 1 * placement_weight
                            comp_score += 1
                        
    people1Index = img1Df[img1Df['image'].str.split('_').str[0] == 'person'].index
    people2Index = img2Df[img2Df['image'].str.split('_').str[0] == 'person'].index

    for i in people1Index:
        print("PERSON: " + img1Df.iloc[i]['image'] )

    for i in people1Index:
        personiimg=img1Df.iloc[-1]['image'].split('.')[0] + "/masks_" + img1Df.iloc[-1]['image'].split('.')[0]  + "/" + img1Df.iloc[i]['image']
        personisktldata = {"image": personiimg, "skeltal_data": {"body_parts":  img1Df.iloc[i, 5:].to_dict()}}
        print(personisktldata) 
        if any(value is not None for value in personisktldata['skeltal_data']['body_parts'].values()):
            print("This passed")
            personisktl = sk.Skeleton(personisktldata)

###Upper#####
                        
            if(type(personisktl.get_upper_cloth()) is not type(None) and check_dimensions_within_bounds(Image.open(personiimg),personisktl.get_upper_cloth())):
                personiUp = personisktl.get_upper_cloth()
                personiUpimage = 'person_i_Up.png'
                print("\n Type: " + str(type(personiUp)) + "\n")
                try:
                    personiUp.save(personiUpimage)
                    print("\n" + imgdir1 + ":" + personiimg + " Upper \n")
                    personiUp.show()
                    print("\n")
                    for j in people2Index:
                        personjimg=img2Df.iloc[-1]['image'].split('.')[0] + "/masks_" + img2Df.iloc[-1]['image'].split('.')[0]  + "/" + img2Df.iloc[j]['image']
                        personjsktldata = {"image": personjimg, "skeltal_data": {"body_parts":  img2Df.iloc[j, 5:].to_dict()}}
                        personjsktl = sk.Skeleton(personjsktldata)

                        ###People to pepole

                        if cp.lab_space_comp_percentage(personiimg,personjimg) < 0.2:
                                comp_score +=1
                                person_score +=1 * person_weight

                        if(type(personjsktl.get_upper_cloth()) is not type(None) and check_dimensions_within_bounds(Image.open(personjimg),personjsktl.get_upper_cloth())):
                            personjUp = personjsktl.get_upper_cloth()
                            personjUpimage = 'person_j_Up.png'
                            print("\n Type: " + str(type(personjUp)) + "\n")
                            try:
                                personjUp.save(personjUpimage)
                                print("\n" + imgdir2 + ":" + personjimg + " Upper \n")
                                personjUp.show()
                                print("\n")
                                if cp.lab_space_comp_percentage(personiUpimage,personjUpimage) < colour_lab_thres:
                                    comp_score +=1
                                    person_score +=1 * person_weight
                                
                                os.remove(personjUpimage)
                            except:
                                print(personjimg + "in up caused problem")

                    os.remove(personiUpimage)
                except:
                    print(personiimg + "in up caused problem")

####Lower####
            if(type(personisktl.get_lower_cloth()) is not type(None) and check_dimensions_within_bounds(Image.open(personiimg),personisktl.get_lower_cloth())):
                personiLow = personisktl.get_lower_cloth()
                personiLowImage = 'person_i_Low.png'
                print("\n Type: " + str(type(personiLow)) + "\n")
                try:
                    personiLow.save(personiLowImage)
                    print("\n" + imgdir1 + ":" + personiimg + " Lower \n")
                    personiLow.show()
                    print("\n")

                    for j in people2Index:
                        personjimg=img2Df.iloc[-1]['image'].split('.')[0] + "/masks_" + img2Df.iloc[-1]['image'].split('.')[0]  + "/" + img2Df.iloc[j]['image']
                        personjsktldata = {"image": personjimg, "skeltal_data": {"body_parts":  img2Df.iloc[j, 5:].to_dict()}}
                        personjsktl = sk.Skeleton(personjsktldata)
                        if(type(personjsktl.get_lower_cloth()) is not type(None) and check_dimensions_within_bounds(Image.open(personjimg),personjsktl.get_lower_cloth())):
                            personjLow = personjsktl.get_lower_cloth()
                            personjLowImage = 'person_j_Low.png'
                            print("\n Type: " + str(type(personjLow)) + "\n")
                            try:
                                personjLow.save(personjLowImage)
                                print("\n" + imgdir2 + ":" + personjimg + " Lower \n")
                                personjLow.show()
                                print("\n")
                                if cp.lab_space_comp_percentage(personiLowImage,personjLowImage) < colour_lab_thres:
                                    comp_score +=1
                                    person_score +=1 * person_weight
                                
                                os.remove(personjLowImage)
                            except:
                                print(personjimg + "in low caused problem")


                    os.remove(personiLowImage)
                except:
                     print(personiimg + "in low caused problem")

                          
    for i1,obj1 in img1Df.iloc[:-1].iterrows():
        if obj1['label'] != 'person':
            for i2,obj2 in img2Df.iloc[:-1].iterrows():
                if obj1['label'] == obj2['label']:
                    objimg1=img1Df.iloc[-1]['image'].split('.')[0] + "/masks_" + img1Df.iloc[-1]['image'].split('.')[0]  + "/" + img1Df.iloc[i1]['image']
                    objimg2=img2Df.iloc[-1]['image'].split('.')[0] + "/masks_" + img2Df.iloc[-1]['image'].split('.')[0]  + "/" + img2Df.iloc[i2]['image']


                    if cp.get_distance(objimg1,objimg2) < colour_distance_thres:
                        comp_score +=1
                        colour_score +=1 * colour_weight

                    if cp.hist_color_intersect(objimg1,objimg2) > colour_hist_thres:
                        comp_score +=1
                        colour_score +=1 * colour_weight
                    if cp.lab_space_comp_percentage(objimg1,objimg2) < 0.2:
                        comp_score +=1 * colour_weight

    return [comp_score,occurences_score,placement_score,colour_score,person_score]


# #Get list of all dirs in the images dir 

print(os.getcwd())

# List all entries in the directory
entries = os.listdir()

# Filter out the entries that are directories
imagefolders = [entry for entry in entries if os.path.isdir(os.path.join(entry))]

comp_data = []

print(imagefolders)

for i in range(len(imagefolders)):
     for j in range(i, len(imagefolders)):
            print('comparing :' + imagefolders[i] + " to " + imagefolders[j] )
            scores_array = comp_img_2_img(imagefolders[i],imagefolders[j])
            comp = {
                'image1' : imagefolders[i],
                'image2' : imagefolders[j],
                'comp_score' : scores_array[0],
                'occurnences_score': scores_array[1],
                'placement_score' : scores_array[2],
                'colour_score' : scores_array[3],
                'person_score' : scores_array[4]
            }
            comp_data.append(comp)



with open('image_comparison_rev_2.json', 'w') as f:
    json.dump(comp_data, f)


