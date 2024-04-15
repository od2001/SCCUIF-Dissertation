#Package Imports
from Detector import *
import dlib
import os
from PIL import Image
from exif import Image as exifImage
from txt_recog import txt_recog
import argparse
from skeltal_recog import skeltal_recog
from colour_recog import colour_recog
import pandas as pd

from dataframecreation import create_df

#Grab args from cmd line
def get_arguments():

    ap = argparse.ArgumentParser()
    #Image folder path arg
    ap.add_argument('-i', '--images', type=str,default="",
                    help='full path to images folder')
    #Object Recognition flag
    ap.add_argument('-o', '--object', action='store_true',
                help='indication to use object recognition')
    #Text Recogntion flag
    ap.add_argument('-t', '--text',action='store_true',
                    help='indication to use text detection')
    #Skeletal Recogntion flag
    ap.add_argument('-s', '--skeletal',action='store_true',
                help='indication to use skeltal detection')
    arguments = vars(ap.parse_args())

    return arguments



def main(imagespath,objR,txtR,sktlR):
    # Load the pre-trained face detection model
    faceDetector = dlib.get_frontal_face_detector()

    #Instatie classes with panoptic segementatiom model type
    detector = Detector(model_type="PS")

    #If imagepath not set then go to default 
    if imagespath == "":
        #Image directory (dir)
        images_dir = "././images"
        #Later code should throw error
    else:
        images_dir = imagespath
    
    #Grab all the files in the images dir
    list_of_images = os.listdir(images_dir)

    #Change currenmt working directory (cwd) to the images dir
    os.chdir(images_dir)

    # Itterate over all the files in the images dir
    for orignal_image in list_of_images:
        #If the current file is a png or jpg
        if(orignal_image.lower().endswith(('.png', '.jpg', 'jpeg'))):
            
            # Make a copy of the image
            image = orignal_image
            
            # #Open the image with exif TRY WITH PIL AND IMAGEmagick
            # with open(image, 'rb') as unstripped_image_file:
            #     stripped_image = exifImage(unstripped_image_file)


            stripped_image = cv2.imread(image)
            # Image.open(image)

            #Strip out all the meta data
            # stripped_image.delete_all()

            #Create a new folder for the curent image and change the cwd to the new image folder
            image_dir = image.split('.')[0]
            os.mkdir(image_dir)
            os.chdir(image_dir)

            

            # Convert the image to grayscale
            gray = cv2.cvtColor(stripped_image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = faceDetector(gray)

            # Loop over the detected faces
            for face in faces:
                x, y = face.left(), face.top()
                x1, y1 = face.right(), face.bottom()

                # Extract the face region from the image
                face_region = stripped_image[y:y1, x:x1]

                # Apply Gaussian blur to this face region
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)

                # Replace the original image's face region with the blurred face
                stripped_image[y:y1, x:x1] = blurred_face

            cv2.imwrite(filename=image,img=stripped_image)
            # #Save the stripped image in the new dir
            # with open(image, 'wb') as stripped_image_file:
            #     stripped_image_file.write(stripped_image.tobytes())

            
            #If object recogniton is selected
            if objR:
                #Create dir for all mask images
                masks_dir = "masks_"+image_dir
                os.mkdir(masks_dir)

                #Perform PS on the image
                detector.onImage(image,masks_dir)

            #If text recogntion is selected
            if txtR:
                #Create array to hold text from the masks
                txt_data = []
                # txt = {
                #             "image": image,
                #             #Grab text from the image
                #             "text": txt_recog(image,0.5)      
                # }
                # txt_data.append(txt)


                #If object recognition has been performed check the masks dir
                if objR:
                    mask_images = os.listdir(masks_dir)
                    os.chdir(masks_dir)

                    #For each mask image grab text
                    for mask_image in mask_images:
                        txt = {
                                "image": mask_image,
                                "text": txt_recog(mask_image,0.5)      
                        }
                        txt_data.append(txt)
                
                #Write the ext to a json file
                with open('image_text.json', 'w') as f:
                        json.dump(txt_data, f)
                os.chdir("..")

            #Check json for people and do skeletal
            if sktlR:
                sktl_data = []

                #Go through all images looking for people
                sktl = {
                        "image": image, 
                        "skeltal_data" : skeltal_recog(image,"mobilenet_thin")       
                }
                sktl_data.append(sktl)

                if objR:
                    #Main image
                    # skeltal_recog(image,model="mobilenet_thin")
                    #Use person image masks
                    mask_images = os.listdir(masks_dir)
                    os.chdir(masks_dir)

                    #For each mask image grab text
                    for mask_image in mask_images:
                        if(mask_image.split('_')[0] == "person"):

                            sktl = {
                                    "image": mask_image, 
                                    "skeltal_data" : skeltal_recog(mask_image,"mobilenet_thin")       
                            }
                            sktl_data.append(sktl)
                
                #Write the ext to a json file
                with open('image_skeltal.json', 'w') as f:
                        json.dump(sktl_data, f)
                os.chdir("..")

                if objR:
                    object_json = 'panoptic_predictions.json'
                else:
                    object_json = None

                if txtR:
                    if objR:
                        text_json = masks_dir + "/image_text.json"
                       
                    else:
                        text_json = "/image_text.json"
                else:
                    text_json = None
                if sktlR:
                    if objR:
                        skeltal_json = masks_dir + "/image_skeltal.json"
                    else:

                        skeltal_json = "/image_skeltal.json"
                else:
                    skeltal_json = None

                feature_dataframe = create_df(object_json,skeltal_json,text_json)

                feature_dataframe.to_json(image.split('.')[0]+"_feature_df.json")



            #Go back to the images dir
            os.chdir("..")

        #The current photo isnt a png or jpg
        else:
            print(orignal_image + " must be of type .png or .jpg")
            raise(Exception)
        

if __name__ == '__main__':

    args = get_arguments()

    main(imagespath=args['images'], objR=args['object'],txtR=args['text'],sktlR=args['skeletal'])
