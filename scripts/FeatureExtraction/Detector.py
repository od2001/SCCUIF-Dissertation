# Package Import

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import json
import numpy as np
from PIL import Image

### --- Code From : https://www.youtube.com/watch?v=Pb3opEFP94U --- ###

class Detector:
    def __init__(self,model_type):
        self.cfg = get_cfg()
        self.moodel_type = model_type
        
    
        if model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "PS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

### --- Code From : https://www.youtube.com/watch?v=Pb3opEFP94U --- ###
        
    #Perform object detection/segmentation on image
    def onImage(self, imagePath,masks_path):
        #Read Image to np array
        image = cv2.imread(imagePath)
        #If model type is not panoptic segementation
        if self.moodel_type != "PS":
            # Create predictions for specifed image
            predictions = self.predictor(image)
            
            #Initializes a visualizer with BGR converted image, dataset metadata, and sets instance visualization mode to black and white
            viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            instance_mode = ColorMode.IMAGE_BW)

            #Get models predictions, using CPU.
            direct_outputs = predictions["instances"].to("cpu")

            #Draws model's predictions on the image.
            output = viz.draw_instance_predictions(direct_outputs)

            #Get the metadata catalog for this configuration
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            #Get the class list for 'things'
            class_names = metadata.thing_classes

            #Create an array for the data to be written to the json
            data_to_save = []
            #For each of the outputs (predictions)
            for i in range(len(direct_outputs)):
                #Create a predictions dict with the bounding box, label and score
                pred = {
                    "bbox": direct_outputs.pred_boxes.tensor.numpy()[i].tolist(),
                    "label": class_names[direct_outputs.pred_classes[i].item()],
                    "score": direct_outputs.scores[i].item(),
                }
                #Appepnd the predictions to the array
                data_to_save.append(pred)

            # Save the array to a JSON file
            with open('predictions.json', 'w') as f:
                json.dump(data_to_save, f)

        # If model selected is panoptic segmentation
        else:
            # Create predictions for specifed image, using panoptic segmenation
            predictions, segementInfo = self.predictor(image)["panoptic_seg"]
            # Initializes a visualizer with BGR converted imagea and dataset metadata
            viz = Visualizer(image[:,:,::-1],MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            # Draws model's predictions on the image.
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"),segementInfo)
            # Get the metadata catalog for this configuration
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            # Get the class list for 'things'
            thing_classes = metadata.thing_classes
            # Get the class list for 'stuff'
            stuff_classes = metadata.stuff_classes

            #Create an array for the data to be written to the json
            data_to_save = []

            # For each of the segements detected
            for segment_info in segementInfo:
                #If the current segement is a part of the 'thing' class
                if segment_info["isthing"]:  # Check if the segment is a "thing"
                    # ID of the the current segement from the 'thing' class
                    category_id = segment_info["category_id"]
                    # Human label for the current 'thing' i.e person
                    label = thing_classes[category_id] 
                    # Unqiue id for this 'stuff'
                    segment_id = segment_info["id"]


                    # Create a mask for the current segment
                    mask = predictions == segment_id
                    mask_np = np.array(mask.cpu(), dtype=np.uint8)  # Mask with 1s for the object

                    # Find contours in the mask
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        # Compute the bounding box for the contour
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        # Bounding box as (x, y, w, h)
                        bbox = [x, y, x+w, y+h]  # Format: [x_min, y_min, x_max, y_max]

                    # Use the mask to extract the object from the original image
                    # Converting to rgb since that was the orginal
                    orignal_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #Create the mask image, removing all pixels 
                    object_image = cv2.bitwise_and(orignal_image, orignal_image, mask=mask_np)

                    # Save the extracted object as an image
                    object_image_pil = Image.fromarray(object_image)
                    object_image_filename = f"{label}_{segment_id}.png"
                    object_image_pil.save(masks_path+"/"+object_image_filename)

                    # Save the data including the object image filename
                    data = {
                        "label": label,
                        "image": object_image_filename,
                        "bbox": bbox
                    }
                    data_to_save.append(data)
                #The current segement must be a part of the 'stuff' class
                else:
                    # ID of the the current segement from the 'thing' class
                    category_id = segment_info["category_id"]
                    # Human label for the current 'stuff' i.e sky
                    label = stuff_classes[category_id] 
                    # Unqiue id for this 'thing'
                    segment_id = segment_info["id"]

                    # Create a mask for the current segment
                    mask = predictions == segment_id
                    mask_np = np.array(mask.cpu(), dtype=np.uint8)  # Mask with 1s for the object

                    # Use the mask to extract the object from the original image
                    # Converting to rgb since that was the orginal
                    orignal_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #Create the mask image, removing all pixels 
                    object_image = cv2.bitwise_and(orignal_image, orignal_image, mask=mask_np)

                    # Save the extracted object as an image
                    object_image_pil = Image.fromarray(object_image)
                    object_image_filename = f"{label}_{segment_id}.png"
                    object_image_pil.save(masks_path+"/"+object_image_filename)

                    # Save the data including the object image filename
                    data = {
                        "label": label,
                        "image": object_image_filename,
                        "bbox" : [-1]
                    }
                    data_to_save.append(data)

            #Order the data by the x values in the bounding boxes
            data_to_save = sorted(data_to_save,key=lambda item: item['bbox'][0])


            #Create an id for each of the entries with a bbox 
            counter = 0
            for i in data_to_save:
                if i['bbox'][0] == -1:
                    i["idNo"] = -1
                else:
                    i["idNo"] = counter
                    counter += 1


            # Save to a JSON file
            with open('panoptic_predictions.json', 'w') as f:
                json.dump(data_to_save, f)

        # # Display the results
        # cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Result', 600, 400)
        # cv2.imshow("Result", output.get_image()[:,:,::-1])
        # cv2.waitKey(0)   
        # cv2.destroyAllWindows()  