import argparse
import logging
import sys
import time
sys.path.insert(0,"C:/Users/Ossia/Documents/SCCUIF/tf-pose-estimation")

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger("TfPoseEstimatorRun")
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def skeltal_recog(image,model):
    # parser = argparse.ArgumentParser(description="tf-pose-estimation run")
    # parser.add_argument("--image", type=str, default="./images/p1.jpg")
    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     default="cmu",
    #     help="cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small",
    # )
    # parser.add_argument(
    #     "--resize",
    #     type=str,
    #     default="0x0",
    #     help="if provided, resize images before they are processed. "
    #     "default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ",
    # )
    # parser.add_argument(
    #     "--resize-out-ratio",
    #     type=float,
    #     default=4.0,
    #     help="if provided, resize heatmaps before they are post-processed. default=1.0",
    # )

    # args = parser.parse_args()

   

    # Model's preferred input dimensions
    model_width = 656
    model_height = 368

    # estimate human poses from a single image !
    image = common.read_imgfile(image, None, None)

    # Original image dimensions
    og_height, og_width = image.shape[:2]

    # Calculate scale factors
    scale_by_width = model_width / og_width
    scale_by_height = model_height / og_height

    # Use the smaller scale factor to ensure the resized image fits within the model's dimensions
    scale_factor = min(scale_by_width, scale_by_height)

    # Calculate new dimensions
    new_width = int(og_width * scale_factor)
    new_height = int(og_height * scale_factor)
        # if w == 0 or h == 0:
    #     e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    # else:
    e = TfPoseEstimator(get_graph_path(model), target_size=(new_width, new_height))


    
    if image is None:
        logger.error("Image can not be read, path=%s" % image)
        sys.exit(-1)

    t = time.time()
    poses = e.inference(
        image, resize_to_default=(new_width > 0 and new_height > 0), upsample_size=4.0
    )
    
    
    # # Extract pose data
    # people_data = {}
    # for human in poses:



    #     human_data = {
    #         'body_parts': {}
    #     }
    #     for i in range(common.CocoPart.Background.value):
    #         if i not in human.body_parts.keys():
    #             continue

    #         min_x, min_y = float('inf'), float('inf')
    #         max_x, max_y = -float('inf'), -float('inf')

    #         body_part = human.body_parts[i]
    #         human_data['body_parts'][common.CocoPart(i).name] = {
    #             'x': body_part.x * width,
    #             'y': body_part.y * height,
    #         }
    #     people_data = {human : human_data}


    # return people_data


 # Extract pose data
    human_data = {}
    for human in poses:



        human_data = {
            'body_parts': {}
        }
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = -float('inf'), -float('inf')

            body_part = human.body_parts[i]
            human_data['body_parts'][common.CocoPart(i).name] = {
                'x': (body_part.x)* og_width,
                'y':(body_part.y) * og_height,
            }


    return human_data