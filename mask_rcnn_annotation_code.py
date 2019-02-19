import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from samples.insect import insect


# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

 

config = insect.InsectConfig()
INSECT_DIR = os.path.join(ROOT_DIR, "insects")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
# Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Path to insect trained weights

INSECT_WEIGHTS_PATH = "./logs/insect20181219T1721/mask_rcnn_insect_0025.h5"  # TODO: update this path 

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


    # Load validation dataset
dataset = insect.InsectDataset()
dataset.load_insect(INSECT_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                            config=config)

# Set path to insect weights file

# Or, load the last model trained

# Load weights
print("Loading weights ", INSECT_WEIGHTS_PATH)
model.load_weights(INSECT_WEIGHTS_PATH, by_name=True)


# Pass the directory path where images are saved. Model will create bounding box and save it in the current directory.
def image_annot(path):
    for image_name in os.listdir(path):
        #print(image_name)
    	image = skimage.io.imread(os.path.join(path,image_name))
    	results = model.detect([image], verbose=1)
    	r = results[0]
    	try:
            visualize.save_image(image, image_name, r['rois'], r['masks'],r['class_ids'],r['scores'],dataset.class_names,scores_thresh=0.5,mode=1)
    	except:
        	continue


if __name__ == '__main__':
    import argparse


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Creating annotated images')

    parser.add_argument('--path', required=False,
                        metavar="/path/to/insect/dataset/",
                        help='Directory of the Insect images')

    args = parser.parse_args()


    image_annot(args.path)
