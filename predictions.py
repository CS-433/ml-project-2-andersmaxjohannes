import torch
import os
import torchvision
import json

from torchvision.io import read_image

from ShapeDataset import *
from implementations import *

import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


###### Paramters #######
root = './' # Folder where the saved model, and the image folder is located, and where the predictions will be saved
img_dir = os.path.join(root, 'Images') # Folder with png/jpg images to analyze. Should only contain jpg and/or png images, and no annotation/other files
pred_img_dir = os.path.join(root,'PredImg')

modelname = 'max_preprocess'

classes = ['Background', 'Cube', 'Octahedron', 'Sphere']

box_thresh=0.2  # The minimum confidence level for a bounding box to be included as a prediction. In the range [0,1]
iou_thresh=0.3  # A measure of how much overlap between bounding boxes is allowed. In the range [0,1]
printout=False  # If the predicted number of each class should be printed to screen as the predictions are made
show_img=False  # If each image with predictions should be displayed as it is predicted
save_images=True    # If the images should be saved to pred_img_dir
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # The device to be used for predictions - Now it is set to the gpu if available, if not, to the cpu

loadmodel = torch.load(os.path.join('./', modelname), map_location=torch.device(device))

############


images = list(sorted(os.listdir(img_dir)))
predictions = {'pred_boxes': [],
               'pred_masks': [],
               'pred_labels': [],
               'instances': [],
               'images': images
    }             

for i in range(len(images)):
    if save_images:
        imgSavefile = os.path.join(pred_img_dir,f'Pred_model_{modelname}_img{i}')
    else: 
        imgSavefile = None

    pred_boxes_i, pred_masks_i, pred_labels_i, instances_i = predict(loadmodel,os.path.join(img_dir,images[i]), classes, box_thresh=box_thresh, printout=printout, show_img=show_img, img_savefile=imgSavefile, device=device)
    
    predictions['pred_boxes'].append(pred_boxes_i)
    predictions['pred_masks'].append(pred_masks_i)
    predictions['pred_labels'].append(pred_labels_i)
    predictions['instances'].append(instances_i)

print(classes)
print(np.array(predictions['instances']))



