import torch
import os
import torchvision
import json

from torchvision.io import read_image

from ShapeDataset import *
from implementations import *

import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

root = './' # Folder where the saved model, and the image (Data) folder is located, and where the predictions will be saved
img_dir = os.path.join(root, 'Data') # Folder with png/jpg images to analyze. Should only contain jpg and/or png images, and no annotation/other files
pred_img_dir = os.path.join(root,'PredImg')

loadmodel = torch.load('./modelsave')
classes = ['Background', 'Cube', 'Octahedron', 'Sphere']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

images = list(sorted(os.listdir(img_dir)))

predictions = {'pred_boxes': [],
               'pred_masks': [],
               'pred_labels': [],
               'instances': [],
               'images': images
    }             

for i in range(len(images)):
    imgSavefile = os.path.join(pred_img_dir,images[i])

    pred_boxes_i, pred_masks_i, pred_labels_i, instances_i = predict(loadmodel,os.path.join(img_dir,images[i]), classes, img_savefile=imgSavefile)
    
    predictions['pred_boxes'].append(pred_boxes_i)
    predictions['pred_masks'].append(pred_masks_i)
    predictions['pred_labels'].append(pred_labels_i)
    predictions['instances'].append(instances_i)

print(classes)
print(np.array(predictions['instances']))



