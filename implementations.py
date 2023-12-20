
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# Helper function for data augmentation
def get_transform(train):
    transforms = []
    if train:
        transforms.append(v2.RandomHorizontalFlip(0.5))
    transforms.append(v2.ToDtype(torch.float, scale=True))
    transforms.append(v2.ToPureTensor())
    return v2.Compose(transforms)

def countInstances(classes,labels,printout=False):
    instances = np.zeros(len(classes),dtype=np.int64)
    for i in range(len(labels)):
        image_int, trash = labels[i].split(',')
        for j in range(1,len(classes)):
            if int(image_int)==j:
                instances[j] += 1

    if printout:
        print('The number of detected instances of each shape in the selected image are:')
        for i in range(1,len(classes)):
            print(f'{classes[i]}:{instances[i]}')
    return instances


def predict(model, image_path, classes, printout=False, show_img=False, img_savefile=None, device=None):
    if device == None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image = read_image(image_path)
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"{label},{classes[label]}: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    pred_masks = (pred["masks"] > 0.7).squeeze(1)

    instances = countInstances(classes,pred_labels,printout)

    if show_img or (img_savefile != None):        
        output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

        output_image = draw_segmentation_masks(output_image, pred_masks, alpha=0.5, colors='blue')

        plt.figure(figsize=(9, 9))
        plt.imshow(output_image.permute(1, 2, 0))

    if show_img:
        plt.show()
    
    if img_savefile != None:
        plt.savefig(img_savefile + '.png')

    if show_img or (img_savefile != None):
        plt.close()

    return pred_boxes, pred_masks, pred_labels, instances