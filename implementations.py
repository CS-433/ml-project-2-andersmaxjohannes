
import torch
import torchvision
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


def apply_nms(orig_prediction, iou_thresh=0.3):
    ''' Function for reducing the number of bounding boxes/masks that are on top of each other'''
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    final_prediction['masks'] = final_prediction['masks'][keep]
    
    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchvision.transforms.ToPILImage()(img).convert('RGB')

def thresholdForBoundingBoxes(pred, threshold=0.1):
    keep = torch.gt(pred['scores'],threshold)

    thresholdedPredictions = pred
    thresholdedPredictions['boxes'] = pred['boxes'][keep]
    thresholdedPredictions['scores'] = pred['scores'][keep]
    thresholdedPredictions['labels'] = pred['labels'][keep]
    thresholdedPredictions['masks'] = pred['masks'][keep]

    return thresholdedPredictions

def predict(model, image_path, classes, box_thresh=0.1, iou_thresh=0.3, printout=False, show_img=False, img_savefile=None, device=None):
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

    thres_pred = thresholdForBoundingBoxes(pred, threshold=box_thresh) # Removing instances the model is too unsure of
    nms_pred = apply_nms(thres_pred, iou_thresh=iou_thresh) # Removing boxes with too much overlap

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"{label},{classes[label]}: {score:.3f}" for label, score in zip(nms_pred["labels"], nms_pred["scores"])]
    pred_boxes = nms_pred["boxes"].long()
    pred_masks = (nms_pred["masks"] > 0.7).squeeze(1)

    instances = countInstances(classes,pred_labels,printout)

    if show_img:
        output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
        output_image = draw_segmentation_masks(output_image, pred_masks, alpha=0.5, colors='blue')

        plt.figure(figsize=(9, 9))
        plt.imshow(output_image.permute(1, 2, 0))
        plt.show()
        plt.close()
    
    if img_savefile != None:
        output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
        output_image = draw_segmentation_masks(output_image, pred_masks, alpha=0.5, colors='blue')

        plt.figure(figsize=(9, 9))
        plt.imshow(output_image.permute(1, 2, 0))
        plt.savefig(img_savefile + '.png')
        plt.close()

    return pred_boxes, pred_masks, pred_labels, instances