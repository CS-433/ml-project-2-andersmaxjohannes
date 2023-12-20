from typing import Any, Tuple
import numpy as np
import os
from PIL import Image

import torch
import torchvision
from torchvision import transforms as T


import os
import torch.utils.data
from pycocotools.coco import COCO



class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self , images , masks, root, transforms = None):
        self.imgs = images
        self.masks = masks
        self.transforms = transforms
        self.root = root
        

    def __getitem__(self , idx):
        img  = Image.open(os.path.join(self.root, "images", self.imgs[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.root, "masks", self.masks[idx]))
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        masks = np.zeros((num_objs , mask.shape[0] , mask.shape[1]))
        for i in range(num_objs):
            masks[i][mask == i+1] = True
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin , ymin , xmax , ymax])
        boxes = torch.as_tensor(boxes , dtype = torch.float32)
        labels = torch.ones((num_objs,) , dtype = torch.int64) # TODO reflect the actual class
        masks = torch.as_tensor(masks , dtype = torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        return T.ToTensor()(img) , target

    def __len__(self):
        return len(self.imgs)
    
class ShapeDataset2(torchvision.datasets.CocoDetection):
    def __init__(self, root, feature_extractor, train=True):
        ann_file = os.path.join(root, "_annotations.coco.json")
        super(ShapeDataset2, self).__init__(root, ann_file)
        self.feature_extractor = feature_extractor

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(ShapeDataset2, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        #print(target["annotations"][0]["bbox"])

        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        num_objs = len(coco_annotation)
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        #boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target["annotations"][0]["bbox"] = boxes

        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target












class ShapeDataset3(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(os.path.join(root,annotation))
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = self.coco.loadAnns(ann_ids)
        # path for input image
        path = self.coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = torchvision.tv_tensors.Image(img)

        # number of objects in the image
        num_objs = len(coco_annotation)
        #print(coco_annotation)
        
        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes)

        # Labels
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = np.zeros(num_objs,dtype=np.int64)
        for i in range(num_objs):
            labels[i] = coco_annotation[i]["category_id"]
        labels = torch.as_tensor(labels)        

        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # Segmentation
        masks = []
        for seg in coco_annotation:
            masks.append(self.coco.annToMask(seg))
        masks = torch.as_tensor(masks)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes      # I think this one is good
        my_annotation["labels"] = labels
        my_annotation["image_id"] = index
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation["masks"] = masks     # I think this one is good


        # Debugging prints
        #print(my_annotation)

        if self.transforms is not None:
            image = self.transforms(image)

        return (image, my_annotation)

    def __len__(self):
        return len(self.ids)
    
# ,transform=torchvision.transforms.PILToTensor

class CustCoco(torchvision.datasets.CocoDetection):
    def __init__(self,root, ann_file, transforms=None):
        super().__init__(root, ann_file, transforms)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(index)
        img = torchvision.transforms.PILToTensor()(image)
        return img, target

