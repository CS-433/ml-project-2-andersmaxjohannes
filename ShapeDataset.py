import numpy as np
import os
from PIL import Image

import torch
from torchvision import transforms as T



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
        labels = torch.ones((num_objs,) , dtype = torch.int64)
        masks = torch.as_tensor(masks , dtype = torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        return T.ToTensor()(img) , target

    def __len__(self):
        return len(self.imgs)