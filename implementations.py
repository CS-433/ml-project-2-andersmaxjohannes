import numpy as np
import torch
from ShapeDataset import *


def custom_collate(data):
  return data



def getTrainVal(images, masks):
  num = int(0.9 * len(images))
  num = num if num % 2 == 0 else num + 1
  if (num - int(0.9 * len(images)) <= 1):
    num-=1
  train_imgs_inds = np.random.choice(range(len(images)) , num , replace = False)
  val_imgs_inds = np.setdiff1d(range(len(images)) , train_imgs_inds)

  train_imgs = np.array(images)[train_imgs_inds]
  val_imgs = np.array(images)[val_imgs_inds]
  train_masks = np.array(masks)[train_imgs_inds]
  val_masks = np.array(masks)[val_imgs_inds]

  return train_imgs, val_imgs, train_masks, val_masks


def get_dataloader(images , masks, root, set_num_workers):
    return torch.utils.data.DataLoader(
        ShapeDataset(images , masks, root) ,
        batch_size = 2 ,
        shuffle = True ,
        collate_fn = custom_collate,
        num_workers = set_num_workers ,
        pin_memory = True if torch.cuda.is_available() else False
    )