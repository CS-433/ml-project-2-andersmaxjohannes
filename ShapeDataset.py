
from torchvision.datasets import CocoDetection
from torchvision.transforms import PILToTensor


class ShapeDataset(CocoDetection):
    def __init__(self,root, ann_file, transforms=None):
        super().__init__(root, ann_file, transforms)
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        img = PILToTensor()(image)
        return img, target