import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

"""
Args:
    index (int): Index
    image of index
    class of index
    tuple
Returns:
    tuple: (sample, target) where target is class_index of the
            target class.
"""

class Skin7(Dataset): 
    """Skin Lesion"""
    def __init__(self, root="./datas/images", train=True, transform=None):
        # Image Path
        self.root = os.path.join(root)
        # Label Path
        self.train = train
        if(self.train):
            self.labelpath = os.path.join("./datas/annotation/train.csv")
        else:
            self.labelpath = os.path.join("./datas/annotation/test.csv")
        # Transforms
        self.transform = transform

    def __getitem__(self, index):
        # .iloc to locate the image file, class index
        # Image.open from PIL; .jpg cannot be opened by read_image() 
        csv = pd.read_csv(self.labelpath)
        # Remove axes
        data = csv#.values()
        label = data.iloc[index, 1]
        # image = Image.open(os.path.join(self.root, data.iloc[index, 0]))

        with open(os.path.join(self.root, data.iloc[index, 0]), "rb") as afile:
            image = Image.open(afile)
            image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return (image, label)

    def __len__(self):
        csv = pd.read_csv(self.labelpath)
        return len(csv)