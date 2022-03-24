# Customize Dataset

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Skin7(Dataset):
    """Skin Lesion"""
    # root=datas or change data folder
    def __init__(self, root="./datas", train=True, transform=None):
        self.root = os.path.join(root)
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        """
        # Image and Annotation Paths

        if(self.train):
            labelpath = os.path.join("./datas/annotation/train.csv")
        else:
            labelpath = os.path.join("./datas/annotation/test.csv") 
        
        labels = pd.read_csv(labelpath) 
        imagepath = os.path.join("./datas/images")

        
        # Read csv and iterate
        # 
        #         Args:
            index (int): Index
            image of index
            class of index
            tuple
        Returns:
            tuple: (sample, target) where target is class_index of the
                   target class.
        """
        pass

    def __len__(self):
        return len(self.data)

   

