# Dataset for Resnet50 on INDIVIDUAL FRAMES

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class CholecFrame(Dataset): 
    # Cholec80 Videos
    def __init__(self, root="./datas", train=True, transform=None, video=0):
        self.train = train
        # video number: 1-5 train, 41 test
        self.video = str(video)
        if(self.train):
            # Annotations Path
            lapath = "./datas/annotation/video_{vid}.csv".format(vid=self.video)
            self.labelpath = os.path.join(lapath)
            # Frames Path
            frpath = "./datas/{vid}".format(vid=self.video)
            self.root = os.path.join(frpath)
        else:
            self.labelpath = os.path.join("./datas/annotation/video_41.csv")
            self.root = os.path.join("./datas/41")
        # Transforms
        self.transform = transform


    def __getitem__(self, index):
        # .iloc to locate the image file, class index
        # Image.open from PIL; .jpg cannot be opened by read_image() 
        csv = pd.read_csv(self.labelpath)
        # Remove axes
        data = csv#.values()
        label = data.iloc[index, 1]
        image = Image.open(os.path.join(self.root, data.iloc[index, 0]))
        # with open(os.path.join(self.root, data.iloc[index, 0]), "rb") as afile:
        #     image = Image.open(afile)
        #     image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)    
        return (image, label)

    def __len__(self):
        csv = pd.read_csv(self.labelpath)
        return len(csv)

