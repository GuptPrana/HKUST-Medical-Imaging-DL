import os
import torch
import h5py
import random
import numpy as np
from torchvision.transforms.transforms import ToTensor
import scipy.ndimage as sp

def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label

def Random3DCrop(image, label, dims=[112, 112, 80]):
    print(image.shape)
    h, w, d = image.shape
    # dims = output dimensions 
    # input > output
    assert h > dims[0]
    assert w > dims[1]
    assert d > dims[2]
    # crop index
    index_h = random.randint(0, h-dims[0])
    index_w = random.randint(0, h-dims[1])
    index_d = random.randint(0, h-dims[2])
    # slice image and label
    cropped_image = image[index_h:index_h+dims[0], index_w:index_w+dims[1], index_d:index_d+dims[2]]
    cropped_label = label[index_h:index_h+dims[0], index_w:index_w+dims[1], index_d:index_d+dims[2]]
    return cropped_image, cropped_label

class LAHeart(torch.utils.data.Dataset):
    def __init__(self, split="train", transform=False, crop=True):
        # Locations of all h5py data files
        self.list = os.path.join("problem1_datas/filenames.csv")
        self.split = split
        self.transform = transform
        self.RandomCrop = crop

    def __len__(self):
        # 14 train images, 20 test images
        if self.split == "train":
            return 14
        else:
            return 20

    def __getitem__(self, index):
        csv = pd.read_csv(self.list)
        if self.split == "train":
            filename = csv.iloc[index, 0]
            path = os.path.join("problem1_datas/train/{}".format(filename))
        else:
            filename = csv.iloc[index, 1]
            path = os.path.join("problem1_datas/test/{}".format(filename))
        image, label = read_h5(path)
        # Need cropping because image sizes are not fixed
        if self.RandomCrop:
            image, label = Random3DCrop(image, label)
            image = np.expand_dims(image, 0)
            label = np.expand_dims(label, 0)
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
        # if self.transform:
        #     rd = random.randint(-180, 180)
        #     image = sp.rotate(image, rd)
        #     label = sp.rotate(label, rd)
        return (image, label)
