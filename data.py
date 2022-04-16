# Dataset
import torch
import os 
import cv2
import pandas as pd
import numpy as np
import collections
import random
import scipy.ndimage as sp
import torchvision 

def load_video(filename: str, num_frames: int):
    # Check for file
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    cap = cv2.VideoCapture(filename)
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Empty array 
    video = np.zeros((num_frames, 112, 112, 3), np.uint8)
    start_frame = random.randint(0, max_frames-num_frames)
    # Add grayscale frame to video container
    for i in range(num_frames):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video[i, :, :] = frame
    # Formatting
    # video = video[start_frame, start_frame+num_frames]
    video = video.transpose((3, 0, 1, 2))
    return video

class EchoNet(torch.utils.data.Dataset):
    def __init__(self, root=None, split="train", num_frames=64, transform=None):
        super().__init__(root, transform=transform)        
        self.root = root
        if (root == None):
            self.root = "./SUB_Echo/subvideos/"
        self.root = root
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        # Label path
        if (self.split == "train"):
            self.list = os.path.join("./SUB_Echo/FileListSUB_Train.csv")
        elif (self.split == "val"):
            self.list = os.path.join("./SUB_Echo/FileListSUB_Val.csv")
        else:
            self.list = os.path.join("./SUB_Echo/FileListSUB_Test.csv")
        
        
    def __getitem__(self, index):
        csv = pd.read_csv(self.list)
        EF = csv.iloc[index, 1]
        filename = csv.iloc[index, 0] + ".avi"
        filepath = os.path.join(self.root, filename)
        # Load video
        video = load_video(filepath, 64).astype(np.float32)
        if self.transform is not None:
            video = self.transform(video)
        # Video and Ejection Fraction returned
        return video, EF

    def __len__(self):
        csv = pd.read_csv(self.list)
        return len(csv)
