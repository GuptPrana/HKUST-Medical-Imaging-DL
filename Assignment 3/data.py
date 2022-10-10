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
    video = np.zeros((max_frames, 112, 112, 3), np.uint8)
    start_frame = random.randint(0, max_frames-num_frames)
    # Add grayscale frame to video container
    for i in range(max_frames):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video[i, :, :] = frame
    # Formatting
    sample = video[start_frame : start_frame+num_frames-1]
    sample = sample.transpose((3, 0, 1, 2))
    return sample

class EchoNet(torch.utils.data.Dataset):
    def __init__(self, root=None, split="train", num_frames=64, transform=None):    
        self.root = root
        if (root == None):
            self.root = "./SUB_Echo/subvideos/"
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
        EF = np.array([0])
        frame_count = csv.iloc[index, 7]
        # Skip videos with frames < self.num_frames
        if frame_count < self.num_frames:
          while frame_count < self.num_frames:
            index += 1
            frame_count = csv.iloc[index, 7]
        # Get videos
        EF[0] = csv.iloc[index, 1]
        filename = csv.iloc[index, 0] + ".avi"
        print(filename)
        filepath = os.path.join(self.root, filename)
        # Load video
        video = load_video(filepath, 64).astype(np.float32)
        if self.transform is not None:
            rd = random.randint(-45, 45)
            video = sp.rotate(video, rd, reshape=False, axes=(2,3))
        # Video and Ejection Fraction returned
        EF = torch.from_numpy(EF).float()
        return (video, EF)

    def __len__(self):
        # Manual because need to exclude videos with too little frames
        # csv = pd.read_csv(self.list)
        if self.split == "train":
          return 64
        if self.split == "val":
          return 32
        if self.split == "test":
          return 32
        # return len(csv)
