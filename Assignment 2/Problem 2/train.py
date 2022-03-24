"""Trainer

    Train all your model here.
"""

import torch
import torch.optim as optim
import torchvision.models as models
import os
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils.metric import mean_class_recall

import dataset

from losses import NCELoss
# from loss import class_balanced_loss


# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation
# Random Flip, Rotate and Crop 
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                    transforms.RandomVerticalFlip(), 
                    transforms.RandomRotation(90),
                    transforms.RandomResizedCrop(224)])
test_transform = None

# Set up Dataset
trainset = dataset.Skin7(train=True, transform=train_transform)
testset = dataset.Skin7(train=False, transform=test_transform)

batch_size = 24
num_workers = 4
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, pin_memory=True,
                                        num_workers=num_workers)

# Loss Functions
# nce = NCELoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)

# Resnet50 (pass pretrained=True)
model = models.resnet50()

# Optmizer; Learning Rate
learn_rate = 1e-3
optmizer = optim.Adam(model.parameters(), lr=learn_rate)

# Train 
Max_epoch = 50
def train(model,trainloader):
    # load = iter(trainloader)
    model = model.to(device)
    model.train()

    for epoch in range(1, Max_epoch+1):
        
       
        for batch_idx, ([data,data_aug], target) in tqdm(enumerate(trainloader)):
            pass
   
def test(model, testloader, epoch):
    model.eval()

    y_true = []
    y_pred = []
    for _, ([data,data_aug], target) in enumerate(testloader):
       pass

    acc = accuracy_score(y_true, y_pred)
    print("=> Epoch:{} - val acc: {:.4f}".format(epoch, acc))
    
    
