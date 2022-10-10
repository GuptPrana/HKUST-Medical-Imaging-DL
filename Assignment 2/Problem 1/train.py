import torch
import torch.optim as optim
import torchvision.models as models
import os
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import scipy as sp
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm

from model import Model
from dataset import LAHeart
from torch.utils.data import DataLoader
from loss import DiceBCELoss

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
# Data Augmentation 
# Random Flip, Rotate and Crop
# Cannot use transforms because 3D 
train_transform = transforms.Compose([
                    transforms.RandomAffine(degrees=180, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                    transforms.ColorJitter(brightness=0.01, contrast=0.01),#saturation=0.1)
                    transforms.ToTensor()])
test_transform = transforms.ToTensor()
'''
# 3D Transforms
train_transform=None # Scipy ndimage rotate, affine
# test_transform=None
# Set up Dataset
trainset = LAHeart(transform=train_transform)
# testset = LAHeart(split="test", transform=test_transform, crop=False)

# Iterable loader
batch_size = 1
chwd = 112*112*80
num_workers = 8
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)

# Model
unet = Model()
unet = unet.to(device)
# Check for saved models
train_path = "train.pth"

# Loss Functions
criterion = DiceBCELoss().to(device)

# Keep slices
# copy = np.zeros((112, 112)).unqueeze(0).unsqueeze.(0)
# copy = copy.transforms.toTensor()
# copy = copy.to(device)

# Optmizer
learn_rate = 3e-4
optimizer = optim.Adam(unet.parameters(), lr=learn_rate, weight_decay=2e-5)

# Train 
def train(model, trainloader):
    Max_epoch = 120
    # Epoch
    for epoch in range(1, Max_epoch+1):
        print(f'Epoch {epoch}/{Max_epoch}')
        print('-' * 20)

        model.train()
        running_loss = 0.0
        running_dice = 0

        # for batch_idx, ([data,data_aug], target) in tqdm(enumerate(trainloader)):
        for batch_idx, (data, target) in tqdm(enumerate(trainloader)):
            # Train
            optimizer.zero_grad() 
            images, labels = data, target.float()
            images = images.to(device)
            labels = labels.to(device)
            outputs = unet(images)
            # Step
            loss, dice = criterion(outputs, labels, batch_size, chwd)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice.item()

        # Save data for plotting curves
        train_loss = running_loss/len(trainset)
        # scheduler.step(train_loss)
        train_dice = running_dice/len(trainset)
        # log
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Dice/Train", train_dice, epoch)
        print("=> Epoch:{} - train dice: {:.4f}".format(epoch, train_dice))
        print("loss:{:.4f} ".format(train_loss))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), train_path)
   
            
# Logging all information, log_dir=
writer = SummaryWriter() #close after running 
train(unet, trainloader)
writer.flush()
writer.close()
