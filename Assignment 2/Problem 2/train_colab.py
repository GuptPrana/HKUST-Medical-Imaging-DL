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

# from utils.metric import mean_class_recall
import dataset
# from losses import NCELoss
# from loss import class_balanced_loss


# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation
# Random Flip, Rotate and Crop 
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                    transforms.RandomVerticalFlip(), 
                    transforms.RandomRotation(45),
                    transforms.RandomAffine(180, translate=[0.2, 0.2], scale=[0.7, 1.3]),
                    transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                    transforms.RandomResizedCrop(224),
                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize(224),
                    transforms.ToTensor()])

# Set up Dataset
trainset = dataset.Skin7(train=True, transform=train_transform)
testset = dataset.Skin7(train=False, transform=test_transform)

# Iterable loader
batch_size = 42
num_workers = 4
# Convert PIL to tensor?
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, pin_memory=True,
                                        num_workers=num_workers)

# Resnet50
model = models.resnet18(pretrained=True)
# Freeze the first 6 layers out of 10 in Resnet50 

layer = 0
for child in model.children():
    layer += 1
    if layer < 7:
        # print(child)
        for params in child.parameters():
            params.requires_grad = False
    else:
        break

model.fc = nn.Linear(model.fc.in_features, 7)
model = model.to(device)
print(model)

# Loss Functions
# nce = NCELoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)

# Optmizer
# Learning Rate Scheduler - Can set amsgrad=True
learn_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learn_rate, amsgrad=True)
# optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                optimizer, mode='min', factor=0.1, patience=50, verbose=False,
#                threshold=3e-4)


tracc = []
trloss = []
teacc = []

# Logging all information, log_dir=
#writer = SummaryWriter() #close after running 
#train(model, trainloader, testloader)
#writer.close()

# Train 
def train(model, trainloader, testloader):
    Max_epoch = 80
    # Epoch
    for epoch in range(1, Max_epoch+1):
        print(f'Epoch {epoch}/{Max_epoch}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_correct = 0

        # for batch_idx, ([data,data_aug], target) in tqdm(enumerate(trainloader)):
        for batch_idx, (data, target) in tqdm(enumerate(trainloader)):
            # Train
            optimizer.zero_grad() 
            images, labels = data, target
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # Step
            loss = criterion(outputs, labels)
            # nce loss =
            loss.backward()
            optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            running_loss += loss.item()
            running_correct += torch.sum(pred == labels)

        # Save data for plotting curves
        train_loss = running_loss/len(trainset)
        # scheduler.step(train_loss)
        train_acc = running_correct/len(trainset)
        # Can add mean class recall
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        print("=> Epoch:{} - train acc: {:.4f}".format(epoch, train_acc))
        print("loss:{:.4f} ".format(loss))
        tracc.append(train_acc.cpu())
        trloss.append(train_loss)
        # Run Test for this Epoch
        test(model, testloader, epoch) 


def test(model, testloader, epoch):
    # test mode
    model.eval()
    y_true = []
    y_pred = []

    # for _, ([data,data_aug], target) in enumerate(testloader):
    for _, (data, target) in enumerate(testloader):
        images, labels = data, target
        images = images.to(device)
        # labels = labels.to(device)
        outputs = model(images)
        pred = torch.argmax(outputs, dim=1)
        pred = pred.cpu()
        # labels = labels.cpu()
        # .extend
        y_pred.extend(pred)
        y_true.extend(target)
    
    test_acc = accuracy_score(y_true, y_pred)
    # Mean class recall
    writer.add_scalar("Acc/Test", test_acc, epoch)
    # writer.add_scalar("MCR/Test", , epoch)
    print("=> Epoch:{} - val acc: {:.4f}".format(epoch, test_acc))
    teacc.append(test_acc)

# Logging all information, log_dir=
writer = SummaryWriter() #close after running 
train(model, trainloader, testloader)
print(tracc)
print(trloss)
print(teacc)
writer.flush()
writer.close()