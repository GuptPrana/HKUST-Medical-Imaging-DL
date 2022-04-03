# Train LSTM Model with Pretrained Resnet50 for Feature Extraction 

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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm
import dataset_lstm as dataset
from model_lstm import R50LSTM

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

# Set up Dataset * Videos have different lengths
trainset1 = dataset.CholecLSTM(train=True, transform=train_transform, video=1)
trainset2 = dataset.CholecLSTM(train=True, transform=train_transform, video=2)
trainset3 = dataset.CholecLSTM(train=True, transform=train_transform, video=3)
trainset4 = dataset.CholecLSTM(train=True, transform=train_transform, video=4)
trainset5 = dataset.CholecLSTM(train=True, transform=train_transform, video=5)
testset = dataset.CholecLSTM(train=False, transform=test_transform)

# Iterable loader
batch_size = 20
num_workers = 8
test_path = "test.pth"
train_path = "train.pth"

# Dataloaders
trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)
trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)
trainloader3 = torch.utils.data.DataLoader(trainset3, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)
trainloader4 = torch.utils.data.DataLoader(trainset4, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)
trainloader5 = torch.utils.data.DataLoader(trainset5, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)                                                                                                                                          
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, pin_memory=True,
                                        num_workers=num_workers)

# LSTM model

model = R50LSTM().to(device)
print(model)

valacc = 0.0
criterion = nn.CrossEntropyLoss().to(device)
learn_rate = 5e-4
trainloader = trainloader1
optimizer = optim.Adam(model.parameters(), learn_rate, weight_decay=1e-5)

# Train
def train(model, trainloader, testloader, optimizer, valacc, learn_rate):
    Max_epoch = 80
    # Epoch
    for epoch in range(1, Max_epoch+1):
        print(f'Epoch {epoch}/{Max_epoch}')
        print('-' * 15)
        # Adam optimizer with Weight decay
        if epoch % 10 == 0:
            learn_rate = learn_rate * 0.5
            optimizer = optim.Adam(model.parameters(), learn_rate, weight_decay=1e-5)
        
        model.train()
        running_loss = 0.0
        running_correct = 0

        # Different trainloaders for different datasets
        vid = epoch % 5 + 1
        if vid == 1:
            trainset = trainset1
            trainloader = trainloader1
        elif vid == 2:
            trainset = trainset2
            trainloader = trainloader2
        elif vid == 3:
            trainset = trainset3
            trainloader = trainloader3
        elif vid == 4:
            trainset = trainset4
            trainloader = trainloader4
        else:
            trainset = trainset5
            trainloader = trainloader5
        
        for batch_idx, (data, target) in tqdm(enumerate(trainloader)):
            # Train
            optimizer.zero_grad() 
            images, labels = data, target
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # Step
            loss = criterion(outputs, labels)
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
        if epoch % 20 == 0:
            torch.save(model.state_dict(), train_path)
        # Run Test for this Epoch
        test(model, testloader, epoch, valacc) 

def test(model, testloader, epoch, valacc):
    # test mode
    model.eval()
    y_true = []
    y_pred = []

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
    test_pres = precision_score(y_true, y_pred, average="micro")
    test_rec = recall_score(y_true, y_pred, average="micro")
    if test_acc > valacc:
        valacc = test_acc
        torch.save(model.state_dict(), test_path)
    # Mean class recall
    writer.add_scalar("Acc/Test", test_acc, epoch)
    writer.add_scalar("Prec/Test", test_pres, epoch)
    writer.add_scalar("Rec/Test", test_rec, epoch)
    # writer.add_scalar("MCR/Test", , epoch)
    print("=> Epoch:{} - val acc: {:.4f}".format(epoch, test_acc))

# Logging all information, log_dir=
writer = SummaryWriter() #close after running 
train(model, trainloader, testloader, optimizer, valacc, learn_rate)
writer.flush()
writer.close()

