import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as Datasets
from data import EchoNet
from loss import MSLE
import numpy as np
import math
from tqdm import tqdm

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model
model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
train_path = "model.pth"
model = model.to(device)
# (Loading weights)
# checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
# model.load_state_dict(checkpoint['state_dict'])
# optim.load_state_dict(checkpoint['opt_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_dict'])

# Data
# Transforms
train_transform=None # Scipy ndimage rotate, affine, normalization
test_transform=None
val_transform=None
# Dataset
trainset = EchoNet(split="train", transform=train_transform)
#trainset = Datasets.MNIST(root='dataset/', train=True, download=True)
testset = EchoNet(split="test", transform=test_transform)
valset = EchoNet(split="val", transform=val_transform)
# Iterable loader
batch_size = 1
num_workers = 8
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, pin_memory=True,
                                         num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=True, pin_memory=True,
                                         num_workers=num_workers)

# Loss Functions 
mseloss = torch.nn.MSELoss().to(device)
maeloss = torch.nn.L1Loss().to(device)
# huberloss = torch.nn.HuberLoss().to(device)
# msleloss = loss.MSLE().to(device)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)

# Num_epochs
num_epochs = 100

def train(model, trainloader, valloader, optimizer, scheduler):
    for epoch in range(num_epochs-1):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        print("TRAINING")
        model.train()
        running_mseloss = 0.0
        running_rmseloss = 0.0
        running_maeloss = 0.0
        for batch_idx, (vid, ef) in tqdm(enumerate(trainloader)):
            optimizer.zero_grad() 
            video = vid.to(device)
            EF = ef.to(device)
            predicted_EF = model(video)
            # Step
            loss = mseloss(predicted_EF, EF)
            print(loss)
            loss.backward()
            optimizer.step()
            running_mseloss += loss.item()
            running_rmseloss += math.sqrt(loss.item())
            running_maeloss += maeloss(predicted_EF, EF)

        # Save data for plotting curves
        train_loss = running_mseloss/len(trainset)
        rmse_loss = running_rmseloss/len(trainset)
        mae_loss = running_maeloss/len(trainset)
        scheduler.step()
        # log
        writer.add_scalar("MSELoss/Train", train_loss, epoch)
        writer.add_scalar("RMSE/Train", rmse_loss, epoch)
        writer.add_scalar("MAE/Train", mae_loss, epoch)
        print("=> Epoch:{} - train loss: {:.4f}".format(epoch, train_loss))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), train_path)
            torch.save(optimizer.state_dict(), "optimizer.pth")
            torch.save(scheduler.state_dict(), "scheduler.pth")    

        # Validation 
        print("VALIDATING")
        with torch.no_grad():
            model.eval()
            running_mseloss = 0.0
            running_rmseloss = 0.0
            running_maeloss = 0.0
            for batch_idx, (vid, ef) in tqdm(enumerate(valloader)):
                video = vid.to(device)
                EF = ef.to(device)
                predicted_EF = model(video)
                loss = mseloss(predicted_EF, EF)
                print(loss)
                running_mseloss += loss.item()
                running_rmseloss += math.sqrt(loss.item())
                running_maeloss += maeloss(predicted_EF, EF)
            # Save data for plotting curves
            train_loss = running_mseloss/len(trainset)
            rmse_loss = running_rmseloss/len(trainset)
            mae_loss = running_maeloss/len(trainset)
            # log
            writer.add_scalar("MSELoss/Val", train_loss, epoch)
            writer.add_scalar("RMSE/Val", rmse_loss, epoch)
            writer.add_scalar("MAE/Val", mae_loss, epoch)

# Tensorboard
writer = SummaryWriter()
train(model, trainloader, valloader, optimizer, scheduler)
writer.flush()
writer.close()




