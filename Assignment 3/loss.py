# MSLE
import torch
import torch.nn as nn

class MSLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, ground):
        return self.mse(torch.log(pred+1), torch.log(ground+1))