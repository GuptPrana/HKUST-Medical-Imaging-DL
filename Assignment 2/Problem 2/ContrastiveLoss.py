# Supervised Contrastive Loss
# Need the following library
# !pip install -q timm pytorch-metric-learning

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
    
    def forward(self, feature, label):
        feature_norm = F.normalize(feature)
        logits = torch.matmul(feature_norm, torch.transpose(feature_norm, 0, 1))
        logits = torch.div(logits, self.temp)
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(label))

