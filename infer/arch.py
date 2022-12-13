import torch
import torch.nn as nn
from torchvision import transforms, models


class MLP(nn.Module):
    def __init__(self, num_concepts, mid_dim, num_classes):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_concepts, num_classes),
            #nn.ReLU(inplace=True),
            #nn.Linear(1000, 1000),
            #nn.ReLU(inplace=True),
            #nn.Linear(mid_dim, num_classes),
        )
        
    def forward(self, x):
        x = self.net(x)
        return x

def initialize_pretrained_model(num_concepts = 312):
    model = models.resnet18(pretrained=True)
    for params in model.parameters():
        params.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_concepts)
    return model