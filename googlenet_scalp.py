import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

class GoogLeNet_Scalp(nn.Module):
    def __init__(self, device, num_classes):
        super().__init__()
        self.model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        # Used for fixed feature extraction
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # self.model.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, num_ftrs//2),
        #     nn.Linear(num_ftrs//2, num_classes),
        # )

        self.model.to(device)
    
    def forward(self, x):
        return self.model(x)
