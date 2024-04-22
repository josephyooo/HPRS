import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights, alexnet, AlexNet_Weights, resnext50_32x4d, ResNeXt50_32X4D_Weights

# https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py
class GoogLeNet_Scalp(nn.Module):
    def __init__(self, device, num_classes, ffe=False):
        """
        ffe: Fine-tuning or Fixed feature extraction
        """
        super().__init__()
        self.model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features

        if ffe:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.model.to(device)
    
    def forward(self, x):
        return self.model(x)

# https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
class AlexNet_Scalp(nn.Module):
    def __init__(self, device, num_classes, ffe=False):
        """
        ffe: Fine-tuning or Fixed feature extraction
        """
        super().__init__()
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)
        num_ftrs = self.model.classifier[6].in_features
        if ffe:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        self.model.to(device)
    
    def forward(self, x):
        return self.model(x)

# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class ResNeXt_Scalp(nn.Module):
    def __init__(self, device, num_classes, ffe=False):
        """
        ffe: Fine-tuning or Fixed feature extraction
        """
        super().__init__()
        self.model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights)
        num_ftrs = self.model.fc.in_features
        if ffe:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.model.to(device)
    
    def forward(self, x):
        return self.model(x)
