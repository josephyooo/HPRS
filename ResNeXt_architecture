import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1, padding=0):
        super(GroupedConvolutionBlock, self).__init__()
        self.grouped_channels = out_channels // groups
        self.groups = groups
        
        # Define a list of convolutional layers
        self.grouped_convs = nn.ModuleList([
            nn.Conv2d(in_channels, self.grouped_channels, kernel_size, stride, padding) for _ in range(groups)
        ])
        
    def forward(self, x):
        # Apply grouped convolutions
        grouped_outputs = [conv(x[:, g * self.grouped_channels:(g + 1) * self.grouped_channels]) 
                           for g, conv in enumerate(self.grouped_convs)]
        # Concatenate the outputs along the channel dimension
        x = torch.cat(grouped_outputs, dim=1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1, padding=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = GroupedConvolutionBlock(in_channels, out_channels, kernel_size, groups, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = GroupedConvolutionBlock(out_channels, out_channels, kernel_size, groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(residual)
        x = F.relu(x, inplace=True)
        return x

class ResNeXt(nn.Module):
    def __init__(self, groups=32):
        super(ResNeXt, self).__init__()
        self.groups = groups
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.block1 = self._make_layer(64, 256, 3, groups)
        self.block2 = self._make_layer(256, 512, 4, groups, stride=2)
        self.block3 = self._make_layer(512, 1024, 6, groups, stride=2)
        self.block4 = self._make_layer(1024, 2048, 3, groups, stride=2)
        
        # Final fully connected layer
        self.fc = nn.Linear(2048, 1000)
        
    def _make_layer(self, in_channels, out_channels, blocks, groups, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, 3, groups, stride, 1))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 3, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
