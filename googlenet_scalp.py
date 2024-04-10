import time
import os
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import googlenet, GoogLeNet_Weights
import torchvision.transforms.v2 as v2
import torchvision

from data_prep import get_data_loaders

class GoogLeNetHair(nn.Module):
    def __init__(self, device, weights=None, num_classes=3):
        super().__init__()
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights=GoogLeNet_Weights.DEFAULT)
        self.model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        if weights:
            self.model.load_state_dict(torch.load(weights))

        self.model.to(device)
    
    def forward(self, x):
        return self.model(x)

'''
set batchsize to 4 to work well
import matplotlib.pyplot as plt
import numpy as np

from data import DermNet

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.rcParams['figure.figsize'] = (20, 20)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

dataset = DermNet()
class_names = dataset.classes
imshow(out, title=[class_names[x] for x in classes])
'''

# Reference: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, augmenter=None):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        dataset_sizes = {phase: len(dataloader.dataset) for phase, dataloader in dataloaders.items()}

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs} | Learning Rate {optimizer.param_groups[0]["lr"]}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in dataloaders.keys():
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if augmenter:
                        inputs = augmenter(inputs)

                    # zero the parameter gradients
                    optimizer.zero_grad(set_to_none=True)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

if __name__ == "__main__":
    # hyperparameters
    batch_size = 64
    lr = 1e-3
    eps = 1e-4
    weight_decay = 1e-3
    step_size = 7
    gamma = 0.1
    num_epochs = 25

    # optimizations
    num_workers = 12
    pin_memory = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # finetune model
    model_ft = GoogLeNetHair(device=device)
    model_ft.load_state_dict(torch.load('weights/googlenet_hair_100.pt'))

    transform = GoogLeNet_Weights.DEFAULT.transforms()
    augmenter = v2.AugMix()

    train_loader, val_loader = get_data_loaders(transform=transform,
                                                batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders = {'train': train_loader, 'val': val_loader}


    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, lr_scheduler,
                           num_epochs=num_epochs, device=device, augmenter=augmenter)
    torch.save(model_ft.state_dict(), 'weights/googlenet_hair.pt')



