import os

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms._presets import ImageClassification

def DermNet(data_dir=None, transform=None):
    if not data_dir:
        cwd = os.path.dirname(__file__)
        data_dir = os.path.join(cwd, "Image_Data/DermNet")
    if not transform:
        # Use standard ImageNet normalization
        transform = ImageClassification(crop_size=224)

    return datasets.ImageFolder(data_dir, transform=transform)

def get_dataloaders(dataset, transform=None, batch_size=64, split=0.2, shuffle=True, num_workers = 4, pin_memory=False):
    train_size = int((1 - split) * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision

    plt.rcParams['font.size'] = 10

    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    def imshow(inp, title=None):
        """Display image for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    # Get a batch of training data
    dataset = DermNet()
    loader = get_dataloaders(dataset, batch_size=4)[0]
    inputs, classes = next(iter(loader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    dataset = DermNet()
    class_names = dataset.classes
    # [:-9] to remove to '-pictures' suffix
    imshow(out, title=[class_names[x][:-9] for x in classes])
