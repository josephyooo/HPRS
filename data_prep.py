from collections import Counter
import os

from torch import Tensor
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms._presets import ImageClassification

def DermNet(data_dir=None, transform=None):
    if not data_dir:
        cwd = os.path.dirname(__file__)
        data_dir = os.path.join(cwd, "Image_Data")
    if not transform:
        # Use standard ImageNet normalization
        transform = ImageClassification(crop_size=224)

    return datasets.ImageFolder(data_dir, transform=transform)

def get_balanced_weights(dataset, datasubset):
    counts = Counter(dataset.targets[i] for i in datasubset.indices)
    counts = [x for _,x in sorted(counts.items())]
    counts = Tensor(counts)
    # The numerator can be any constant but I guessed using a greater value would preserve precision
    return counts.max() / counts

def get_dataloaders(dataset, batch_size=64, split=0.2, shuffle=True, num_workers = 4, pin_memory=False, weighted_random_sampling=False):
    train_size = int((1 - split) * len(dataset))
    val_size = len(dataset) - train_size

    # if using pretrained weights to train, ensure that the same torch.random.seed is used
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if weighted_random_sampling:
        weights = get_balanced_weights(dataset, train_dataset)
        sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

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
