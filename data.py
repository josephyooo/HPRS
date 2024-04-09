import os

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torchvision.transforms._presets import ImageClassification

def get_data_loaders(data_dir=None, transform=None, augmenter=None, batch_size=64, test_split=0.2, shuffle=True):
    if not data_dir:
        cwd = os.path.dirname(__file__)
        data_dir = os.path.join(cwd, "Image_Data/DermNet")

    if not transform:
        # Use standard ImageNet normalization
        transform = ImageClassification(crop_size=224)
    if not augmenter:
        # Alternatives: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#augmentation-transforms
        augmenter = transforms.AugMix()
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    train_loader, test_loader = get_data_loaders()
    train_features, train_labels = next(iter(train_loader))
    img = train_features[0]
    label = train_labels[0]
    #%%
    figure = plt.figure()
    plt.imshow(img.movedim(0, 2))
    plt.show()