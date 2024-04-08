import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from torchvision.io import read_image

# Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class DermNet(Dataset):
    def __init__(self, img_dir, subfolders=None, transform=None, augmenter=None):
        if not subfolders:
            # self.subfolders = ['Alopecia-pictures', 'seborrheic-dermatitis-pictures', 'Psoriasis-pictures']
            # Prefer to infer subfolders
            self.subfolders = os.listdir(img_dir)
        else:
            self.subfolders = subfolders

        self.img_labels = {}
        for i in range(len(self.subfolders)):
            img_list = os.listdir(os.path.join(img_dir, self.subfolders[i]))
            self.img_labels.update({os.path.join(self.subfolders[i], img): i for img in img_list})
        self.img_labels = pd.DataFrame.from_dict(self.img_labels, orient='index').reset_index()
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = transforms.Lambda(lambda y: F.one_hot(torch.tensor(y), num_classes=len(self.subfolders)))
        self.augmenter = augmenter
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        # Tensor[image_channels, image_height, image_width] -> Tensor[image_height, image_width, image_channels]
        # image = image.movedim(0, 2)

        label = self.img_labels.iloc[idx, 1]

        # apply transforms
        if self.augmenter:
            image = self.augmenter(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

def get_data_loaders(data_dir=None, transform=None, augmenter=None, batch_size=64, test_split=0.2, shuffle=True):
    if not data_dir:
        cwd = os.path.dirname(__file__)
        data_dir = os.path.join(cwd, "Image_Data/DermNet")

    if not transform:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
        ])
    if not augmenter:
        # Alternatives: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#augmentation-transforms
        augmenter = transforms.AugMix()
    dataset = DermNet(img_dir=data_dir, transform=transform, augmenter=augmenter)

    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader

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