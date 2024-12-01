import torch
from torchvision import transforms
import random

class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rotations):
        self.dataset = dataset
        self.rotations = rotations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        rotated_images = []
        rotation_labels = []

        for rotation in self.rotations:
            rotated_image = transforms.functional.rotate(image, rotation)
            rotated_images.append(rotated_image)
            rotation_labels.append(self.rotations.index(rotation))

        return rotated_images, rotation_labels
