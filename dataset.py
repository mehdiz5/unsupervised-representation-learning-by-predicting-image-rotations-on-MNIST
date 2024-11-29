import torch
from torchvision import transforms
import random

class RotationDataset(torch.utils.data.Dataset,):
    """Dataset that generates rotated versions of images with rotation labels (0°, 90°)."""
    def __init__(self, dataset, rotations):
        self.dataset = dataset
        self.rotations = rotations
        self.rotations_label = {element: index for index, element in enumerate(rotations)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        rotation_angle = random.choice(self.rotations)
        rotated_image = transforms.functional.rotate(image, rotation_angle)
        rotation_label=self.rotations_label[rotation_angle]

        return rotated_image, rotation_label