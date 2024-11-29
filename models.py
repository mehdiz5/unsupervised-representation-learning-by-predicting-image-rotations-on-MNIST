import torch
import torch.nn as nn

# -----------------------------------------
# 4. Define RotNet Model for Rotation Prediction
# -----------------------------------------import torch
class ConvNet(nn.Module):
    def __init__(self, input_image_size=28, num_blocks=4, num_classes=8,checkpoint_path=None):
        super(ConvNet, self).__init__()
        
        self.input_image_size = input_image_size
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        # Initialize convolutional blocks
        in_channels = 1  # Assuming grayscale input
        out_channels = 32

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)  # Reduces the image size by half
                )
            )
            in_channels = out_channels
            out_channels *= 2

        # Adaptive average pooling to reduce to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer 
        self.classification_head = nn.Linear(in_channels, num_classes)

        # load checkpoint
        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path))
            

    def forward(self, x):
        """
        Forward pass of the RotNet model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output logits of shape (batch_size, num_classes).
        """
        for block in self.blocks:
            x = block(x)
        x = self.adaptive_pool(x)  # Output size becomes (batch_size, channels, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.classification_head(x)
        return x

    def transfer_layers(self, target_model, num_transfer_blocks):
        """
        Transfers the first `num_transfer_blocks` layers (blocks) from `source_model` to `target_model`.

        Args:
            source_model (nn.Module): The source network (e.g., RotNet).
            target_model (nn.Module): The target network to receive layers.
            num_transfer_blocks (int): Number of blocks to transfer.
        """
        # Ensure the target model has enough blocks to accept the transfer
        assert len(target_model.blocks) >= num_transfer_blocks, "Target model has fewer blocks than required."

        # Copy parameters for the specified number of blocks
        for i in range(num_transfer_blocks):
            target_model.blocks[i].load_state_dict(self.blocks[i].state_dict())

        print(f"Transferred {num_transfer_blocks} blocks from source model to target model.")
