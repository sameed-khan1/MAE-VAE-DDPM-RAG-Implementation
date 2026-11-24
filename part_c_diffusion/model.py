
import torch
import torch.nn as nn
from diffusers import UNet2DModel


class DiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = UNet2DModel(
            sample_size=28,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"
            ),
            num_class_embeds=40
        )

    def forward(self, images, timesteps, class_labels):
        return self.model(
            images,
            timesteps,
            class_labels=class_labels
        )
