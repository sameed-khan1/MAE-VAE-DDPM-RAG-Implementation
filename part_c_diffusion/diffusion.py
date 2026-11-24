
import torch
import torch.nn as nn


class Diffusion:
    def __init__(self, timesteps=1000, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(
            1e-4, 0.02, timesteps, device=device
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(
            1.0 - self.alpha_bars
        )

    def sample_timesteps(self, batch_size):
        return torch.randint(
            0,
            self.timesteps,
            (batch_size,),
            device=self.device
        )

    def add_noise(self, images, noise, timesteps):
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps]

        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None]
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[:, None, None, None]

        noisy_images = (
            sqrt_alpha_bar * images
            + sqrt_one_minus_alpha_bar * noise
        )

        return noisy_images

    @torch.no_grad()
    def sample(self, model, labels, image_size=28):
        model.eval()

        batch_size = labels.shape[0]

        x = torch.randn(
            batch_size, 1, image_size, image_size,
            device=self.device
        )

        for t in reversed(range(self.timesteps)):
            timesteps = torch.full(
                (batch_size,),
                t,
                device=self.device,
                dtype=torch.long
            )

            predicted_noise = model(
                x,
                timesteps,
                class_labels=labels
            ).sample

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise
            )

            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise

            x = torch.clamp(x, -1.0, 1.0)

        return x
