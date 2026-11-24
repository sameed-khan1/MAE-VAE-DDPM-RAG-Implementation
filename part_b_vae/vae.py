
import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, latent_dim=16):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.decoder_input = nn.Linear(
            latent_dim,
            64 * 7 * 7
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            )
        )

    def encode(self, x):

        features = self.encoder(x)

        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        return mu + std * epsilon

    def decode(self, z):

        x = self.decoder_input(z)

        x = x.view(-1, 64, 7, 7)

        return self.decoder(x)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar

    def sample(self, number, device):

        z = torch.randn(
            number,
            self.latent_dim,
            device=device
        )

        with torch.no_grad():
            logits = self.decode(z)
            images = torch.sigmoid(logits)

        return images
