
import torch
import torch.nn as nn


class MaskedAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.patch_dim = 192
        self.encoder_dim = 192
        self.decoder_dim = 128
        self.num_patches = 64
        self.num_visible = 16

        self.patch_embedding = nn.Linear(192, 192)

        self.encoder_position = nn.Parameter(
            torch.zeros(1, 64, 192)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=3,
            dim_feedforward=768,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )

        self.encoder_norm = nn.LayerNorm(192)

        self.decoder_embedding = nn.Linear(192, 128)

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 128)
        )

        self.decoder_position = nn.Parameter(
            torch.zeros(1, 64, 128)
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512,
            batch_first=True
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=2
        )

        self.decoder_prediction = nn.Linear(128, 192)

        nn.init.normal_(self.encoder_position, std=0.02)
        nn.init.normal_(self.decoder_position, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(self, patches):

        batch_size, num_patches, _ = patches.shape

        noise = torch.rand(
            batch_size,
            num_patches,
            device=patches.device
        )

        ids_shuffle = torch.argsort(noise, dim=1)

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :self.num_visible]

        visible_patches = torch.gather(
            patches,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(
                -1, -1, patches.shape[-1]
            )
        )

        mask = torch.ones(
            batch_size,
            num_patches,
            device=patches.device
        )

        mask[:, :self.num_visible] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return visible_patches, mask, ids_restore

    def forward(self, patches):

        visible, mask, ids_restore = self.random_masking(patches)

        visible = self.patch_embedding(visible)

        ids_keep = torch.argsort(ids_restore, dim=1)[:, :self.num_visible]

        visible_positions = torch.gather(
            self.encoder_position.expand(patches.shape[0], -1, -1),
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, self.encoder_dim)
        )

        encoded = self.encoder(visible + visible_positions)

        encoded = self.encoder_norm(encoded)

        encoded = self.decoder_embedding(encoded)

        batch_size = patches.shape[0]

        mask_tokens = self.mask_token.expand(
            batch_size,
            self.num_patches - self.num_visible,
            -1
        )

        decoder_tokens = torch.cat(
            [encoded, mask_tokens],
            dim=1
        )

        decoder_tokens = torch.gather(
            decoder_tokens,
            1,
            ids_restore.unsqueeze(-1).expand(
                -1, -1, self.decoder_dim
            )
        )

        decoder_tokens = decoder_tokens + self.decoder_position

        decoded = self.decoder(decoder_tokens)

        predictions = self.decoder_prediction(decoded)

        return predictions, mask

    def encode_all(self, patches):

        x = self.patch_embedding(patches)
        x = x + self.encoder_position

        x = self.encoder(x)
        x = self.encoder_norm(x)

        return x.mean(dim=1)
