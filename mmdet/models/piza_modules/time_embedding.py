import math
import torch
from torch import nn

class TimeEmbedding(nn.Module):
    def __init__(self, ch, out_ch):
        super().__init__()
        self.ch = ch
        self.temb_ch = out_ch

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                torch.nn.Linear(self.ch, self.temb_ch),
                torch.nn.Linear(self.temb_ch, self.temb_ch),
            ]
        )

    def get_timestep_embedding(self, timesteps, embedding_dim):
        assert len(timesteps.shape) == 1

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb
    
    def nonlinearity(self, x):
        return x * torch.sigmoid(x)

    def forward(self, t):
        temb = self.get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = self.nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        return temb