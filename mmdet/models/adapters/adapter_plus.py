import torch
import torch.nn as nn
from transformers.activations import ACT2FN

class AdapterPlus(nn.Module):
    def __init__(self, hidden_size, embedding_size,
                 act='gelu', init='houlsby',
                 init_std=1e-3,
                 use_embed=False,
                 emb_in_ch=256,
                 emb_out_ch=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.init = init
        self.init_std = init_std
        self.linear_down = nn.Linear(hidden_size, embedding_size)
        self.act = ACT2FN[act] if act else None
        self.linear_up = nn.Linear(embedding_size, hidden_size)
        self.scaling = nn.Parameter(torch.ones(hidden_size))

        self.use_embed = use_embed
        if self.use_embed:
            self.emb_proj = nn.Linear(emb_in_ch, emb_out_ch)

    def initialize(self):
        if self.init == "houlsby":
            std = 0.01
            nn.init.trunc_normal_(
                self.linear_down.weight, std=std, a=-2 * std, b=2 * std
            )
            if self.linear_down.bias is not None:
                nn.init.zeros_(self.linear_down.bias)
            nn.init.trunc_normal_(
                self.linear_up.weight, std=std, a=-2 * std, b=2 * std
            )
            if self.linear_up.bias is not None:
                nn.init.zeros_(self.linear_up.bias)
        else:
            NotImplementedError

    def nonlinearity(self, x):
        return x * torch.sigmoid(x)

    def forward(self, hidden_states, embed):
        if self.use_embed:
            embed = self.emb_proj(self.nonlinearity(embed))
            embed = embed.unsqueeze(1).expand_as(hidden_states)

        residual = hidden_states
        hidden_states = self.linear_down(hidden_states)
        if self.act:
            hidden_states = self.act(hidden_states)
        hidden_states = self.linear_up(hidden_states)
        hidden_states = hidden_states * self.scaling
        
        if self.use_embed:
            hidden_states = hidden_states + embed
        
        hidden_states = hidden_states + residual
        return hidden_states
