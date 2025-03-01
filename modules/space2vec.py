import torch
import torch.nn as nn
import math

class Space2Vec(nn.Module):
    def __init__(self, d_embed, lambda_min, lambda_max):
        super().__init__()
        self.d_embed = d_embed
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Learnable parameters for spatial encoding
        # Similar to positional encodings
        # We alsp dont need large scales like TrajGPT so doing this should help focus more on local data
        # Frequencies might help capture density patterns
        self.freq = nn.Parameter(torch.linspace(math.log(lambda_min), math.log(lambda_max), d_embed // 2))
        self.linear = nn.Linear(d_embed, d_embed)

    def forward(self, locations):
        batch_size, seq_len, _ = locations.shape
        x, y = locations[..., 0], locations[..., 1]
        freq = torch.exp(self.freq)  # (d_embed // 2)
        x_freq = x.unsqueeze(-1) * freq  # (batch_size, seq_len, d_embed // 2)
        y_freq = y.unsqueeze(-1) * freq

        # Applying different trig functions for x and y for richer location trends
        encoding = torch.cat([torch.sin(x_freq), torch.cos(y_freq)], dim=-1)  # (batch_size, seq_len, d_embed)
        return self.linear(encoding)