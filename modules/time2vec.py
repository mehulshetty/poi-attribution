import torch
import torch.nn as nn
import math

class Time2Vec(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.d_embed = d_embed

        # Introduce this to help capture frequency of visits
        self.freq = nn.Parameter(torch.arange(1, d_embed // 2 + 1, dtype=torch.float))

        self.linear = nn.Linear(d_embed, d_embed)

    def forward(self, time):
        # time: (batch_size, seq_len)
        time = time.unsqueeze(-1)  # (batch_size, seq_len, 1)
        freq = self.freq  # (d_embed // 2)
        time_freq = time * freq  # (batch_size, seq_len, d_embed // 2)

        # Introduce this for richer time encoding (since patterns in time are important)
        encoding = torch.cat([torch.sin(time_freq), torch.cos(time_freq)], dim=-1)  # (batch_size, seq_len, d_embed)
        
        return self.linear(encoding)