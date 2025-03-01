from space2vec import Space2Vec
from time2vec import Time2Vec
import torch
import torch.nn as nn

class SourceInput(nn.Module):
    def __init__(self, d_embed=64, lambda_min=1.0, lambda_max=1000.0):
        super().__init__()
        self.space2vec = Space2Vec(d_embed, lambda_min, lambda_max)
        self.time2vec = Time2Vec(d_embed)
        self.duration2vec = Time2Vec(d_embed)  # Treat duration as a temporal feature
        self.travel2vec = Time2Vec(d_embed)    # For travel time
        self.proj = nn.Linear(d_embed * 5, d_embed)  # Combine all embeddings

    def forward(self, visits):
        # visits: (batch_size, seq_len, 6) - [x, y, start_timestamp, stop_timestamp, duration, travel_time]
        x = visits[..., 0]          # (batch_size, seq_len)
        y = visits[..., 1]          # (batch_size, seq_len)
        start_time = visits[..., 2] # (batch_size, seq_len)
        stop_time = visits[..., 3]  # (batch_size, seq_len)
        duration = visits[..., 4]   # (batch_size, seq_len)
        travel_time = visits[..., 5] # (batch_size, seq_len)

        # Encode features
        locations = torch.stack([x, y], dim=-1)  # (batch_size, seq_len, 2)
        location_encoding = self.space2vec(locations)      # (batch_size, seq_len, d_embed)
        start_encoding = self.time2vec(start_time)         # (batch_size, seq_len, d_embed)
        stop_encoding = self.time2vec(stop_time)           # (batch_size, seq_len, d_embed)
        duration_encoding = self.duration2vec(duration)    # (batch_size, seq_len, d_embed)
        travel_encoding = self.travel2vec(travel_time)     # (batch_size, seq_len, d_embed)

        # Combine embeddings
        combined = torch.cat([
            location_encoding,
            start_encoding,
            stop_encoding,
            duration_encoding,
            travel_encoding
        ], dim=-1)  # (batch_size, seq_len, d_embed * 5)

        # Project to final embedding size
        visit_embedding = self.proj(combined)  # (batch_size, seq_len, d_embed)
        return visit_embedding