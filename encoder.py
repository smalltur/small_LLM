import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MutiHeadAttention(d_model, num_heads)
        self.norm_1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm_2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.self_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.n orm_1(x + attn_output)  # Residual connection and layer normalization

        # Feed-forward network
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm_2(x + ff_output)  # Residual connection and layer normalization

        return x