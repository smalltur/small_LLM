import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class MutiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MutiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None):
        batch, seq_len, dimension = q.size()
        n_d = self.d_model // self.num_heads
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch, seq_len, self.num_heads, n_d).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, n_d).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, n_d).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(n_d)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
            score = self.softmax(score)
        output = torch.matmul(score, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.w_combine(output)
        return output
    

