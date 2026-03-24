import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 线性变换并分头
        q = self.w_q(q).view(batch_size, q.size(1), self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        k = self.w_k(k).view(batch_size, k.size(1), self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        v = self.w_v(v).view(batch_size, v.size(1), self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        seq_len_q = q.size(2)  # 注意：这里固定seq_len_q，避免-1导致的维度错误
        seq_len_k = k.size(2)  # 注意：这里固定seq_len_k
        
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // self.num_heads)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)
        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, v)
       
          # 3. 合并多头（核心修复！固定seq_len_q，避免-1导致的维度错误）
        attn_output = attn_output.transpose(1, 2).contiguous()  # [8,50,8,16]
        attn_output = attn_output.view(batch_size, seq_len_q, self.d_model)  # [8,50,128]（和输入x形状一致）
        
        output = self.w_combine(attn_output)
        return output
    

