import torch.nn as nn
import torch
import torch.nn.functional as F
import math

import layernorm as ly
from attention import MutiHeadAttention
import tokeners as tk
import encoder as enc

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MutiHeadAttention(d_model, num_heads)
        self.norm_1 = ly.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.enc_dec_attention = MutiHeadAttention(d_model, num_heads)
        self.norm_2 = ly.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = enc.PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm_3 = ly.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # ========== 修复1：统一参数名（dec_output → x，与调用端对齐） ==========
        # 1. 掩码自注意力（目标序列内部）
        attn_output = self.self_attention(x, x, x, tgt_mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm_1(x + attn_output)

        # 2. 编码器-解码器注意力（关注编码器输出）
        enc_dec_attn_output = self.enc_dec_attention(x, enc_output, enc_output, src_mask)
        enc_dec_attn_output = self.dropout2(enc_dec_attn_output)
        x = self.norm_2(x + enc_dec_attn_output)

        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        ff_output = self.dropout3(ff_output)
        x = self.norm_3(x + ff_output)

        # ========== 修复2：添加返回值（核心！否则输出None） ==========
        return x

class Decoder(nn.Module):
    def __init__(self,dec_voc_size, max_len, num_layers, d_model, num_heads, d_ff, dropout=0.1, device = torch.device('cpu')):
        super(Decoder, self).__init__()
        self.embedding = tk.TokenEmbedding(dec_voc_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        #self.norm = ly.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec_input, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(dec_input)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        x = self.fc(x)
        return x