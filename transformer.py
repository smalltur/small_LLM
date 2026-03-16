import torch.nn as nn
import torch
import torch.nn.functional as F
import math

import layernorm as ly
from attention import MultiHeadAttention
import tokeners as tk
import encoder as enc
import decoder as dec

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, max_len, num_layers, d_model, num_heads, d_ff, dropout=0.1, device = torch.device('cpu')):
        super(Transformer, self).__init__()
        self.encoder = enc.Encoder(enc_voc_size, max_len, num_layers, d_model, num_heads, d_ff, dropout, device)
        self.decoder = dec.Decoder(dec_voc_size, max_len, num_layers, d_model, num_heads, d_ff, dropout, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.num_heads = num_heads
        self.device = device
    
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):

        batch_size = q.size(0)
        len_q, len_k = q.size(1), k.size(1)
        
        # 1. 生成K的填充掩码 [batch, 1, 1, len_k] → 广播到 [batch, 1, len_q, len_k]
        k_pad_mask = k.eq(pad_idx_k).unsqueeze(1).unsqueeze(2)  # [batch,1,1,len_k]
        # 2. 生成Q的填充掩码 [batch, 1, len_q, 1] → 广播到 [batch, 1, len_q, len_k]
        q_pad_mask = q.eq(pad_idx_q).unsqueeze(1).unsqueeze(3)  # [batch,1,len_q,1]
        # 3. 合并掩码（True表示填充位，需要遮挡）
        pad_mask = k_pad_mask | q_pad_mask  # [batch,1,len_q,len_k]
        
        # 4. 扩展到num_heads维度（核心：只扩展num_heads，不修改len_q/len_k）
        pad_mask = pad_mask.expand(batch_size, self.num_heads, len_q, len_k)
        pad_mask = pad_mask.to(self.device)
        
        return pad_mask

    def make_causal_mask(self, q, k):

        batch_size = q.size(0)
        len_q, len_k = q.size(1), k.size(1)
        
        # 1. 生成下三角掩码（适配len_q≠len_k的场景）
        # 推理时len_q=1、len_k=50 → 生成1×50的掩码，仅第一列True
        causal_mask = torch.tril(torch.ones((len_q, len_k), dtype=torch.bool, device=self.device))
        
        # 2. 扩展维度到 [batch, num_heads, len_q, len_k]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1,1,len_q,len_k]
        causal_mask = causal_mask.expand(batch_size, self.num_heads, len_q, len_k)
        
        return causal_mask

    def forward(self, src_input, tgt_input):

        # ========== 1. 编码器掩码：源序列自注意力掩码 ==========
        src_mask_enc = self.make_pad_mask(src_input, src_input, self.src_pad_idx, self.src_pad_idx)
        enc_output = self.encoder(src_input, src_mask_enc)  # [B,Ls,d_model]
        
        # ========== 2. 解码器自注意力掩码：填充+因果 ==========
        tgt_pad_mask = self.make_pad_mask(tgt_input, tgt_input, self.trg_pad_idx, self.trg_pad_idx)
        tgt_causal_mask = self.make_causal_mask(tgt_input, tgt_input)
        tgt_mask_dec = tgt_pad_mask & tgt_causal_mask  # [B,H,Lt,Lt]
        
        # ========== 3. 解码器跨注意力掩码：源-目标填充掩码 ==========
        # 关键：Q=解码器输入（Lt），K=编码器输出（Ls），生成[B,H,Lt,Ls]的掩码
        src_mask_dec = self.make_pad_mask(tgt_input, src_input, self.trg_pad_idx, self.src_pad_idx)
        

        # ========== 解码器前向传播 ==========
        dec_output = self.decoder(tgt_input, enc_output, src_mask_dec, tgt_mask_dec)
        return dec_output
