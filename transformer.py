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
        """
        生成填充掩码：True=遮挡（padding位），False=可见（有效token）
        q: [batch, len_q]
        k: [batch, len_k]
        return: [batch, num_heads, len_q, len_k]
        """
        batch_size = q.size(0)
        len_q, len_k = q.size(1), k.size(1)
        
        # 1. 生成K的填充掩码：[batch, 1, 1, len_k] → True表示K的padding位
        k_pad_mask = (k == pad_idx_k).unsqueeze(1).unsqueeze(2)  # [B,1,1,Lk]
        # 2. 生成Q的填充掩码：[batch, 1, len_q, 1] → True表示Q的padding位
        q_pad_mask = (q == pad_idx_q).unsqueeze(1).unsqueeze(3)  # [B,1,Lq,1]
        
        # 3. 合并掩码：只要Q或K是padding，该位置就遮挡（True）
        pad_mask = k_pad_mask | q_pad_mask  # [B,1,Lq,Lk]
        
        # 4. 扩展到num_heads维度
        pad_mask = pad_mask.expand(batch_size, self.num_heads, len_q, len_k)
        pad_mask = pad_mask.to(self.device)
        
        # 打印填充掩码信息
        
        return pad_mask

    def make_causal_mask(self, q, k):
        """
        生成因果掩码：True=遮挡（未来token），False=可见（过去/当前token）
        q: [batch, len_q]
        k: [batch, len_k]
        return: [batch, num_heads, len_q, len_k]
        """
        batch_size = q.size(0)
        len_q, len_k = q.size(1), k.size(1)
        
        # 1. 生成上三角掩码（True=未来token，需要遮挡）
        # 正确逻辑：tril生成下三角True → 取反得到上三角True
        causal_mask = ~torch.tril(torch.ones((len_q, len_k), dtype=torch.bool, device=self.device))
        
        # 2. 扩展维度到 [batch, num_heads, len_q, len_k]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1,1,Lq,Lk]
        causal_mask = causal_mask.expand(batch_size, self.num_heads, len_q, len_k)
        

        
        return causal_mask

    def forward(self, src_input, tgt_input):

        
        # ========== 1. 编码器掩码：源序列自注意力掩码 ==========
        
        src_mask_enc = self.make_pad_mask(src_input, src_input, self.src_pad_idx, self.src_pad_idx)
        
       
        enc_output = self.encoder(src_input, src_mask_enc)  # [B,Ls,d_model]
        
        
        # ========== 2. 解码器自注意力掩码：填充+因果 ==========
        
        tgt_pad_mask = self.make_pad_mask(tgt_input, tgt_input, self.trg_pad_idx, self.trg_pad_idx)
        
        
        tgt_causal_mask = self.make_causal_mask(tgt_input, tgt_input)
        
        tgt_mask_dec = tgt_pad_mask | tgt_causal_mask  # 正确逻辑：填充位 或 未来token 都要遮挡

        
        # ========== 3. 解码器跨注意力掩码：源-目标填充掩码 ==========

        src_mask_dec = self.make_pad_mask(tgt_input, src_input, self.trg_pad_idx, self.src_pad_idx)
        
        # ========== 解码器前向传播 ==========

        dec_output = self.decoder(tgt_input, enc_output, src_mask_dec, tgt_mask_dec)


        
        return dec_output
