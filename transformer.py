import torch.nn as nn
import torch
import torch.nn.functional as F
import math

import layernorm as ly
from attention import MutiHeadAttention
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
        len_q, len_k = q.size(1), k.size(1)
        k_pad_mask = k.eq(pad_idx_k).unsqueeze(1).unsqueeze(2)
        q_pad_mask = q.eq(pad_idx_q).unsqueeze(1).unsqueeze(3)
        pad_mask = k_pad_mask | q_pad_mask
        return pad_mask.expand(-1, self.num_heads, len_q, len_k)
    def make_causal_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).bool().to(self.device)
        return mask.unsqueeze(0).unsqueeze(1).expand(q.size(0), self.num_heads, len_q, len_k)

    def forward(self, src_input, tgt_input):
        src_mask = self.make_pad_mask(src_input, src_input, self.src_pad_idx, self.src_pad_idx)
        tgt_mask = self.make_pad_mask(tgt_input, tgt_input, self.trg_pad_idx, self.trg_pad_idx) & self.make_causal_mask(tgt_input, tgt_input)
        enc_output = self.encoder(src_input, src_mask)
        dec_output = self.decoder(tgt_input, enc_output, src_mask, tgt_mask)
        return dec_output