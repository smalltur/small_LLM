import torch.nn as nn
import torch
import torch.nn.functional as F
import math


#将输入的词语转换为对应的词向量
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device = torch.device('cpu')):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    def forward(self, x):
        batch_size, seq_len = x.size()# x的维度是(batch_size, seq_len)
        return self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
    
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000, drop_prob=0.1, device = torch.device('cpu')):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p = drop_prob)#drop_prob是丢弃的概率，默认为0.1
    def forward(self, x):
        return self.drop_out(self.tok_emb(x) + self.pos_emb(x))