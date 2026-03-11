import torch.nn as nn
import torch
import torch.nn.functional as F
import math
print("torch version:", torch.__version__)

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
        