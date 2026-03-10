import torch.nn as nn
import torch
import torch.nn.functional as F
import math
print("torch version:", torch.__version__)

class Tokenembedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)