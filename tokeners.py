import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 1. 词嵌入层：将token索引转换为词向量（修复核心错误）
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx=0):
        super(TokenEmbedding, self).__init__()  # 父类nn.Module的__init__无需参数
        # 核心：定义nn.Embedding层，指定padding_idx（填充索引）
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 词汇表大小
            embedding_dim=d_model,      # 词向量维度
            padding_idx=padding_idx     # 填充token的索引（设为1）
        )

    def forward(self, x):
        # x: 输入token索引，形状 [batch_size, seq_len]
        # 输出词向量，形状 [batch_size, seq_len, d_model]
        return self.embedding(x)


# 2. 位置编码层：为词向量添加位置信息（适配3维输入）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device=torch.device('cpu')):
        super(PositionalEncoding, self).__init__()
        self.device = device
        # 预计算位置编码矩阵（max_len, d_model）
        self.encoding = torch.zeros(max_len, d_model).to(device)
        self.encoding.requires_grad = False  # 位置编码不参与训练

        # 生成位置索引 [max_len, 1]
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        # 生成2i的序列（对应公式中的2i）
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 偶数位用sin，奇数位用cos（Transformer位置编码公式）
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # x: 词嵌入输出，形状 [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        # 取出对应长度的位置编码，并扩展到batch维度
        # 输出形状：[batch_size, seq_len, d_model]
        return self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)


# 3. 组合层：词嵌入 + 位置编码 + Dropout
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000, drop_prob=0.1, device=torch.device('cpu')):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)  # 词嵌入
        self.pos_emb = PositionalEncoding(d_model, max_len, device)  # 位置编码
        self.drop_out = nn.Dropout(p=drop_prob)  # Dropout防止过拟合

    def forward(self, x):
        # x: token索引，形状 [batch_size, seq_len]
        tok_emb = self.tok_emb(x)  # 词嵌入：[batch_size, seq_len, d_model]
        pos_emb = self.pos_emb(tok_emb)  # 位置编码：[batch_size, seq_len, d_model]
        # 词嵌入+位置编码后加Dropout
        return self.drop_out(tok_emb + pos_emb)

