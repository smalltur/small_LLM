from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
input = tokenizer.encode("我喜欢小企鹅", return_tensors='pt')
print(input)

tok_emb = torch.nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=512)
# 这里由于字在不同的位置，含义是不一样的，所以需要一个位置嵌入，来区分不同的位置
pos_emb = torch.nn.Embedding(num_embeddings=512, embedding_dim=512)
print(tok_emb(input)+pos_emb(torch.arange(input.shape[1])))