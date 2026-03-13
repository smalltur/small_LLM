import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
import spacy
import random

# ====================== 1. 导入torchtext并加载WMT19数据集 ======================
from torchtext.legacy import data, datasets

# 初始化分词器
spacy_en = spacy.load('en_core_web_sm')  # 英文分词器
def tokenize_en(text):
    """英文分词（转小写+分词）"""
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_zh(text):
    """中文分词（jieba）"""
    return [tok for tok in jieba.cut(text)]

# 定义Field（处理文本的字段）
SRC = data.Field(tokenize=tokenize_en, lower=True, pad_token='<pad>', init_token='<sos>', eos_token='<eos>')
TGT = data.Field(tokenize=tokenize_zh, lower=False, pad_token='<pad>', init_token='<sos>', eos_token='<eos>')

# 加载WMT19 英中数据集（注意：WMT19 zh-en是中文→英文，en-zh是英文→中文，按需调整）
# 若加载慢，可手动下载数据集后指定路径：path='./wmt19_data'
train_data, val_data, test_data = datasets.WMT19.splits(
    exts=('.en', '.zh'),  # 源语言(en)，目标语言(zh)
    fields=(SRC, TGT),
    root='./data',  # 数据集保存路径
    language_pair=('en', 'zh')  # 英文→中文
)

# 构建词表（限制词表大小，避免过大）
MAX_VOCAB_SIZE = 10000
SRC.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
TGT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)

# 获取关键参数（适配你的Transformer模型）
src_pad_idx = SRC.vocab.stoi[SRC.pad_token]  # padding索引
tgt_pad_idx = TGT.vocab.stoi[TGT.pad_token]  # padding索引
enc_voc_size = len(SRC.vocab)  # 编码器词表大小
dec_voc_size = len(TGT.vocab)  # 解码器词表大小

# ====================== 2. 自定义Dataset适配DataLoader ======================
class WMT19Dataset(Dataset):
    def __init__(self, examples, src_field, tgt_field, max_len):
        self.examples = examples
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        # 获取索引序列
        src_idx = [self.src_field.vocab.stoi[tok] for tok in example.src]
        tgt_idx = [self.tgt_field.vocab.stoi[tok] for tok in example.trg]
        
        # 截断/补全到max_len（包含<sos>/<eos>）
        src_idx = self._pad_truncate(src_idx, self.max_len)
        tgt_idx = self._pad_truncate(tgt_idx, self.max_len)
        
        return torch.tensor(src_idx, dtype=torch.long), torch.tensor(tgt_idx, dtype=torch.long)
    
    def _pad_truncate(self, seq, max_len):
        # 截断过长序列
        if len(seq) > max_len:
            seq = seq[:max_len]
        # 补全过短序列（用pad_token的索引）
        else:
            seq += [self.src_field.vocab.stoi[self.src_field.pad_token]] * (max_len - len(seq))
        return seq

# ====================== 3. 训练/验证函数（复用之前的逻辑） ======================
def train_transformer(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, tgt_pad_idx):
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_idx, (src_input, tgt_input) in enumerate(train_loader):
            src_input = src_input.to(device)
            tgt_input = tgt_input.to(device)
            
            optimizer.zero_grad()
            output = model(src_input, tgt_input)
            
            # 调整维度计算损失
            output = output.reshape(-1, output.size(-1))
            tgt_input = tgt_input.reshape(-1)
            
            loss = criterion(output, tgt_input)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device, tgt_pad_idx)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_wmt19_transformer.pth')
            print(f'Best model saved! Val Loss: {best_val_loss:.4f}')

def evaluate(model, val_loader, criterion, device, tgt_pad_idx):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for src_input, tgt_input in val_loader:
            src_input = src_input.to(device)
            tgt_input = tgt_input.to(device)
            
            output = model(src_input, tgt_input)
            output = output.reshape(-1, output.size(-1))
            tgt_input = tgt_input.reshape(-1)
            
            loss = criterion(output, tgt_input)
            val_loss += loss.item()
    
    model.train()
    return val_loss / len(val_loader)

# ====================== 4. 主函数（运行入口） ======================
if __name__ == '__main__':
    # 1. 配置超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据相关
    max_len = 50  # 最大序列长度
    batch_size = 16  # WMT19数据较大，建议减小batch_size
    
    # 模型相关
    num_layers = 2  # 层数不宜过多，避免训练过慢
    d_model = 256
    num_heads = 8
    d_ff = 1024
    dropout = 0.1
    
    # 训练相关
    num_epochs = 5
    lr = 0.0001
    
    # 2. 创建数据集和数据加载器
    train_dataset = WMT19Dataset(train_data.examples, SRC, TGT, max_len)
    val_dataset = WMT19Dataset(val_data.examples, SRC, TGT, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. 初始化模型（替换为你的Transformer类路径）
    from transformer import Transformer  # 替换为实际模型文件路径
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=tgt_pad_idx,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=max_len,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        device=device
    ).to(device)
    
    # 损失函数（忽略padding）
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    
    # 4. 开始训练
    print('Start training on WMT19 En-Zh dataset...')
    train_transformer(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, tgt_pad_idx)