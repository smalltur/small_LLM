import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
import os
import matplotlib.pyplot as plt

# ====================== 1. 复用之前的数据加载代码 ======================
class Vocabulary:
    def __init__(self):
        self.token2idx = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
        self.idx2token = {0:'<pad>', 1:'<sos>', 2:'<eos>', 3:'<unk>'}

    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = len(self.token2idx)
            self.idx2token[self.token2idx[token]] = token

    def convert_tokens_to_ids(self, tokens):
        return [self.token2idx.get(token, self.token2idx['<unk>']) for token in tokens]

    def __len__(self):
        return len(self.token2idx)

def tokenize_zh(text):
    return text.strip().split()

def tokenize_en(text):
    return text.strip().lower().split()

def load_your_data(zh_file_path, en_file_path, max_len=50):
    # 读取文件
    zh_lines = []
    en_lines = []
    with open(zh_file_path, 'r', encoding='utf-8') as f_zh, \
         open(en_file_path, 'r', encoding='utf-8') as f_en:
        for zh_line, en_line in zip(f_zh, f_en):
            if zh_line.strip() and en_line.strip():
                zh_lines.append(zh_line.strip())
                en_lines.append(en_line.strip())

    # 构建词表
    src_vocab = Vocabulary()  # 英文（源）
    tgt_vocab = Vocabulary()  # 中文（目标）
    for zh_line, en_line in zip(zh_lines, en_lines):
        zh_tokens = tokenize_zh(zh_line)
        en_tokens = tokenize_en(en_line)
        for token in zh_tokens:
            tgt_vocab.add_token(token)
        for token in en_tokens:
            src_vocab.add_token(token)

    # 文本转索引
    data = []
    pad_idx = src_vocab.token2idx['<pad>']
    for zh_line, en_line in zip(zh_lines, en_lines):
        zh_tokens = tokenize_zh(zh_line)
        en_tokens = tokenize_en(en_line)
        src_ids = [src_vocab.token2idx['<sos>']] + src_vocab.convert_tokens_to_ids(en_tokens) + [src_vocab.token2idx['<eos>']]
        tgt_ids = [tgt_vocab.token2idx['<sos>']] + tgt_vocab.convert_tokens_to_ids(zh_tokens) + [tgt_vocab.token2idx['<eos>']]
        
        # 截断/补全
        src_ids = src_ids[:max_len] if len(src_ids) > max_len else src_ids + [pad_idx]*(max_len - len(src_ids))
        tgt_ids = tgt_ids[:max_len] if len(tgt_ids) > max_len else tgt_ids + [pad_idx]*(max_len - len(tgt_ids))
        data.append((src_ids, tgt_ids))

    return src_vocab, tgt_vocab, data

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# ====================== 2. 训练核心函数 ======================
def train(model, train_loader, criterion, optimizer, device, tgt_pad_idx, epoch, log_interval=1):
    """单轮训练函数"""
    model.train()  # 切换到训练模式
    total_loss = 0.0

    for batch_idx, (src_input, tgt_input) in enumerate(train_loader):
        # 数据移到指定设备（CPU/GPU）
        src_input = src_input.to(device)
        tgt_input = tgt_input.to(device)

        # 清零梯度（必须！否则梯度会累积）
        optimizer.zero_grad()

        # 前向传播：模型输出 shape [batch_size, max_len, dec_voc_size]
        output = model(src_input, tgt_input)

        # 调整维度适配交叉熵损失：
        # 损失函数要求输入为 [batch_size*seq_len, vocab_size]，标签为 [batch_size*seq_len]
        output_flat = output.reshape(-1, output.size(-1))
        tgt_input_flat = tgt_input.reshape(-1)

        # 计算损失（ignore_index 忽略padding位置的损失）
        loss = criterion(output_flat, tgt_input_flat)

        # 反向传播：计算梯度
        loss.backward()

        # 梯度裁剪（防止梯度爆炸，Transformer必备）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 优化器更新参数
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

        # 打印训练日志
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f'Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}')

    # 返回本轮平均损失
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device, tgt_pad_idx):
    """验证函数（无梯度计算，避免显存占用）"""
    model.eval()  # 切换到验证模式
    total_loss = 0.0

    with torch.no_grad():  # 禁用梯度计算
        for src_input, tgt_input in val_loader:
            src_input = src_input.to(device)
            tgt_input = tgt_input.to(device)

            output = model(src_input, tgt_input)
            output_flat = output.reshape(-1, output.size(-1))
            tgt_input_flat = tgt_input.reshape(-1)

            loss = criterion(output_flat, tgt_input_flat)
            total_loss += loss.item()

    # 切回训练模式
    model.train()
    return total_loss / len(val_loader)

# ====================== 3. 主训练流程 ======================
if __name__ == '__main__':
    # ---------------------- 配置参数 ----------------------
    # 文件路径（替换为你的实际路径）
    ZH_FILE = './cn.test.txt'
    EN_FILE = './en.test.txt'
    
    # 设备配置（自动检测GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据参数
    MAX_LEN = 50  # 序列最大长度
    BATCH_SIZE = 16  
    
    # 模型参数（适配小数据，简化模型）
    NUM_LAYERS = 3  # 层数不宜过多，避免过拟合
    D_MODEL = 128   # 模型维度减小，降低计算量
    NUM_HEADS = 8   # 注意力头数
    D_FF = 512      # 前馈网络维度
    DROPOUT = 0.1   # dropout概率
    
    # 训练参数
    NUM_EPOCHS = 50  # 小数据需要更多轮次才能收敛
    LR = 0.0001      # 学习率（Transformer推荐小学习率）
    SAVE_PATH = './best_transformer_model.pth'  # 最优模型保存路径

    # ---------------------- 加载数据 ----------------------
    src_vocab, tgt_vocab, all_data = load_your_data(ZH_FILE, EN_FILE, MAX_LEN)
    
    # 划分训练/验证集 （这里简单使用全部数据作为训练和验证，实际项目中应划分为不同的集合）
    train_data = all_data[:]
    val_data = all_data[:]
    
    # 创建DataLoader
    train_dataset = TranslationDataset(train_data)
    val_dataset = TranslationDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ---------------------- 初始化模型 ----------------------
    # 导入你的Transformer模型（替换为实际路径）
    from transformer import Transformer  # 关键：替换为你的模型文件路径
    
    # 模型关键参数
    src_pad_idx = src_vocab.token2idx['<pad>']
    tgt_pad_idx = tgt_vocab.token2idx['<pad>']
    enc_voc_size = len(src_vocab)
    dec_voc_size = len(tgt_vocab)
    
    # 初始化模型并移到设备
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=tgt_pad_idx,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=MAX_LEN,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        device=device
    ).to(device)

    # ---------------------- 定义损失函数和优化器 ----------------------
    # 交叉熵损失：ignore_index 忽略padding位置，避免无效损失
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    
    # Adam优化器：Transformer论文推荐的参数（betas=(0.9, 0.98), eps=1e-9）
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    # ---------------------- 开始训练 ----------------------
    print("\n开始训练...")
    best_val_loss = float('inf')  # 记录最优验证损失
    train_losses = []  # 保存每轮训练损失
    val_losses = []    # 保存每轮验证损失

    for epoch in range(NUM_EPOCHS):
        # 训练
        train_loss = train(model, train_loader, criterion, optimizer, device, tgt_pad_idx, epoch)
        # 验证
        val_loss = validate(model, val_loader, criterion, device, tgt_pad_idx)
        
        # 保存损失曲线数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 打印本轮结果
        print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}] Summary:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最优模型（验证损失下降时保存）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab
            }, SAVE_PATH)
            print(f'最优模型已保存！当前最优验证损失: {best_val_loss:.4f}')

    # ---------------------- 训练完成：绘制损失曲线 ----------------------
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, NUM_EPOCHS+1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('./loss_curve.png')
    plt.show()
