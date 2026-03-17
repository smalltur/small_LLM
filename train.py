import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
import os
import matplotlib.pyplot as plt

# ====================== 1. 数据加载代码（支持独立验证集） ======================
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

# 中文分词（使用jieba）
def tokenize_zh(text):
    text = text.strip().replace(' ', '')  # 清洗多余空格
    return jieba.lcut(text)  # jieba精准分词

def tokenize_en(text):
    return text.strip().lower().split()

# 加载单组数据（训练/验证）
def load_single_dataset(zh_file_path, en_file_path, max_len=50, vocab=None, is_train=True):
    """
    加载数据并可选构建词表
    :param vocab: 训练集的词表（验证集时传入，避免OOV）
    :param is_train: 是否是训练集（训练集构建词表，验证集复用）
    """
    # 读取文件
    zh_lines = []
    en_lines = []
    if not os.path.exists(zh_file_path) or not os.path.exists(en_file_path):
        raise FileNotFoundError(f"找不到数据文件！请检查 {zh_file_path} 和 {en_file_path} 是否存在")
    
    with open(zh_file_path, 'r', encoding='utf-8') as f_zh, \
         open(en_file_path, 'r', encoding='utf-8') as f_en:
        for zh_line, en_line in zip(f_zh, f_en):
            zh_clean = zh_line.strip()
            en_clean = en_line.strip()
            if zh_clean and en_clean:  # 过滤空行
                zh_lines.append(zh_clean)
                en_lines.append(en_clean)

    # 构建/复用词表
    if is_train:
        src_vocab = Vocabulary()  # 英文词表（源）
        tgt_vocab = Vocabulary()  # 中文词表（目标）
    else:
        src_vocab, tgt_vocab = vocab  # 复用训练集词表

    # 处理token并构建词表（仅训练集）
    data = []
    src_pad_idx = src_vocab.token2idx['<pad>']
    tgt_pad_idx = tgt_vocab.token2idx['<pad>']
    
    for zh_line, en_line in zip(zh_lines, en_lines):
        zh_tokens = tokenize_zh(zh_line)
        en_tokens = tokenize_en(en_line)
        
        # 训练集：添加token到词表
        if is_train:
            for token in zh_tokens:
                tgt_vocab.add_token(token)
            for token in en_tokens:
                src_vocab.add_token(token)
        
        # 构建序列：<sos> + 正文 + <eos>
        # 源序列（src）= 英文 → 对应src_vocab
        src_ids = [src_vocab.token2idx['<sos>']] + src_vocab.convert_tokens_to_ids(en_tokens) + [src_vocab.token2idx['<eos>']]
        # 目标序列（tgt）= 中文 → 对应tgt_vocab
        tgt_ids = [tgt_vocab.token2idx['<sos>']] + tgt_vocab.convert_tokens_to_ids(zh_tokens) + [tgt_vocab.token2idx['<eos>']]
        
        # 截断/补全到max_len
        src_ids = src_ids[:max_len] if len(src_ids) > max_len else src_ids + [src_pad_idx]*(max_len - len(src_ids))
        tgt_ids = tgt_ids[:max_len] if len(tgt_ids) > max_len else tgt_ids + [tgt_pad_idx]*(max_len - len(tgt_ids))
        
        data.append((src_ids, tgt_ids))

    print(f"{'训练' if is_train else '验证'}集加载完成！样本数: {len(data)}")
    if is_train:
        print(f"英文词表大小: {len(src_vocab)}, 中文词表大小: {len(tgt_vocab)}")
    # 仅打印最后一条序列用于验证
    if len(data) > 0:
        print("示例中文序列:", tgt_ids[:20])  # 只打印前20个token，避免刷屏
    
    # ========== 新增：打印前20条数据集 ==========
    print(f"\n===== 打印前20条{'训练' if is_train else '验证'}数据集 =====")
    display_num = min(20, len(data))  # 避免数据不足20条的情况
    for i in range(display_num):
        # 获取原始句子
        zh_sent = zh_lines[i]
        en_sent = en_lines[i]
        # 获取编码后的序列
        src_seq = data[i][0]
        tgt_seq = data[i][1]
        
        # 打印格式：序号 | 英文原文 | 中文原文 | 英文ID序列（前10） | 中文ID序列（前10）
        print(f"\n【第{i+1}条】")
        print(f"英文原文: {en_sent}")
        print(f"中文原文: {zh_sent}")
        print(f"英文ID序列（前10）: {src_seq[:10]}")
        print(f"中文ID序列（前10）: {tgt_seq[:10]}")
    
    return (src_vocab, tgt_vocab, data) if is_train else data

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# ====================== 2. 训练/验证核心函数 ======================
def train(model, train_loader, criterion, optimizer, device, tgt_pad_idx, epoch, log_interval=10):
    """单轮训练函数（教师强制）"""
    model.train()
    total_loss = 0.0

    for batch_idx, (src_input, tgt_full) in enumerate(train_loader):
        src_input = src_input.to(device)
        tgt_full = tgt_full.to(device)

        # 教师强制：输入去掉最后一个token，标签去掉第一个token
        tgt_input = tgt_full[:, :-1]
        tgt_label = tgt_full[:, 1:]

        optimizer.zero_grad()
        output = model(src_input, tgt_input)

        # 调整维度计算损失
        output_flat = output.reshape(-1, output.size(-1))
        tgt_label_flat = tgt_label.reshape(-1)
        loss = criterion(output_flat, tgt_label_flat)

        # 反向传播 + 梯度裁剪
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # 打印日志
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f'Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}')

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device, tgt_pad_idx):
    """验证函数"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src_input, tgt_full in val_loader:
            src_input = src_input.to(device)
            tgt_full = tgt_full.to(device)

            # 同样应用教师强制
            tgt_input = tgt_full[:, :-1]
            tgt_label = tgt_full[:, 1:]

            output = model(src_input, tgt_input)
            output_flat = output.reshape(-1, output.size(-1))
            tgt_label_flat = tgt_label.reshape(-1)

            loss = criterion(output_flat, tgt_label_flat)
            total_loss += loss.item()

    model.train()
    return total_loss / len(val_loader)

# ====================== 3. 主训练流程 ======================
if __name__ == '__main__':
    # ---------------------- 配置参数 ----------------------
    # 数据文件路径（训练集+独立验证集）
    TRAIN_ZH_FILE = './c.txt'        # 训练集中文
    TRAIN_EN_FILE = './e.txt'        # 训练集英文
    VAL_ZH_FILE = './c.txt'     # 验证集中文
    VAL_EN_FILE = './e.txt'     # 验证集英文
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据参数
    MAX_LEN = 50
    BATCH_SIZE = 16  
    
    # 模型参数
    NUM_LAYERS = 3
    D_MODEL = 128
    NUM_HEADS = 8
    D_FF = 512
    DROPOUT = 0.1
    
    # 训练参数（关键修正：降低学习率，避免梯度爆炸）
    NUM_EPOCHS = 50
    LR = 0.0001  # 修正：从0.001降到0.0001，Transformer推荐小学习率
    SAVE_PATH = './best_transformer_model.pth'
    PATIENCE = 10  # 修正：从20降到10，避免过拟合

    # ---------------------- 加载数据 ----------------------
    print("\n========== 加载训练集 ==========")
    # 加载训练集并构建词表（参数顺序正确：中文文件在前，英文文件在后）
    src_vocab, tgt_vocab, train_data = load_single_dataset(
        TRAIN_ZH_FILE, TRAIN_EN_FILE, MAX_LEN, is_train=True
    )
    
    print("\n========== 加载验证集 ==========")
    # 加载验证集（复用训练集词表，避免OOV）
    val_data = load_single_dataset(
        VAL_ZH_FILE, VAL_EN_FILE, MAX_LEN, vocab=(src_vocab, tgt_vocab), is_train=False
    )
    
    # 创建DataLoader
    train_dataset = TranslationDataset(train_data)
    val_dataset = TranslationDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------------- 初始化模型 ----------------------
    try:
        from transformer import Transformer
        print("\n成功导入Transformer模型！")
    except ImportError as e:
        raise ImportError(f"导入Transformer失败: {e}\n请确保transformer.py文件在当前目录")
    
    # 模型初始化
    src_pad_idx = src_vocab.token2idx['<pad>']
    tgt_pad_idx = tgt_vocab.token2idx['<pad>']
    enc_voc_size = len(src_vocab)
    dec_voc_size = len(tgt_vocab)
    
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

    # ---------------------- 损失函数 + 优化器 ----------------------
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    # Adam优化器：使用Transformer论文推荐参数
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    # ---------------------- 开始训练 ----------------------
    print("\n======= 开始训练 =======")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # 训练
        train_loss = train(model, train_loader, criterion, optimizer, device, tgt_pad_idx, epoch)
        # 验证（用独立验证集）
        val_loss = validate(model, val_loader, criterion, device, tgt_pad_idx)
        
        # 保存损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 打印本轮结果
        print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}] Summary:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Best Val Loss: {best_val_loss:.4f}')

        # 保存最优模型
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
            print(f'✅ 最优模型已保存！当前最优验证损失: {best_val_loss:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'⚠️ 验证损失未下降，耐心值: {patience_counter}/{PATIENCE}')
            if patience_counter >= PATIENCE:
                print(f'🛑 早停触发！训练提前终止（Epoch {epoch+1}）')
                break

    # ---------------------- 绘制损失曲线 ----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=训练完成 ==")
    print(f"最终最优验证损失: {best_val_loss:.4f}")
    print(f"最优模型保存路径: {SAVE_PATH}")