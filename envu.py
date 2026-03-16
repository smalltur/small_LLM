import torch
import torch.nn as nn

# 复用Vocabulary类（和训练时一致）
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

def tokenize_en(text):
    return text.strip().lower().split()

# 修正模型加载函数
def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    from transformer import Transformer
    # 确保模型参数和训练时完全一致
    model = Transformer(
        src_pad_idx=src_vocab.token2idx['<pad>'],
        trg_pad_idx=tgt_vocab.token2idx['<pad>'],
        enc_voc_size=len(src_vocab),
        dec_voc_size=len(tgt_vocab),
        max_len=50,
        num_layers=3,
        d_model=128,  # 必须能被num_heads=8整除（128/8=16）
        num_heads=8,
        d_ff=512,
        dropout=0.1,
        device=device
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载完成！最优验证损失: {checkpoint['best_val_loss']:.4f}")
    return model, src_vocab, tgt_vocab

# 修正翻译函数
def translate(model, src_text, src_vocab, tgt_vocab, device, max_len=50, batch_size=8):
    """
    兼容单文本输入的批量翻译函数（修复类型错误）
    :param src_text: 单个源文本字符串（原调用方式）
    """
    # 关键修复：将单文本转为列表，适配批量逻辑
    src_texts = [src_text]  # 把单个字符串转为长度为1的列表
    
    # 0. 补全到batch_size（不足则用空文本填充）
    pad_text = ""  # 空文本
    if len(src_texts) < batch_size:
        # 补全空文本到batch_size
        src_texts += [pad_text] * (batch_size - len(src_texts))
    
    # 1. 批量预处理输入文本
    src_ids_batch = []
    pad_idx = src_vocab.token2idx['<pad>']
    sos_idx = src_vocab.token2idx['<sos>']
    eos_idx = src_vocab.token2idx['<eos>']
    
    for text in src_texts:
        src_tokens = tokenize_en(text)
        src_ids = [sos_idx] + src_vocab.convert_tokens_to_ids(src_tokens) + [eos_idx]
        # 截断/补全到max_len
        if len(src_ids) > max_len:
            src_ids = src_ids[:max_len]
        else:
            src_ids += [pad_idx] * (max_len - len(src_ids))
        src_ids_batch.append(src_ids)
    
    # 转换为batch张量：[8, max_len]
    src_input = torch.tensor(src_ids_batch, dtype=torch.long).to(device)
    
    # 2. 初始化目标序列：[8, 1]
    tgt_input = torch.tensor([[tgt_vocab.token2idx['<sos>']]] * batch_size, dtype=torch.long).to(device)
    
    # 3. 批量自回归生成
    model.eval()  # 推理模式
    with torch.no_grad():
        for _ in range(max_len - 1):
            output = model(src_input, tgt_input)
            # 取最后一个token的预测
            next_token_logits = output[:, -1, :]
            next_token_idx = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            # 拼接
            tgt_input = torch.cat([tgt_input, next_token_idx], dim=1)
            # 检查是否所有样本都生成<eos>（提前终止）
            eos_mask = (next_token_idx == tgt_vocab.token2idx['<eos>']).squeeze(1)
            if torch.all(eos_mask):
                break
    
    # 4. 解码结果（只取第一个样本的结果，因为输入是单文本）
    tgt_eos_idx = tgt_vocab.token2idx['<eos>']
    tgt_sos_idx = tgt_vocab.token2idx['<sos>']
    tgt_pad_idx = tgt_vocab.token2idx['<pad>']
    
    # 只解码第一个样本（对应输入的src_text）
    tgt_ids = tgt_input[0].cpu().numpy()
    tgt_tokens = []
    for idx in tgt_ids:
        token = tgt_vocab.idx2token.get(idx, '<unk>')
        if token in ['<sos>', '<pad>']:
            continue
        if token == '<eos>':
            break
        tgt_tokens.append(token)
    
    return ' '.join(tgt_tokens)

# 主函数
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = './best_transformer_model.pth'
    
    # 加载模型
    model, src_vocab, tgt_vocab = load_model(checkpoint_path, device)
    
    # 测试翻译
    test_sentences = [
        "wahid"
    ]
    
    print("\n===== 翻译测试结果 =====")
    for i, src_text in enumerate(test_sentences):
        translated_text = translate(model, src_text, src_vocab, tgt_vocab, device)
        print(f"\n测试句子 {i+1}:")
        print(f"英文输入: {src_text}")
        print(f"中文翻译: {translated_text}")