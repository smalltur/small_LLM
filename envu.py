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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
def translate(model, src_text, src_vocab, tgt_vocab, device, max_len=50):
    # 1. 预处理输入文本（确保长度不超过max_len）
    src_tokens = tokenize_en(src_text)
    src_ids = [src_vocab.token2idx['<sos>']] + src_vocab.convert_tokens_to_ids(src_tokens) + [src_vocab.token2idx['<eos>']]
    pad_idx = src_vocab.token2idx['<pad>']
    
    # 关键修正：截断/补全到max_len，确保维度正确
    if len(src_ids) > max_len:
        src_ids = src_ids[:max_len]
    else:
        src_ids += [pad_idx] * (max_len - len(src_ids))
    
    # 添加batch维度：[1, max_len]
    src_input = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # 2. 初始化目标序列（仅<sos>，长度1）
    tgt_input = torch.tensor([[tgt_vocab.token2idx['<sos>']]], dtype=torch.long).to(device)
    
    # 3. 自回归生成（禁用梯度）
    with torch.no_grad():
        for _ in range(max_len - 1):
            # 每次生成都调用model.forward，动态生成掩码
            output = model(src_input, tgt_input)  # [1, len(tgt_input), dec_voc_size]
            
            # 取最后一个token的预测结果
            next_token_logits = output[:, -1, :]
            next_token_idx = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # 拼接新token
            tgt_input = torch.cat([tgt_input, next_token_idx], dim=1)
            
            # 终止条件
            if next_token_idx.item() == tgt_vocab.token2idx['<eos>']:
                break
    
    # 4. 解码结果
    tgt_ids = tgt_input.squeeze(0).cpu().numpy()
    tgt_tokens = []
    for idx in tgt_ids:
        token = tgt_vocab.idx2token.get(idx, '<unk>')
        if token in ['<pad>', '<sos>', '<eos>']:
            continue
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