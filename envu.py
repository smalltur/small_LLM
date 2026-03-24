import torch
import torch.nn.functional as F
import jieba
import os
from collections import defaultdict

# ====================== 1. 复用训练代码中的Vocabulary类和数据加载逻辑 ======================
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
    text = text.strip().replace(' ', '')
    return jieba.lcut(text)

def tokenize_en(text):
    return text.strip().lower().split()

def rebuild_train_vocab(train_zh_file, train_en_file):
    """
    重新生成和训练完全一致的词表（必须用训练数据文件）
    :param train_zh_file: 训练集中文文件路径（和训练时一致）
    :param train_en_file: 训练集英文文件路径（和训练时一致）
    :return: src_vocab（英文）, tgt_vocab（中文）
    """
    # 1. 初始化空词表（和训练时完全一样）
    src_vocab = Vocabulary()  # 英文词表
    tgt_vocab = Vocabulary()  # 中文词表

    # 2. 读取训练数据文件（必须用训练数据，不是测试数据）
    zh_lines = []
    en_lines = []
    if not os.path.exists(train_zh_file) or not os.path.exists(train_en_file):
        raise FileNotFoundError(f"找不到训练数据文件！请检查 {train_zh_file} 和 {train_en_file}")
    
    with open(train_zh_file, 'r', encoding='utf-8') as f_zh, \
         open(train_en_file, 'r', encoding='utf-8') as f_en:
        for zh_line, en_line in zip(f_zh, f_en):
            zh_clean = zh_line.strip()
            en_clean = en_line.strip()
            if zh_clean and en_clean:
                zh_lines.append(zh_clean)
                en_lines.append(en_clean)

    # 3. 遍历训练数据，构建词表（和训练时的逻辑完全一致）
    for zh_line, en_line in zip(zh_lines, en_lines):
        zh_tokens = tokenize_zh(zh_line)
        en_tokens = tokenize_en(en_line)
        
        # 添加token到词表（顺序和训练时完全一致，保证ID相同）
        for token in zh_tokens:
            tgt_vocab.add_token(token)
        for token in en_tokens:
            src_vocab.add_token(token)

    print(f"✅ 重新生成训练同款词表完成！")
    print(f"英文词表大小: {len(src_vocab.token2idx)}, 中文词表大小: {len(tgt_vocab.token2idx)}")
    print(f"英文词表示例（前20个token及ID）: {list(src_vocab.token2idx.items())[:20]}")
    return src_vocab, tgt_vocab

# ====================== 辅助函数：N-gram阻塞核心逻辑 ======================
def get_ngrams(sequence, n):
    """
    从已生成的序列中提取所有n-gram元组
    :param sequence: 已生成的token ID列表
    :param n: n-gram的n值（如2表示二元组）
    :return: 所有n-gram元组的集合
    """
    ngrams = set()
    if len(sequence) < n:
        return ngrams
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i+n])
        ngrams.add(ngram)
    return ngrams

def block_ngram_candidates(next_token_logits, generated_tokens, candidate_ids, n, device):
    """
    阻塞会导致重复n-gram的候选token
    :param next_token_logits: 模型输出的下一个token的logits
    :param generated_tokens: 已生成的token ID列表
    :param candidate_ids: top-k筛选后的候选token ID列表
    :param n: n-gram的n值
    :param device: 设备（cpu/gpu）
    :return: 处理后的logits（重复n-gram对应的token logits设为-∞）
    """
    if len(generated_tokens) < n-1:
        return next_token_logits  # 序列长度不足，无需阻塞
    
    # 提取已生成序列的(n-1)-gram（前缀）
    prefix = tuple(generated_tokens[-(n-1):])
    # 遍历所有候选token，检查是否会形成重复的n-gram
    all_ngrams = get_ngrams(generated_tokens, n)
    
    blocked_ids = []
    for token_id in candidate_ids:
        # 拼接前缀+当前token，形成新的n-gram
        new_ngram = prefix + (token_id,)
        if new_ngram in all_ngrams:
            blocked_ids.append(token_id)
    
    # 阻塞重复n-gram对应的token（将其logits设为-∞）
    if blocked_ids:
        next_token_logits[:, blocked_ids] = -1e9
        print(f"  - N-gram阻塞：禁止了 {len(blocked_ids)} 个会导致重复{n}元组的token ID: {blocked_ids}")
    
    return next_token_logits

# ====================== 2. 翻译函数（新增N-gram阻塞功能） ======================
def translate(model, src_text, src_vocab, tgt_vocab, device, max_len=50, n_gram_block=10):
    """终极修复版：新增N-gram阻塞 + 压制标点 + 提升中文词权重 + 动态采样"""
    # 1. 预处理输入文本
    src_tokens = tokenize_en(src_text)
    pad_idx = src_vocab.token2idx['<pad>']
    sos_idx = src_vocab.token2idx['<sos>']
    eos_idx = src_vocab.token2idx['<eos>']
    
    # ========== 输入文本编码详情 ==========
    print("\n===== 输入文本编码详情 =====")
    print(f"原始输入文本: {src_text}")
    print(f"分词后的token列表: {src_tokens}")
    # 计算每个token对应的ID（不含sos/eos/pad）
    src_token_ids = [src_vocab.token2idx.get(token, src_vocab.token2idx['<unk>']) for token in src_tokens]
    print(f"token对应的原始ID: {src_token_ids}")
    
    # 构建完整序列（sos + token_ids + eos + pad）
    src_ids = [sos_idx] + src_token_ids + [eos_idx]
    src_ids = src_ids[:max_len] if len(src_ids) > max_len else src_ids + [pad_idx] * (max_len - len(src_ids))
    print(f"添加特殊token+填充后的完整ID序列: {src_ids}")
    print(f"序列长度: {len(src_ids)} (max_len={max_len})")
    print("===========================\n")
    
    src_input = torch.tensor([src_ids], dtype=torch.long).to(device)
    # 打印源输入最终形态
    print(f"【源输入最终形态】形状: {src_input.shape}, 设备: {src_input.device}")
    print(f"源输入具体数值: {src_input.tolist()[0]}")
    print("-" * 80)

    # 2. 初始化目标序列
    tgt_sos_idx = tgt_vocab.token2idx['<sos>']
    tgt_eos_idx = tgt_vocab.token2idx['<eos>']
    tgt_input = torch.tensor([[tgt_sos_idx]], dtype=torch.long).to(device)
    
    # 关键：获取标点token的索引（压制高频标点）
    punctuation_tokens = [',', '。', '，', '、', '；', '：', '！', '？']
    punctuation_ids = [tgt_vocab.token2idx.get(p, -1) for p in punctuation_tokens if p in tgt_vocab.token2idx]
    
    model.eval()
    generated_tokens = []
    min_gen_len = 1 # 强制至少生成1个有效token
    temperature = 0.2  # 降低温度，减少随机，提升准确性
    top_k = 30         # 扩大采样范围，覆盖更多中文词
    top_p = 0.9        # 新增top-p采样，过滤低概率token

    with torch.no_grad():
        for step in range(max_len):
            # ========== 打印每一步传入模型的输入数据 ==========
            print(f"\n===== 生成步骤 {step+1}/{max_len} =====")
            print(f"【传入模型的源输入】")
            print(f"  - 形状: {src_input.shape}")
            print(f"  - 数值（前20个）: {src_input.tolist()[0][:20]} {'...' if len(src_ids)>20 else ''}")
            print(f"【传入模型的目标输入】")
            print(f"  - 形状: {tgt_input.shape}")
            print(f"  - 数值: {tgt_input.tolist()[0]}")  # 目标序列较短，打印完整数值
            
            # 前向传播
            output = model(src_input, tgt_input)
            next_token_logits = output[:, -1, :]
            
            # ========== 打印模型输出关键信息 ==========
            print(f"【模型输出信息】")
            print(f"  - 解码器输出形状: {output.shape}")
            print(f"  - 最后一个token的logits维度: {next_token_logits.shape}")
            print(f"  - 最后一个token的logits最大值: {next_token_logits.max().item():.4f}, 最小值: {next_token_logits.min().item():.4f}")
            
            # ========== 核心修复 1：压制高频标点 ==========
            for p_id in punctuation_ids:
                if p_id != -1:
                    # 前5步完全禁止标点，后续降低标点概率
                    if step < 5:
                        next_token_logits[:, p_id] = -1e9
                    else:
                        next_token_logits[:, p_id] *= 0.1  # 标点概率×0.1
            
            # ========== 核心修复 2：强制前min_gen_len步不输出<eos> ==========
            if step < min_gen_len:
                next_token_logits[:, tgt_eos_idx] = -1e9
                print(f"  - 强制禁止<eos>（当前步数{step} < 最小生成长度{min_gen_len}）")
            else:
                next_token_logits[:, tgt_eos_idx] *= 0.3  # 降低结束符概率
                print(f"  - 降低<eos>概率（×0.3），当前步数{step} ≥ 最小生成长度{min_gen_len}")
            
            # ========== 核心新增：N-gram阻塞 ==========
            # 先获取top-k候选token ID（用于后续阻塞检查）
            top_k_vals_temp, top_k_idx_temp = torch.topk(next_token_logits, k=top_k, dim=-1)
            candidate_ids = top_k_idx_temp[0].tolist()
            
            # 滑动窗口重复阻塞：禁止候选token出现在最近window_size个已生成的词中
            window_size = n_gram_block  # 将参数直接用作窗口大小（此处传入10）
            if generated_tokens:
                recent_tokens = generated_tokens[-window_size:]  # 最近window_size个词（不足则取全部）
                blocked_ids = [token_id for token_id in candidate_ids if token_id in recent_tokens]
                if blocked_ids:
                    next_token_logits[:, blocked_ids] = -1e9
                    print(f"  - 滑动窗口阻塞：禁止了 {len(blocked_ids)} 个与最近{len(recent_tokens)}个词重复的token ID: {blocked_ids}")
            
            # ========== 温度缩放、top-k + top-p采样（与原代码一致） ==========
            next_token_logits = next_token_logits / temperature
            top_k_vals, top_k_idx = torch.topk(next_token_logits, k=top_k, dim=-1)
            top_k_probs = F.softmax(top_k_vals, dim=-1)
            cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
            mask = cumulative_probs <= top_p
            mask[:, 0] = True
            filtered_probs = top_k_probs * mask.float()
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            
            sample_idx = torch.multinomial(filtered_probs, num_samples=1)
            next_token_idx = top_k_idx.gather(1, sample_idx).item()
            
            # ========== 打印采样结果 ==========
            print(f"【采样结果】")
            print(f"  - Top-{top_k}中概率最高的token ID: {top_k_idx[0,0].item()} (概率: {top_k_probs[0,0].item():.4f})")
            print(f"  - 最终采样的下一个token ID: {next_token_idx}")
            print(f"  - 对应的token: {tgt_vocab.idx2token.get(next_token_idx, '<unk>')}")
            
            # 记录并拼接token
            generated_tokens.append(next_token_idx)
            tgt_input = torch.cat([tgt_input, torch.tensor([[next_token_idx]], device=device)], dim=1)
            
            # 满足最小长度后，遇到<eos>停止
            if next_token_idx == tgt_eos_idx and step >= min_gen_len:
                print(f"\n✅ 生成终止：遇到<eos>且满足最小长度（步数{step} ≥ {min_gen_len}）")
                break

    # 3. 解码为中文（过滤无效token）
    tgt_tokens = []
    for idx in generated_tokens:
        token = tgt_vocab.idx2token.get(idx, '<unk>')
        if token in ['<sos>', '<pad>', '<eos>', '<unk>']:
            continue
        tgt_tokens.append(token)
    
    # 兜底逻辑
    if len(tgt_tokens) == 0:
        return "[模型未生成有效翻译]"
    
    # 拼接结果，清理多余标点
    result = ''.join(tgt_tokens)
    # 移除开头/结尾的标点
    result = result.strip(',，。、；：！？')
    
    # ========== 打印最终生成结果汇总 ==========
    print("\n===== 生成结果汇总 =====")
    print(f"生成的token ID列表: {generated_tokens}")
    print(f"过滤后的有效token: {tgt_tokens}")
    print(f"最终翻译结果: {result}")
    print(f"N-gram阻塞配置: {n_gram_block}元组")
    print("========================\n")
    
    return result if result else "[模型未生成有效翻译]"

# ====================== 3. 主预测流程（重新生成词表） ======================
if __name__ == '__main__':
    # 配置路径（必须和训练时的路径完全一致！）
    TRAIN_ZH_FILE = './cn.txt'        # 训练集中文文件（和训练时一致）
    TRAIN_EN_FILE = './en.txt'        # 训练集英文文件（和训练时一致）
    MODEL_PATH = './best_transformer_model.pth'  # 训练好的模型路径

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== 关键步骤：重新生成和训练完全一致的词表 ==========
    src_vocab, tgt_vocab = rebuild_train_vocab(TRAIN_ZH_FILE, TRAIN_EN_FILE)
    
    # ========== 加载模型（只加载权重，不加载保存的词表） ==========
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # 初始化模型（参数必须和训练时完全一致！）
    from transformer import Transformer
    model = Transformer(
        src_pad_idx=src_vocab.token2idx['<pad>'],
        trg_pad_idx=tgt_vocab.token2idx['<pad>'],
        enc_voc_size=len(src_vocab),
        dec_voc_size=len(tgt_vocab),
        max_len=50,
        num_layers=3,
        d_model=128,
        num_heads=8,
        d_ff=512,
        dropout=0.1,
        device=device
    ).to(device)
    # 只加载模型权重，忽略保存的词表
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试翻译（可调整n_gram_block参数，建议2或3）
    test_sentence = "it turned out that the rocket 's range lengthened after some of the propellant was removed"
    result = translate(model, test_sentence, src_vocab, tgt_vocab, device, n_gram_block=10)
    
    print("===== 翻译测试结果 =====")
    print(f"\n测试句子 1:")
    print(f"英文输入: {test_sentence}")
    print(f"中文翻译: {result}")