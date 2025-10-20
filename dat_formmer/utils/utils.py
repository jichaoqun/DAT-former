from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from dat_formmer.datamodule import vocab
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric


class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        for pred, truth in zip(indices_hat, indices):
            pred = vocab.indices2label(pred)
            truth = vocab.indices2label(truth)

            is_same = pred == truth

            if is_same:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate


def ce_loss(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    ignore_idx: int = vocab.PAD_IDX,
    reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    # loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction, label_smoothing=0.1)
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction, )

    return loss


def structure_loss(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    ignore_idx: int = 0,
    reduction: str = "mean",
    ) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    flag = flat != ignore_idx

    # loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction, label_smoothing=0.1)
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction='none')
    loss = loss[flag].mean()

    return loss

def to_tgt_output(
    tokens: Union[List[List[int]], List[LongTensor]],
    direction: str,
    device: torch.device,
    pad_to_len: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
    tokens: List[List[int]], device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out



def to_bi_cursor_pos(
    cursor_pos_list: List[List[int]], device: torch.device, pad_to_len: Optional[int] = None
) -> torch.LongTensor:
    """
    构建双向结构位置信息 cursor_pos 张量（与 to_bi_tgt_out 对应）

    参数:
        cursor_pos_list: List of [L_i], 每个样本对应的结构位置信息
        device: 当前使用的设备（torch.device）
        pad_to_len: 若指定，则将序列 pad 到指定长度

    返回:
        cursor_pos_tensor: [2B, max_len] 长度的 tensor
    """

    # Step 1: 构建正向与反向 cursor pos
    l2r_cursor = [torch.tensor(c, dtype=torch.long) for c in cursor_pos_list]
    r2l_cursor = [torch.flip(c, dims=[0]) for c in l2r_cursor]

    all_cursor = l2r_cursor + r2l_cursor  # [2B]

    # Step 2: 确定最大长度
    lengths = [len(c) for c in all_cursor]
    max_len = max(lengths)
    if pad_to_len is not None:
        max_len = max(max_len, pad_to_len)

    # Step 3: padding
    padded_cursor = torch.full(
        (len(all_cursor), max_len), fill_value=0, dtype=torch.long, device=device
    )

    for i, c in enumerate(all_cursor):
        padded_cursor[i, :len(c)] = c

    return padded_cursor  # shape: [2B, max_len]






from typing import List, Tuple, Optional, Dict
import torch
from torch import LongTensor

# 假设这是您的结构词汇表
# 确保 <pad> 是 0, default 是 1
# CURSOR_VOCAB = {
#     '<pad>': 0,
#     'default': 1,     # 默认位置, 也用作 <SOS> 和 <EOS> 的位置
#     'frac_num': 2,    # 分子
#     'frac_den': 3,    # 分母
#     'sqrt_inner': 4,  # 根号内
#     'subscript': 5,   # 下标
#     'superscript': 6, # 上标
#     'integral': 7,    # 积分体
#     'lim': 8          # 极限参数
# }

from comer.datamodule.datamodule import CURSOR_VOCAB

# CURSOR_VOCAB = {
#             '<pad>': 0,
#             '<sos>': 1,  # 开始符
#             '<eos>': 2,  # 结束符
#             # 位置标记
#             'default': 3,     # 默认位置（无嵌套结构）
#             'frac_num': 4,    # 分子
#             'frac_den': 5,    # 分母
#             'sqrt_inner': 6,  # 根号内
#             'subscript': 7,   # 下标
#             'superscript': 8, # 上标
#             'integral': 9,    # 积分体
#             'lim': 10          # 极限参数
#         }

def _to_single_direction_cursor_pos(
    cursor_pos_list: List[List[int]],
    direction: str,
    device: torch.device,
    cursor_vocab: Dict[str, int],
    pad_to_len: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor]:
    """
    为单向解码器生成对齐的 cursor_pos 输入和目标张量。
    此函数逻辑严格模仿 to_tgt_output。
    """
    assert direction in {"l2r", "r2l"}

    # 1. 准备数据和特殊标记
    pos_tokens = [torch.tensor(p, dtype=torch.long) for p in cursor_pos_list]
    start_pos = cursor_vocab['<sos>']
    stop_pos = cursor_vocab['<eos>']
    pad_pos = cursor_vocab['<pad>']

    if direction == "r2l":
        pos_tokens = [torch.flip(p, dims=[0]) for p in pos_tokens]

    # 2. 计算与主任务完全一致的长度
    batch_size = len(pos_tokens)
    lens = [len(p) for p in pos_tokens]
    
    # 关键点：长度必须是 max(lens) + 1，以容纳 <SOS> 或 <EOS>
    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    # 3. 创建输入和目标张量
    cursor_input = torch.full(
        (batch_size, length),
        fill_value=pad_pos,
        dtype=torch.long,
        device=device,
    )
    cursor_target = torch.full(
        (batch_size, length),
        fill_value=pad_pos,
        dtype=torch.long,
        device=device,
    )

    # 4. 填充张量，模仿 to_tgt_output 的逻辑
    for i, p_token in enumerate(pos_tokens):
        # 构造 cursor_input: [start_pos, p1, p2, ...]
        cursor_input[i, 0] = start_pos
        cursor_input[i, 1 : (1 + lens[i])] = p_token

        # 构造 cursor_target: [p1, p2, ..., stop_pos]
        cursor_target[i, : lens[i]] = p_token
        cursor_target[i, lens[i]] = stop_pos

    return cursor_input, cursor_target


def build_bi_cursor_pos_input_target(
    cursor_pos_list: List[List[int]], 
    device: torch.device,
    cursor_vocab: Dict[str, int] = CURSOR_VOCAB
) -> Tuple[LongTensor, LongTensor]:
    """
    构建与 to_bi_tgt_out 完全对齐的双向结构位置输入和目标张量。

    参数:
        cursor_pos_list: List of [L_i], 每个样本对应的原始结构位置信息 (不含<SOS>/<EOS>)
        device: 当前使用的设备（torch.device）
        cursor_vocab: 结构词汇表

    返回:
        Tuple[LongTensor, LongTensor]:
            - cursor_pos_input: [2B, max_len], 用于辅助解码器的输入
            - cursor_pos_target: [2B, max_len], 用于计算辅助 loss 的目标
    """
    # 确定可能的 padding 目标长度
    # 主任务的 l2r_tgt 和 r2l_tgt 长度可能不同，这里分别计算
    l2r_tgt_len = max([len(t) for t in cursor_pos_list]) + 1
    
    # 1. 生成 L2R 的输入和目标
    l2r_input, l2r_target = _to_single_direction_cursor_pos(
        cursor_pos_list, "l2r", device, cursor_vocab, pad_to_len=l2r_tgt_len
    )

    # 2. 生成 R2L 的输入和目标
    r2l_input, r2l_target = _to_single_direction_cursor_pos(
        cursor_pos_list, "r2l", device, cursor_vocab, pad_to_len=l2r_tgt_len
    )
    
    # 3. 合并
    cursor_pos_input = torch.cat((l2r_input, r2l_input), dim=0)
    cursor_pos_target = torch.cat((l2r_target, r2l_target), dim=0)

    return cursor_pos_input, cursor_pos_target




