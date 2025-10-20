import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm1d


class MaskBatchNorm2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [b, d, h, w]
        mask : Tensor
            [b, 1, h, w]

        Returns
        -------
        Tensor
            [b, d, h, w]
        """
        x = rearrange(x, "b d h w -> b h w d")
        mask = mask.squeeze(1)

        not_mask = ~mask

        flat_x = x[not_mask, :]
        flat_x = self.bn(flat_x)
        x[not_mask, :] = flat_x

        x = rearrange(x, "b h w d -> b d h w")

        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage

        if cross_coverage and self_coverage:
            in_chs = 2 * nhead
        else:
            in_chs = nhead

        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)

        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.post_norm = MaskBatchNorm2d(nhead)

        # ======= 新增：可学习的门控网络 =======
        # 它的任务是为每个时间步的注意力图生成一个“遗忘系数” (0到1)
        # 输入是注意力图本身，输出是一个标量门控值
        self.gating_network = nn.Sequential(
            nn.Conv2d(in_chs, in_chs // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 使用全局平均池化将 HxW 的特征图压缩成一个向量
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_chs // 2, in_chs, kernel_size=1), # 1x1 卷积代替线性层
            nn.Sigmoid() # Sigmoid函数确保输出在 0 到 1 之间
        )
        # ======================================

    def forward(
        self, prev_attn: Tensor, key_padding_mask: Tensor, h: int, curr_attn: Tensor,tgt_vocab: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        prev_attn : Tensor
            [(b * nhead), t, l]
        key_padding_mask : Tensor
            [b, l]
        h : int

        Returns
        -------
        Tensor
            [(b * nhead), t, l]
        """
        t = curr_attn.shape[1]
        b = prev_attn.shape[0] // self.nhead
        w = key_padding_mask.shape[1] // h
        mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t)

        curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
        prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)
        b=curr_attn.shape[0] // 2
        # tgt_vocab=tgt_vocab.repeat(self.nhead,1)
        attns = []
        if self.cross_coverage:
            attns.append(prev_attn)
        if self.self_coverage:
            attns.append(curr_attn)
        attns_bntl = torch.cat(attns, dim=1)

        # ======= 应用可学习的门控 =======
        # 1. 将注意力历史变形以适应卷积
        attns_bt_chw = rearrange(attns_bntl, "b n t (h w) -> (b t) n h w", h=h, w=w)
        
        # 2. 通过门控网络计算“遗忘系数”
        # gate_values 的形状会是 [(B*T), in_chs, 1, 1]
        gate_values = self.gating_network(attns_bt_chw)
        
        # 3. 将门控值 reshape 并应用到原始注意力历史中
        # gate_values_bntl 的形状是 [B, in_chs/nhead * n, T, 1]
        # gate_values_bntl = rearrange(gate_values, "(b t) n h w -> b n t (h w)", b=b, t=t)
        gate_values_bntl = rearrange(gate_values, "(b t) n 1 1 -> b n t 1", t=t)
        
        # 逐元素相乘，实现智能过滤
        # 结构符号相关的注意力图，其门控值会被学习到接近0
        # 实体符号相关的注意力图，其门控值会被学习到接近1
        gated_attns_bntl = attns_bntl * gate_values_bntl
        # =================================

        # 使用被门控过的注意力历史来计算覆盖惩罚
        cumulative_gated_attns = gated_attns_bntl.cumsum(dim=2) - gated_attns_bntl
        cumulative_gated_attns_2d = rearrange(cumulative_gated_attns, "b n t (h w) -> (b t) n h w", h=h)

        # 后续的覆盖计算流程完全不变
        mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t)
        cov = self.conv(cumulative_gated_attns_2d)
        cov = self.act(cov)
        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)
        cov = self.post_norm(cov, mask)
        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
        
        # 返回最终的、被智能过滤过的惩罚项
        return cov
        
        # # tgt_vocab_l=tgt_vocab[:b,:]
        # # mask_vocab_l=torch.logical_not(torch.logical_or(tgt_vocab_l == 110, torch.logical_or(tgt_vocab_l == 82, tgt_vocab_l == 83)))
        # # tgt_vocab_r=tgt_vocab[b:,:]
        # # mask_vocab_r=torch.logical_not(torch.logical_or(tgt_vocab_r == 112, torch.logical_or(tgt_vocab_r == 82, tgt_vocab_r == 83)))
        # # mask_vocab=torch.cat((mask_vocab_l, mask_vocab_r), dim=0)
        # # mask_vocab=mask_vocab.unsqueeze(1).repeat(1, 2*self.nhead, 1)
        # tgt_vocab=tgt_vocab.unsqueeze(1).repeat(1, 2*self.nhead, 1)
        # # mask_vocab = torch.logical_not(torch.logical_or(torch.logical_or(tgt_vocab == 110, tgt_vocab == 112), torch.logical_or(tgt_vocab == 82, tgt_vocab == 83)))
        # mask_vocab = torch.logical_not(torch.logical_or(tgt_vocab == 110, torch.logical_or(tgt_vocab == 82, tgt_vocab == 83)))
        # # mask_vocab = torch.logical_not(torch.logical_or(torch.logical_or(tgt_vocab == 110, tgt_vocab == 53), torch.logical_or(tgt_vocab == 82, tgt_vocab == 83)))
        # attns = attns*mask_vocab.unsqueeze(-1).float()
        # attns = attns.cumsum(dim=2) - attns
        # attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)

        # cov = self.conv(attns)
        # cov = self.act(cov)

        # cov = cov.masked_fill(mask, 0.0)
        # cov = self.proj(cov)

        # cov = self.post_norm(cov, mask)

        # cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
        # return cov


