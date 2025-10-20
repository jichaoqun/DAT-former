import math
from typing import Tuple

# import pytorch_lightning as pl
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor
import copy
from typing import Optional, List
from torch import Tensor
from typing import Tuple


from .pos_enc import ImgPosEnc


# 通道注意力机制Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP
        self.fc_in = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.act1 = nn.ReLU()
        self.fc_out = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sigmoid(self.fc_out(self.act1(self.fc_in(self.avg_pool(x)))))
        max_out = self.sigmoid(self.fc_out(self.act1(self.fc_in(self.max_pool(x)))))

        out = avg_out + max_out

        return out


# 空间注意力机制Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 核大小必须为3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.padding = 3 if kernel_size == 7 else 1

        self.cov1 = nn.Conv2d(2, 1, kernel_size, padding=self.padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # concat
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.cov1(x)
        return self.sigmoid(x)


class CMBABlock(nn.Module):
    def __init__(self, in_channel, ratio=16, kernel_size=7):
        super(CMBABlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channel=in_channel, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out + res


# DenseNet-B
# RestNet中提出的bottleneck块
class _Bottleneck(nn.Module):

    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool, use_vision_attention: bool = False):
        super(_Bottleneck, self).__init__()
        self.use_vision_attention = use_vision_attention
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels,
                               interChannels,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        if self.use_vision_attention:
            self.vision_attention1 = CMBABlock(interChannels)
        self.conv2 = nn.Conv2d(interChannels,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        if self.use_vision_attention:
            self.vision_attention2 = CMBABlock(growth_rate)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 1*1卷积增加通道数 -> BN -> ReLu -> 3*3卷积提取特征下采样 -> bn -> ReLu -> 残差连接
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_vision_attention:
            out = self.vision_attention1(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_vision_attention:
            out = self.vision_attention2(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class _SingleLayer(nn.Module):

    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool, use_vision_attention: bool = False):
        super(_SingleLayer, self).__init__()
        self.use_vision_attention = use_vision_attention
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        if use_vision_attention:
            self.vision_attention = CMBABlock(growth_rate)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # ReLu -> 3*3卷积下采样 -> 残差连接
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_vision_attention:
            out = self.vision_attention(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class _Transition(nn.Module):

    def __init__(self, n_channels: int, n_out_channels: int,
                 use_dropout: bool, use_vision_attention: bool = False, down: bool =False):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.use_vision_attention = use_vision_attention
        self.conv1 = nn.Conv2d(n_channels,
                               n_out_channels,
                               kernel_size=1,
                               bias=False)
        if use_vision_attention:
            self.vision_attention = CMBABlock(n_out_channels)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

        self.down = down  # 是否下采样

    def forward(self, x):
        # 1*1卷积改变通道数 -> BN -> ReLu -> 2*2 avg_pooling
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_vision_attention:
            out = self.vision_attention(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.down:
            out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out
    
def downsample_mask(mask_bool, k=2):
    m = F.avg_pool2d(mask_bool.float(), kernel_size=k, ceil_mode=True)
    return (m > 0.5)   # True=pad（多数为 pad 才判定为 pad）


class DenseNet(nn.Module):

    def __init__(
            self,
            growth_rate: int,
            num_layers: int,
            reduction: float = 0.5,
            bottleneck: bool = True,
            use_dropout: bool = True,
            use_vision_attention: bool = False,
    ):
        super(DenseNet, self).__init__()
        # dense块数
        n_dense_blocks = num_layers
        # 输出通道数
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(1,
                               n_channels,
                               kernel_size=7,
                               padding=3,
                               stride=2,
                               bias=False)
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout, use_vision_attention)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout, use_vision_attention, down=True)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout, use_vision_attention)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout, use_vision_attention, down=True)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout, use_vision_attention)

        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck,
                    use_dropout, use_vision_attention):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate,
                                          use_dropout, use_vision_attention))
            else:
                layers.append(
                    _SingleLayer(n_channels, growth_rate, use_dropout, use_vision_attention))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out = self.norm1(out)
        # out_mask = x_mask[:, 0::2, 0::2]
        # out_mask = F.max_pool2d(x_mask.float(), kernel_size=2, ceil_mode=True).bool()
        out_mask = downsample_mask(x_mask, k=2)

        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        # print("Before Dense1 feature shape: ", out.shape)
        # out_mask = out_mask[:, 0::2, 0::2]
        # out_mask = F.max_pool2d(out_mask.float(), kernel_size=2, ceil_mode=True).bool()
        out_mask = downsample_mask(out_mask, k=2)


        out = self.dense1(out)
        out = self.trans1(out)
        # out_mask = out_mask[:, 0::2, 0::2]
        # out_mask = F.max_pool2d(out_mask.float(), kernel_size=2, ceil_mode=True).bool()
        out_mask = downsample_mask(out_mask, k=2)

        # print("Before Dense2 feature shape: ", out.shape)
        out = self.dense2(out)
        out = self.trans2(out)
        # print("Before Dense3 feature shape: ", out.shape)
        # out_mask = x_mask[:, 0::16, 0::16]
        # out_mask = F.max_pool2d(out_mask.float(), kernel_size=2, ceil_mode=True).bool()
        out_mask = downsample_mask(out_mask, k=2)

        out = self.dense3(out)
        out = self.post_norm(out)
        # print("After Dense3 feature shape: ", out.shape)
        return out, out_mask

class Rotary2D(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, H, W):
        device = self.inv_freq.device
        h_pos = torch.arange(H, device=device)
        w_pos = torch.arange(W, device=device)

        sin_h = torch.einsum("i,j->ij", h_pos, self.inv_freq).sin()
        cos_h = torch.einsum("i,j->ij", h_pos, self.inv_freq).cos()
        sin_w = torch.einsum("i,j->ij", w_pos, self.inv_freq).sin()
        cos_w = torch.einsum("i,j->ij", w_pos, self.inv_freq).cos()

        # Expand to [H, W, dim]
        sin_h = sin_h[:, None, :].expand(H, W, -1)
        cos_h = cos_h[:, None, :].expand(H, W, -1)
        sin_w = sin_w[None, :, :].expand(H, W, -1)
        cos_w = cos_w[None, :, :].expand(H, W, -1)
        return cos_h, sin_h, cos_w, sin_w

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


# 新的 apply_2d_rope 函数
def apply_2d_rope(x, cos_h, sin_h, cos_w, sin_w):
    """
    Applies 2D RoPE by splitting the channel dimension.
    
    Args:
        x (Tensor): Input tensor of shape [B, C, H, W]
        cos_h, sin_h, cos_w, sin_w (Tensor): Tensors of shape [H, W, C//4]
    """
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

    # 将通道 C 分成两半，分别给 h 和 w
    x_h, x_w = x.chunk(2, dim=-1) # x_h, x_w shape: [B, H, W, C//2]
    
    # 对 h 部分应用 RoPE
    # cos_h, sin_h 需要 unsqueeze 来广播
    cos_h = cos_h.unsqueeze(0) # [1, H, W, C//4]
    sin_h = sin_h.unsqueeze(0) # [1, H, W, C//4]
    
    # 扩展 cos_h/sin_h 的维度以匹配 x_h 的一半 (因为 rotate_half)
    cos_h = cos_h.repeat(1, 1, 1, 2) # [1, H, W, C//2]
    sin_h = sin_h.repeat(1, 1, 1, 2) # [1, H, W, C//2]

    x_h_rotated = (x_h * cos_h) + (rotate_half(x_h) * sin_h)

    # 对 w 部分应用 RoPE
    cos_w = cos_w.unsqueeze(0)
    sin_w = sin_w.unsqueeze(0)
    cos_w = cos_w.repeat(1, 1, 1, 2)
    sin_w = sin_w.repeat(1, 1, 1, 2)
    
    x_w_rotated = (x_w * cos_w) + (rotate_half(x_w) * sin_w)

    # 拼接回来
    x_rotated = torch.cat([x_h_rotated, x_w_rotated], dim=-1) # [B, H, W, C]
    
    # 转换回序列格式
    return x_rotated.reshape(B, H * W, C)


    

class VisionEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.position_encoder = Rotary2D(d_model)
        # encoder_layer = TransformerBlock(d_model, nhead, dim_feedforward, dropout)
        # self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )

    def forward(self, img_feat: torch.Tensor, mask: torch.Tensor):
        B, C, H, W = img_feat.shape
        x = img_feat.permute(0, 2, 3, 1)  # [B, H, W, C]
        attn_mask = mask.reshape(B, -1)  # [B, H*W], 0=padded

        # 位置编码
        cos_h, sin_h, cos_w, sin_w = self.position_encoder(H, W)

        # 多层 Transformer 编码器
        for layer in self.layers:
            x = layer(x, cos_h, sin_h, cos_w, sin_w, attn_mask)

        x = x.reshape(B, H * W, C)
        return x, attn_mask
    
    
def apply_2d_rope_to_qk(x, cos_h, sin_h, cos_w, sin_w):
    """
    x: Tensor of shape [B, H, W, C]
    cos/sin: [H, W, C//2] → expand to match x
    """
    B, H, W, C = x.shape
    x1, x2 = x[..., :C//2], x[..., C//2:]
    
    # 扩展 cos/sin 维度
    cos_h = cos_h.unsqueeze(0).repeat(B, 1, 1, 2)
    sin_h = sin_h.unsqueeze(0).repeat(B, 1, 1, 2)
    cos_w = cos_w.unsqueeze(0).repeat(B, 1, 1, 2)
    sin_w = sin_w.unsqueeze(0).repeat(B, 1, 1, 2)

    # 分别对两个方向旋转
    x1 = x1 * cos_h + rotate_half(x1) * sin_h
    x2 = x2 * cos_w + rotate_half(x2) * sin_w
    return torch.cat([x1, x2], dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cos_h, sin_h, cos_w, sin_w, attn_mask):
        B, H, W, C = x.shape
        x_flat = x.reshape(B, H * W, C)  # for residual

        # Q K V projection
        q = self.q_proj(x).reshape(B, H, W, C)
        k = self.k_proj(x).reshape(B, H, W, C)
        v = self.v_proj(x).reshape(B, H * W, C)

        # Apply RoPE to Q and K
        q = apply_2d_rope_to_qk(q, cos_h, sin_h, cos_w, sin_w)
        k = apply_2d_rope_to_qk(k, cos_h, sin_h, cos_w, sin_w)

        q = q.reshape(B, H * W, C)
        k = k.reshape(B, H * W, C)

        # Attention
        x_attn, _ = self.attn(q, k, v, key_padding_mask=attn_mask.bool())

        # Add & Norm
        x = self.norm1(x_flat + x_attn)
        x = self.norm2(x + self.mlp(x))
        return x.reshape(B, H, W, C)


class Encoder(L.LightningModule):
    def __init__(self, d_model: int, growth_rate: int, num_layers: int,
                  encoder_nhead : int = 8,
                  encoder_num_layers: int = 3, 
                  encoder_dim_feedforward: int = 1024,
                  encoder_dropout: float = 0.3,):
        super().__init__()

        self.model = DenseNet(growth_rate=growth_rate, num_layers=num_layers)
        self.feature_proj = nn.Conv2d(self.model.out_channels, d_model, kernel_size=1)

        self.versionEncoder = VisionEncoder(d_model=d_model,
                                            nhead=encoder_nhead, 
                                            num_layers=encoder_num_layers, 
                                            dim_feedforward=encoder_dim_feedforward, 
                                            dropout=encoder_dropout)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        # print("Encoder input img shape: ", img.shape, " img_mask shape: ", img_mask.shape)
        # extract feature
        feature, mask = self.model(img, img_mask)
        # print("Encoder after DenseNet feature shape: ", feature.shape, " mask shape: ", mask.shape)
        feature = self.feature_proj(feature)
        # print("Encoder after DenseNet feature shape: ", feature.shape, " mask shape: ", mask.shape)
        # feature = rearrange(feature, "b c h w -> b (h w) c")

        feature, _ = self.versionEncoder(feature, mask)

        feature = rearrange(feature, "b l d -> l b d")

        feature = self.norm(feature)

        return feature, mask
