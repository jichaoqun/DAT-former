from typing import List

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor
import copy
from typing import Optional, List
from torch import Tensor
import torch.nn.functional as F

from dat_formmer.datamodule import vocab, vocab_size
from dat_formmer.model.pos_enc import WordPosEnc, ImgPosEnc
from dat_formmer.model.transformer.arm import AttentionRefinementModule
from dat_formmer.model.transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from dat_formmer.utils.generation_utils import DecodeModel


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
) -> nn.TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)
    else:
        arm = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder


class Decoder(DecodeModel):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        # self.pos_enc = WordPosEnc(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            # setp = self.global_step,
        )

        self.proj = nn.Linear(d_model, vocab_size)


    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        # print(src.shape, src_mask.shape, tgt.shape)
        # print(self.device)
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX
        tgt_vocab=tgt

        tgt = self.word_embed(tgt)  # [b, l, d]
        # tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)

        h, w = src_mask.shape[1], src_mask.shape[2]
        
        # src_pos = self.pos_enc_2d(src, src_mask)
        # src = rearrange(src, "b h w d -> (h w) b d")
        # src_pos = rearrange(src_pos, "b h w d -> (h w) b d")
        # src_mask = rearrange(src_mask, "b h w -> b (h w)")

        # src = self.encoder(src=src, src_key_padding_mask=src_mask, pos=src_pos)


        # src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        # print("src_mask shape:", src_mask.shape, "src shape:", src.shape)

        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            width=w,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            tgt_vocab=tgt_vocab,
            setp=self.global_step,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        return out

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        word_out = self(src[0], src_mask[0], input_ids)
        return word_out



