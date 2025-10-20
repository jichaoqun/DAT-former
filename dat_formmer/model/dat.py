from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from dat_formmer.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder
# from .decoder_str import DecoderStr
# from .my import mit_b0, encoder
# from .resnet import encoderResnet
from einops.einops import rearrange

class DAT(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        encoder_nhead: int,
        encoder_num_layers: int,
        encoder_dim_feedforward: int,
        encoder_dropout: float,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers, encoder_nhead=encoder_nhead,
            encoder_num_layers=encoder_num_layers, encoder_dim_feedforward=encoder_dim_feedforward,
            encoder_dropout=encoder_dropout,
        )
        # self.encoder_new = encoderResnet()
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        # self.decoder_str = DecoderStr(
        #     d_model=d_model,
        #     nhead=nhead,
        #     num_decoder_layers=2,
        #     dim_feedforward=512,
        #     dropout=dropout,
        #     dc=dc,
        #     cross_coverage=cross_coverage,
        #     self_coverage=self_coverage,
        # )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, structure: LongTensor = None
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        # print("img+++++", img.shape, img_mask.shape)

        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        # print("feature shape ++++", feature.shape, mask.shape)
        # feature, mask = self.encoder_new(img, img_mask)
        # feature = rearrange(feature, "b d h w -> b h w d")
        # print("feature shape1111111 ++++", feature.shape, mask.shape)
        feature_ = torch.cat((feature, feature), dim=1)  # [2b, t, d]
        mask_ = torch.cat((mask, mask), dim=0)
        # print("feature shape2222222 ++++", feature_.shape, mask_.shape)

        out = self.decoder(feature_, mask_, tgt)

        # structure_out = self.decoder_str(feature_, mask_, tgt, structure)
        # return out, structure_out

        return out

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        # print("feature shape ++++", feature.shape, mask.shape)
        # feature, mask = self.encoder_new(img, img_mask)
        # feature = rearrange(feature, "b d h w -> b h w d")
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )
