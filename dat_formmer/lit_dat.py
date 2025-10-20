import math
import zipfile
from typing import List

# import pytorch_lightning as pl
import lightning as L
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from dat_formmer.datamodule import Batch, vocab
from dat_formmer.model.dat import DAT
# from comer.utils.utils import str_loss
from dat_formmer.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss, 
                               structure_loss,
                               to_bi_tgt_out, to_bi_cursor_pos, build_bi_cursor_pos_input_target)
from torch.optim.lr_scheduler import LambdaLR

class LitDat(L.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        learning_rate: float,
        patience: int,
        warmup_steps: int,
        max_epochs: int,
        encoder_type: str,
        # encoder transformer
        encoder_nhead: int = 8,
        encoder_num_layers: int = 3,
        encoder_dim_feedforward: int = 1024,
        encoder_dropout: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dat_model = DAT(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            encoder_nhead=encoder_nhead,
            encoder_num_layers=encoder_num_layers,
            encoder_dim_feedforward=encoder_dim_feedforward,
            encoder_dropout=encoder_dropout,
        )

        self.exprate_recorder = ExpRateRecorder()


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
        return self.dat_model(img, img_mask, tgt, structure)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)


        out_hat = self(batch.imgs, batch.mask, tgt, None)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.mask.size(0))

        return loss
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)

        out_hat = self(batch.imgs, batch.mask, tgt, None)


        loss = ce_loss(out_hat, out)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.mask.size(0),
        )

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.imgs.size(0),
        )
    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps]

    def on_test_epoch_end(self, pl_module) -> None:
        exprate = self.exprate_recorder.compute()
        # print(f"Validation ExpRate: {exprate}")
        test_outputs = torch.stack(pl_module.test_step_outputs).mean()
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)
        torch.cuda.empty_cache()
    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.dat_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )
        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
            # min_lr=1e-6,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
