from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger

import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] ="0"


from dat_formmer.datamodule import CROHMEDatamodule
from dat_formmer.lit_dat import LitDat

cli = LightningCLI(
    LitDat,
    CROHMEDatamodule,
)
