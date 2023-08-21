# Full Segmentation Model
import torch
from model.baseline_unet import UNet
from model.attention_unet import A_UNET
import pytorch_lightning as pl
from lightning import LightningModule
import numpy as np
import matplotlib.pyplot as plt
from eval.loss import iou_score, dice_coef as dice_score

LEARNING_RATE = 1e-2
# WEIGHT_DECAY = 0.95


class LightningSegmentation(LightningModule):
    def __init__(self, model, optimizer, loss_fn, learning_rate=1e-4):
        super().__init__()
        print("new change")

        self.model = model.to("cuda")
        self.model.train()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.save_hyperparameters()

    def forward(self, data):
        logits = self.model(data.cuda())
        return logits

    def training_step(self, batch, batch_idx):
        ct, mask = batch
        mask = mask.cuda()

        logits = self(ct)
        loss = self.loss_fn(logits, mask)
        # Logs
        self.log("Train loss", loss)
        dice = dice_score(mask, logits)
        iou = iou_score(mask, logits)
        self.log("Train Dice", dice)
        self.log("Train IOU", iou)

        if batch_idx % 50 == 0:
            self.log_images(ct.cpu(), logits.cpu(), mask.cpu(), "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        ct, mask = batch
        mask = mask.cuda()

        logits = self(ct)
        loss = self.loss_fn(logits, mask)

        # Logs
        self.log("Val loss", loss)
        dice = dice_score(mask, logits)
        iou = iou_score(mask, logits)
        self.log("Val Dice", dice)
        self.log("Val IOU", iou)
        if batch_idx % 50 == 0:
            self.log_images(ct.cpu(), logits.cpu(), mask.cpu(), "Val")

        return loss

    def log_images(self, ct, pred, mask, name):
        results = []

        pred = (
            pred > 0.5
        )  # As we use the sigomid activation function, we threshold at 0.5
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(ct[0][0], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][0] == 0, mask[0][0])
        axis[0].imshow(mask_, alpha=0.6)
        axis[0].set_title("Ground Truth")

        axis[1].imshow(ct[0][0], cmap="bone")
        mask_ = np.ma.masked_where(pred[0][0] == 0, pred[0][0])
        axis[1].imshow(mask_, alpha=0.6, cmap="autumn")
        axis[1].set_title("Pred")

        self.logger.experiment.add_figure(
            f"{name} Prediction vs Label", fig, self.global_step
        )

    def configure_optimizers(self):
        if self.optimizer == None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # We always need to return a list here (just pack our optimizer into one :))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=4, verbose=True, factor=0.5
        )
        # return [self.optimizer]
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": scheduler,  # Changed scheduler to lr_scheduler
            "monitor": "Val Dice",
        }
