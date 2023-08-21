# Full Segmentation Model
import torch
from model.baseline.model import UNet
from model.baseline.attention_unet import A_UNET
import torchmetrics
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-2
# WEIGHT_DECAY = 0.95


def dice_score(pred, mask):
    # flatten label and prediction tensors
    pred = torch.flatten(pred)
    mask = torch.flatten(mask)

    counter = (pred * mask).sum()  # Counter
    denum = pred.sum() + mask.sum()  # denominator
    dice = (2 * counter) / denum

    return dice


class TumorSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        print("Intializing change")

        # self.model = UNet()
        self.model = A_UNET()
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            amsgrad=True,
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.train_iou = torchmetrics.JaccardIndex("binary")
        # self.validation_iou = torchmetrics.JaccardIndex("binary")

    def forward(self, data):
        logits = self.model(data)
        pred = torch.argmax(logits, dim=1)
        return pred, logits

    def training_step(self, batch, batch_idx):
        ct, mask = batch
        mask = mask.float()
        labels = mask.squeeze(1).float()

        pred, logits = self(ct)
        loss = self.loss_fn(logits, mask)
        # Logs
        self.log("Train loss", loss)
        dice = dice_score(torch.sigmoid(logits), mask)
        self.log("Train Dice", dice)

        if batch_idx % 50 == 0:
            self.log_images(ct.cpu(), logits.cpu(), mask.cpu(), "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        ct, mask = batch
        mask = mask.float()
        labels = mask.squeeze(1).float()

        pred, logits = self(ct)
        loss = self.loss_fn(logits, mask)

        # Logs
        self.log("Val loss", loss)
        dice = dice_score(torch.sigmoid(logits), mask)
        self.log("Val Dice", dice)
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
        # We always need to return a list here (just pack our optimizer into one :))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=1, verbose=True
        )
        # return [self.optimizer]
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": scheduler,  # Changed scheduler to lr_scheduler
            "monitor": "Val Dice",
        }
