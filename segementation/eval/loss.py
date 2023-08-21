import torch


def dice_coef(y_true, y_pred, eps=1e-5):
    # pred = torch.flatten(pred)
    # mask = torch.flatten(mask)

    # counter = (pred * mask).sum()  # Counter
    # denum = pred.sum() + mask.sum()  # denominator
    # dice = (2 * counter) / denum

    # return dice
    y_pred = torch.sigmoid(y_pred)
    y_pred_f = torch.flatten(y_pred)
    y_true_f = torch.flatten(y_true)
    intersection = y_true_f * y_pred_f
    score = (
        2.0
        * (torch.sum(intersection) + eps)
        / (torch.sum(y_true_f) + torch.sum(y_pred_f) + eps)
    )
    return score


def iou_score(y_true, y_pred, smooth=1e-5):
    y_pred = torch.sigmoid(y_pred)
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def dice_loss(y_true, y_pred, smooth=1):
    y_pred = torch.sigmoid(y_pred)
    y_pred_f = torch.flatten(y_pred)
    y_true_f = torch.flatten(y_true)
    intersection = y_true_f * y_pred_f
    score = (2.0 * torch.sum(intersection) + smooth) / (
        torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth
    )
    return 1.0 - score


class DiceScore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, mask):
        return dice_coef(mask, pred)


class DiceLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, mask):
        return dice_loss(mask, pred)


class BCEWithLogitAndDice(torch.nn.Module):
    def __init__(self, logDice=False, reduction="mean") -> None:
        super().__init__()
        self.loss_BCE = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.loss_Dice = DiceLoss()
        self.log = logDice

    def forward(self, pred, mask):
        if self.log:
            return self.loss_BCE(pred, mask) - torch.log(dice_coef(pred, mask))
        return self.loss_BCE(pred, mask) + self.loss_Dice(pred, mask)
