import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Hybrid loss = α * BCEWithLogits + (1-α) * (1 - Dice)
    """
    def __init__(self, alpha=0.5, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, targets):
        # bce part
        bce_loss = self.bce(logits, targets)

        # dice part (on probabilities)
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (1, 2, 3)  # assuming [B,1,H,W]
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss
