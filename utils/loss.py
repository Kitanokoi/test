import torch
import torch.nn as nn
import torch.nn.functional as F

class UnsupervisedLoss(nn.Module):
    def __init__(self, a=0.5, w=1.0):
        super(UnsupervisedLoss, self).__init__()
        self.a = a

    def forward(self, Y_raw, Y_u):
        # L1 loss between Y_raw and Y_s
        L1_loss = F.l1_loss(Y_raw, Y_u)

        # L2 loss between Y_raw and Y_s
        L2_loss = F.mse_loss(Y_raw, Y_u)

        # Total loss as weighted sum of L1 and L2
        Ltotal = self.a * L1_loss + (1 - self.a) * L2_loss

        # # Cross-entropy loss between Y_s and Y
        # Lce = F.cross_entropy(Y_s, Y)

        # # Final loss
        # L = Lce + self.w * Ltotal

        return Ltotal

# Usage example:
# criterion = CustomLoss(a=0.5, w=1.0)
# loss = criterion(Y_raw, Y_u, Y_s, Y)