import torch
from abc import ABC

import torch.nn as nn
import torch.nn.init as init

class ContrastLoss(nn.Module, ABC):
    """
    Contrastive Loss per confrontare embedding positivi e negativi.
    """
    def __init__(self, feat_size):
        super(ContrastLoss, self).__init__()
        # Matrice di pesi per la similarità bilineare
        self.w = nn.Parameter(torch.Tensor(feat_size, feat_size))
        init.xavier_uniform_(self.w.data)
        # Loss Binary Cross-Entropy con Logits
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean') # Usiamo 'mean' per ottenere una loss scalare

    def forward(self, x, y, y_neg=None):
        """
        :param x: bs * dim
        :param y: bs * dim
        :param y_neg: bs * dim
        :return:
        """

        # positive
        #  scores = self.bilinear(x, y).squeeze()
        scores = (x @ self.w * y ).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        #  neg2_scores = self.bilinear(x, y_neg).squeeze()
        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]
        neg2_scores = (x @ self.w * y_neg).sum(1)
        neg2_labels = neg2_scores.new_zeros(neg2_scores.shape)
        neg2_loss = self.bce_loss(neg2_scores, neg2_labels)

        loss = pos_loss + neg2_loss
        return loss