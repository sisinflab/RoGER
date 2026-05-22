import torch
from abc import ABC

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ContrastLoss(nn.Module, ABC):
    """
    Contrastive Loss for comparing positive and negative embeddings.
    """
    def __init__(self, feat_size, tau):
        super(ContrastLoss, self).__init__()
        # Weight matrix for bilinear similarity
        self.w = nn.Parameter(torch.Tensor(feat_size, feat_size))
        init.xavier_uniform_(self.w.data)
        # Binary Cross-Entropy Loss with Logits
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean') # Usiamo 'mean' per ottenere una loss scalare
        self.tau = tau

    def forward(self, x, y, y_neg=None):
        """
        Args:
            x (Tensor): Batch of embeddings (bs x dim).
            y (Tensor): Batch of positive embeddings (bs x dim).
            y_neg (Tensor, optional): Batch of negative embeddings (bs x dim).
        Returns:
            Tensor: Scalar contrastive loss.
        """
        # Positive pairs
        scores = (x @ self.w * y ).sum(1)
        scores = scores / self.tau
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        # Negative pairs
        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]
        neg2_scores = (x @ self.w * y_neg).sum(1)
        neg2_scores = neg2_scores / self.tau
        neg2_labels = neg2_scores.new_zeros(neg2_scores.shape)
        neg2_loss = self.bce_loss(neg2_scores, neg2_labels)

        loss = pos_loss + neg2_loss
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, tau):
        super(InfoNCELoss, self).__init__()
        self.tau = tau

    def forward(self, view1, view2):
        """
        view1, view2: Tensori (batch_size x dim) già normalizzati.
        """
        # Calcola la similarità coseno tra tutti gli elementi nel batch
        # Risultato: matrice (batch_size x batch_size)
        sim_matrix = torch.matmul(view1, view2.T) / self.tau

        # I positivi si trovano sulla diagonale della matrice
        labels = torch.arange(view1.size(0)).long().to(view1.device)

        # CrossEntropyLoss applica implicitamente il log-softmax
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class RatingSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss adattata per la Rating Prediction.
    Usa il valore del rating per determinare le coppie positive reali.
    """

    def __init__(self, tau=0.1, threshold=4.0):
        super(RatingSupConLoss, self).__init__()
        self.tau = tau
        self.threshold = threshold

    def forward(self, user_emb, item_emb, ratings):
        """
        Args:
            user_emb: Tensore (batch_size x dim)
            item_emb: Tensore (batch_size x dim)
            ratings: Tensore (batch_size) con i veri rating (es. da 1 a 5)
        """
        # 1. L2 Normalization (fondamentale per la stabilità di InfoNCE/SupCon)
        user_emb = F.normalize(user_emb, dim=1)
        item_emb = F.normalize(item_emb, dim=1)

        # 2. Matrice di similarità Tutti-contro-Tutti (batch_size x batch_size)
        sim_matrix = torch.matmul(user_emb, item_emb.T) / self.tau

        # 3. Log-Softmax sulle colonne (per ogni utente, probabilità su tutti gli item del batch)
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # 4. Estraiamo i valori sulla diagonale (che corrispondono alle interazioni reali Utente-Item del batch)
        diag_log_prob = torch.diag(log_prob)

        # 5. La Supervisione: creiamo una maschera binaria basata sul rating
        # 1.0 se l'utente ha gradito l'item, 0.0 se lo ha detestato (o neutro)
        pos_mask = (ratings >= self.threshold).float().to(user_emb.device)

        # 6. Calcolo della loss
        # Massimizziamo la probabilità SOLO per le coppie utente-item con rating alto.
        # Gli item con rating basso (mask=0) non contribuiscono ad avvicinare i vettori,
        # ma agiranno comunque come negativi (denominatore del softmax) per gli altri!
        valid_positives = pos_mask.sum() + 1e-8  # Evita divisioni per zero
        loss = - (diag_log_prob * pos_mask).sum() / valid_positives

        return loss