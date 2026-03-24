from abc import ABC

from torch_geometric.nn import GCNConv, GATConv
from collections import OrderedDict
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .ContrastLoss import ContrastLoss

import torch
import torch_geometric
import numpy as np
import random

from torch_sparse import SparseTensor


class RoGERModel(torch.nn.Module, ABC):
    """
    RoGERModel implements the core graph neural network for the RoGER recommender.
    It supports GCN, GAT, and dense aggregation, and includes contrastive loss.
    """

    def __init__(
        self,
        num_users,
        num_items,
        learning_rate,
        embed_k,
        n_layers,
        edge_features,
        edge_index,
        lm,
        aggr,
        drop,
        dense,
        random_seed,
        alpha,
        factor,
        patience,
        weight_decay,
        save_adj,
        tau,
        threshold,
        name="RoGER",
        **kwargs
    ):
        super().__init__()

        # set seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model hyperparameters
        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.weight_decay = weight_decay
        self.save_adj = save_adj
        self.tau = tau
        self.threshold = threshold

        self.L0 = self.compute_normalized_edge_weights(
            edge_index, self.num_users + self.num_items
        )
        self.edge_index = edge_index.to(dtype=torch.long).to(self.device)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty((self.num_users, self.embed_k)))
        )
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty((self.num_items, self.embed_k)))
        )
        self.Gi.to(self.device)

        self.Bu = torch.nn.Embedding(self.num_users, 1)
        torch.nn.init.xavier_normal_(self.Bu.weight)
        self.Bu.to(self.device)
        self.Bi = torch.nn.Embedding(self.num_items, 1)
        torch.nn.init.xavier_normal_(self.Bi.weight)
        self.Bi.to(self.device)

        self.Mu = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty((1, 1))))
        self.Mu.to(self.device)

        self.lm = lm
        self.aggr = aggr
        self.drop = drop
        self.alpha = alpha
        self.factor = factor
        self.patience = patience

        self.edge_embeddings_interactions = torch.tensor(
            edge_features, dtype=torch.float32, device=self.device
        ).squeeze()

        self.feature_dim = edge_features.shape[2]

        # Build GCN layers for message passing
        propagation_node_node_textual_list = []
        for _ in range(self.n_layers):
            propagation_node_node_textual_list.append(
                (
                    GCNConv(
                        in_channels=self.embed_k,
                        out_channels=self.embed_k,
                        normalize=True,
                        add_self_loops=False,
                        bias=True,
                    ),
                    "x, edge_index, edge_weight -> x",
                )
            )

        self.node_node_textual_network = torch_geometric.nn.Sequential(
            "x, edge_index, edge_weight", propagation_node_node_textual_list
        )
        self.node_node_textual_network.to(self.device)

        # Aggregation type: 'sim', 'nn', or 'att'
        if self.aggr == "sim":
            # projection
            self.projection = torch.nn.Linear(
                self.edge_embeddings_interactions.shape[-1], self.embed_k
            )
            self.projection.to(self.device)

        elif self.aggr == "nn":
            self.dense_layer_size = [self.embed_k * 2 + self.feature_dim] + dense
            self.num_dense_layers = len(self.dense_layer_size)
            dense_network_list = []
            for idx, _ in enumerate(self.dense_layer_size[:-1]):
                dense_network_list.append(
                    (
                        "dense_" + str(idx),
                        torch.nn.Linear(
                            in_features=self.dense_layer_size[idx],
                            out_features=self.dense_layer_size[idx + 1],
                            bias=True,
                        ),
                    )
                )
                dense_network_list.append(
                    ("drop_" + str(idx), torch.nn.Dropout(p=self.drop))
                )
                dense_network_list.append(("relu_" + str(idx), torch.nn.ReLU()))
            dense_network_list.append(
                (
                    "out",
                    torch.nn.Linear(
                        in_features=self.dense_layer_size[-1], out_features=1, bias=True
                    ),
                )
            )
            dense_network_list.append(("sigmoid", torch.nn.Sigmoid()))
            self.dense_network = torch.nn.Sequential(OrderedDict(dense_network_list))
            self.dense_network.to(self.device)
        else:
            self.attention = GATConv(
                in_channels=self.embed_k,
                out_channels=self.embed_k,
                concat=True,
                edge_dim=self.feature_dim,
                add_self_loops=False,
                bias=True,
            )
            self.attention.to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.factor, patience=self.patience
        )

        # Loss functions
        self.mse_loss = torch.nn.MSELoss()
        self.contrast_loss = ContrastLoss(feat_size=self.embed_k, tau=self.tau).to(self.device)

    def compute_normalized_edge_weights(self, edge_index, num_nodes):
        values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=self.device)
        edge_index = edge_index.to(self.device)
        row = edge_index[0].to(dtype=torch.long, device=self.device)
        col = edge_index[1].to(dtype=torch.long, device=self.device)
        adj = SparseTensor(row=row, col=col, value=values, sparse_sizes=(num_nodes, num_nodes))
        deg = adj.sum(dim=1).to(torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        row, col = edge_index[0], edge_index[1]
        norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
        return norm_values

    def row_normalize(self, A, edge_index, num_nodes):
        """
        Normalizza i pesi degli archi per riga (nodo sorgente).
        A: vettore dei pesi degli archi (shape: [num_edges])
        edge_index: shape [2, num_edges]
        num_nodes: numero totale di nodi
        """
        row = edge_index[0].to(dtype=torch.long, device=self.device)
        # Calcola la somma dei pesi per ogni nodo sorgente
        row_sum = torch.zeros(num_nodes, device=self.device).scatter_add_(0, row, A)
        # Evita divisione per zero
        norm = A / (row_sum[row] + 1e-8)
        return norm

    def soft_threshold(self, x, k=5):
        return torch.sigmoid(k * (x - self.threshold))

    def propagate_embeddings(self, mask_user=None, mask_item=None, evaluate=False):
        """
        Propagate node embeddings through GCN layers.
        If evaluate=True, disables dropout and uses all edges.
        """
        all_embeddings = torch.cat(
            (self.Gu.to(self.device), self.Gi.to(self.device)), 0
        )

        current_edge_index = getattr(self, "current_edge_index", self.edge_index[:2].long())
        current_edge_weight = getattr(self, "current_edge_weights", None)

        for layer in range(self.n_layers):
            if evaluate:
                with torch.no_grad():
                    all_embeddings = torch.relu(
                        list(self.node_node_textual_network.children())[layer](
                            all_embeddings.to(self.device),
                            edge_index=current_edge_index,
                            edge_weight=current_edge_weight
                        )
                    )
            else:
                masked_edge_weight = current_edge_weight[torch.cat([mask_user, mask_item]) & torch.cat([mask_item, mask_user])] if current_edge_weight is not None else None
                all_embeddings = torch.relu(
                    list(self.node_node_textual_network.children())[layer](
                        torch.dropout(
                            all_embeddings.to(self.device),
                            p=self.drop,
                            train=not evaluate,
                        ),
                        edge_index=current_edge_index[:,torch.cat([mask_user, mask_item]) & torch.cat([mask_item, mask_user])],
                        edge_weight=masked_edge_weight
                    )
                )

        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi

    def edge_index_to_adj(self, edge_index):
        """
        Convert edge index and weights to a SparseTensor adjacency matrix.
        """
        rows = edge_index[0].long().to(self.device)
        cols = edge_index[1].long().to(self.device)
        values = edge_index[2].float().to(self.device)

        return SparseTensor(
            row=rows,
            col=cols,
            value=values,
            sparse_sizes=(
                self.num_users + self.num_items,
                self.num_users + self.num_items,
            ),
        )

    def update_adjacency(self, node_embeddings):
        """
        Update edge weights using the selected aggregation method.
        """
        row, col = self.edge_index[:2]

        row, col = row.long().to(self.device), col.long().to(self.device)
        row_nodes = node_embeddings.to(self.device)[row[: row.shape[0] // 2]]
        col_nodes = node_embeddings.to(self.device)[col[: col.shape[0] // 2]]

        if self.aggr == "sim":
            user_item = torch.relu(
                torch.nn.functional.cosine_similarity(
                    torch.mul(
                        row_nodes, self.projection(self.edge_embeddings_interactions)
                    ),
                    torch.mul(
                        col_nodes, self.projection(self.edge_embeddings_interactions)
                    ),
                )
            )
            updates = torch.concat([user_item, user_item], dim=0)
            return updates

        elif self.aggr == "nn":
            user_item = torch.squeeze(
                self.dense_network(
                    torch.concat(
                        [row_nodes, self.edge_embeddings_interactions, col_nodes],
                        dim=-1,
                    )
                )
            )
            updates = torch.concat([user_item, user_item], dim=0)
            return updates

        else:   # 'att'
            edge_index = self.edge_index[:, : self.edge_index.shape[1] // 2].clone()
            edge_index = torch.concat(
                [
                    edge_index.to(self.device),
                    torch.ones((1, edge_index.shape[1]), device=self.device),
                ]
            )
            _, user_item = self.attention(
                node_embeddings,
                self.edge_index_to_adj(edge_index),
                self.edge_embeddings_interactions,
                return_attention_weights=True,
            )
            user_item = torch.squeeze(user_item.coo()[2])
            updates = torch.concat([user_item, user_item], dim=0)
            return updates

    def forward(self, inputs, **kwargs):
        """
        Compute predicted scores for user-item pairs.
        """
        gu, gi, bu, bi = inputs

        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        beta_u = torch.squeeze(bu).to(self.device)
        beta_i = torch.squeeze(bi).to(self.device)

        mu = torch.squeeze(self.Mu).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1) + beta_u + beta_i + mu

        return xui

    def predict(self, gu, gi, users, items, **kwargs):
        """
        Predict ratings for given users and items.
        """
        rui = self.forward(
            inputs=(gu, gi, self.Bu.weight[users], self.Bi.weight[items])
        )
        return rui

    def train_step(self, batch, mask):
        """
        Perform a single training step with contrastive loss.
        Returns loss and gradient statistics for logging.
        """
        all_embeddings = torch.cat(
            (self.Gu.to(self.device), self.Gi.to(self.device)), 0
        )

        updates = self.update_adjacency(all_embeddings)

        f_At = self.row_normalize(updates, self.edge_index, self.num_users + self.num_items)

        continuous_weights = (
                self.lm * self.L0.to(self.device) + (1 - self.lm) * f_At
        )

        final_values = self.soft_threshold(continuous_weights)

        self.current_edge_weights = final_values
        self.current_edge_index = self.edge_index[:2].long()

        # Masks for full graph and two contrastive views
        mask_all_true_user = torch.full(mask[0].shape, True, dtype=bool)
        mask_all_true_item = torch.full(mask[1].shape, True, dtype=bool)

        # Full graph embeddings
        gu, gi = self.propagate_embeddings(mask_user=mask_all_true_user, mask_item=mask_all_true_item)

        # Contrastive views
        gu1, gi1 = self.propagate_embeddings(mask_user=mask[0], mask_item=mask[1])
        gu2, gi2 = self.propagate_embeddings(mask_user=mask[2], mask_item=mask[3])
        
        gu1, gi1 = torch.nn.functional.normalize(gu1, dim=1), torch.nn.functional.normalize(gi1, dim=1)
        gu2, gi2 = torch.nn.functional.normalize(gu2, dim=1), torch.nn.functional.normalize(gi2, dim=1)
        
        user, item, r = batch
        
        # MSE loss for main view
        rui = self.forward(
            inputs=(gu[user], gi[item], self.Bu.weight[user], self.Bi.weight[item])
        )

        mse_loss = self.mse_loss(
            torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float)
        )
        
        # Contrastive loss between views
        user_nd_loss = self.contrast_loss(gu1[user], gu2[user]).mean()
        item_nd_loss = self.contrast_loss(gi1[item], gi2[item]).mean()
        nd_loss = (user_nd_loss + item_nd_loss) / 2.0
        
        total_loss = mse_loss + self.alpha * nd_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        self.optimizer.step()

        # Return loss and statistics for this batch
        return total_loss.detach().cpu().numpy(), mse_loss, nd_loss
