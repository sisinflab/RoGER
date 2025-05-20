from abc import ABC

from torch_geometric.nn import GCNConv, GATConv
from collections import OrderedDict

# modifica aggiunta variabile ambiente per torch.use_deterministic
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # oppure ":16:8"
# aggiunta import scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
# aggiunta import ContrastLoss
from .ContrastLoss import ContrastLoss

import torch
import torch_geometric
import numpy as np
import random

from torch_sparse import SparseTensor


class RoGERModel(torch.nn.Module, ABC):
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
        name="RoGER",
        **kwargs
    ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.weight_decay = weight_decay

        self.L0 = torch.ones(
            (edge_index.shape[1],), dtype=torch.float32, device=self.device
        )
        self.edge_index = edge_index.to(self.device)

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

        # modifica aggiunta squeeze()
        self.edge_embeddings_interactions = torch.tensor(
            edge_features, dtype=torch.float32, device=self.device
        ).squeeze()
        # modifica shape[2] invece di shape[1]
        self.feature_dim = edge_features.shape[2]

        # create node-node textual
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
                    "x, edge_index -> x",
                )
            )

        self.node_node_textual_network = torch_geometric.nn.Sequential(
            "x, edge_index", propagation_node_node_textual_list
        )
        self.node_node_textual_network.to(self.device)

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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # modifica aggiunta scheduler lr
        # mode='min' perché monitoriamo MSE (vogliamo minimizzarlo)
        # factor=0.1 riduce LR a LR * 0.1
        # patience=5 attende 5 epoche senza miglioramenti prima di ridurre LR
        # verbose=True stampa un messaggio quando LR viene ridotto
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.factor, patience=self.patience
        )

        self.mse_loss = torch.nn.MSELoss()
        self.contrast_loss = ContrastLoss(feat_size=self.embed_k).to(self.device)

    def propagate_embeddings(self, mask_user=None, mask_item=None, evaluate=False):
        all_embeddings = torch.cat(
            (self.Gu.to(self.device), self.Gi.to(self.device)), 0
        )
        for layer in range(self.n_layers):
            if evaluate:
                if self.aggr == "nn":
                    self.dense_network.eval()
                with torch.no_grad():
                    updates = self.update_adjacency(all_embeddings, evaluate=evaluate)
                    final_values = (
                        self.lm * self.L0.to(self.device) + (1 - self.lm) * updates
                    )
                    edge_index = torch.stack(
                        [self.edge_index[0], self.edge_index[1], final_values], dim=0
                    )
                    all_embeddings = torch.relu(
                        list(self.node_node_textual_network.children())[layer](
                            all_embeddings.to(self.device),
                            self.edge_index_to_adj(edge_index).to(self.device),
                        )
                    )
            else:
                updates = self.update_adjacency(all_embeddings, mask_user, mask_item)
                final_values = (
                    self.lm * self.L0[np.concatenate((mask_user,mask_item)) & np.concatenate((mask_item, mask_user))].to(self.device) + (1 - self.lm) * updates
                )
                edge_index = torch.stack(
                    [self.edge_index[0, np.concatenate((mask_user,mask_item)) & np.concatenate((mask_item, mask_user))], self.edge_index[1, np.concatenate((mask_user,mask_item)) & np.concatenate((mask_item, mask_user))], final_values], dim=0
                )
                all_embeddings = torch.relu(
                    list(self.node_node_textual_network.children())[layer](
                        torch.dropout(
                            all_embeddings.to(self.device),
                            p=self.drop,
                            train=not evaluate,
                        ),
                        self.edge_index_to_adj(edge_index).to(self.device),
                    )
                )

        if evaluate:
            if self.aggr == "nn":
                self.dense_network.train()

        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi

    def edge_index_to_adj(self, edge_index):
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

    def update_adjacency(self, node_embeddings, mask_user=None, mask_item=None, evaluate=False):
        if evaluate==False:
            edge_embeddings_interactions = self.edge_embeddings_interactions[mask_user & mask_item, :]
            row, col = self.edge_index[:, np.concatenate((mask_user,mask_item)) & np.concatenate((mask_item, mask_user))]
        else:
            edge_embeddings_interactions = self.edge_embeddings_interactions
            row, col = self.edge_index
            mask_user = np.full(edge_embeddings_interactions.shape[0], True, dtype=bool)
            mask_item = np.full(edge_embeddings_interactions.shape[0], True, dtype=bool)
        
        row, col = row.long(), col.long()
        row_nodes = node_embeddings[row[: row.shape[0] // 2]]
        col_nodes = node_embeddings[col[: col.shape[0] // 2]]

        if self.aggr == "sim":
            user_item = torch.relu(
                torch.nn.functional.cosine_similarity(
                    torch.mul(
                        row_nodes, self.projection(edge_embeddings_interactions)
                    ),
                    torch.mul(
                        col_nodes, self.projection(edge_embeddings_interactions)
                    ),
                )
            )
            updates = torch.concat([user_item, user_item], dim=0)
            return updates

        elif self.aggr == "nn":
            user_item = torch.squeeze(
                self.dense_network(
                    torch.concat(
                        [row_nodes, edge_embeddings_interactions, col_nodes],
                        dim=-1,
                    )
                )
            )
            updates = torch.concat([user_item, user_item], dim=0)
            return updates

        else:
            edge_index = self.edge_index[:, : self.edge_index.shape[1] // 2].clone()
            edge_index = edge_index[:, mask_user & mask_item]
            edge_index = torch.concat(
                [
                    edge_index.to(self.device),
                    torch.ones((1, edge_index.shape[1]), device=self.device),
                ]
            )
            _, user_item = self.attention(
                node_embeddings,
                self.edge_index_to_adj(edge_index),
                edge_embeddings_interactions,
                return_attention_weights=True,
            )
            user_item = torch.squeeze(user_item.coo()[2])
            updates = torch.concat([user_item, user_item], dim=0)
            return updates

    def forward(self, inputs, **kwargs):
        gu, gi, bu, bi = inputs

        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        beta_u = torch.squeeze(bu).to(self.device)
        beta_i = torch.squeeze(bi).to(self.device)

        mu = torch.squeeze(self.Mu).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1) + beta_u + beta_i + mu

        return xui

    def predict(self, gu, gi, users, items, **kwargs):
        rui = self.forward(
            inputs=(gu, gi, self.Bu.weight[users], self.Bi.weight[items])
        )
        return rui

    #    def train_step(self, batch):
    #        gu, gi = self.propagate_embeddings()
    #        user, item, r = batch
    #        rui = self.forward(inputs=(gu[user], gi[item],
    #                        self.Bu.weight[user], self.Bi.weight[item]))
    #
    #        loss = self.loss(torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float))
    #
    #        self.optimizer.zero_grad()
    #        loss.backward()
    #        self.optimizer.step()
    #
    #        return loss.detach().cpu().numpy()

    # modifica train_step con l'aggiunta di una contrastive loss
    def train_step(self, batch, mask):
        # Create boolean masks full of True values with the same shape as mask[0] and mask[1]
        mask_all_true_user = np.full(mask[0].shape, True, dtype=bool)
        mask_all_true_item = np.full(mask[1].shape, True, dtype=bool)
        # Pass the all-True masks to propagate_embeddings
        gu, gi = self.propagate_embeddings(mask_user=mask_all_true_user, mask_item=mask_all_true_item)
        #generazione delle due view
        gu1, gi1 = self.propagate_embeddings(mask_user=mask[0], mask_item=mask[1])
        gu2, gi2 = self.propagate_embeddings(mask_user=mask[2], mask_item=mask[3])
        
        gu1, gi1 = torch.nn.functional.normalize(gu1, dim=1), torch.nn.functional.normalize(gi1, dim=1)
        gu2, gi2 = torch.nn.functional.normalize(gu2, dim=1), torch.nn.functional.normalize(gi2, dim=1)
        
        user, item, r = batch
        
        # calcolo della mse loss per la prima view
        rui = self.forward(
            inputs=(gu[user], gi[item], self.Bu.weight[user], self.Bi.weight[item])
        )

        mse_loss = self.mse_loss(
            torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float)
        )
        
        # calcolo della contrastive loss
        user_nd_loss = self.contrast_loss(gu1[user], gu2[user]).mean()
        item_nd_loss = self.contrast_loss(gi1[item], gi2[item]).mean()
        nd_loss = (user_nd_loss + item_nd_loss) / 2.0
        #print(f"\ncontrastive loss: {nd_loss} | mse loss: {mse_loss}\n")
        
        total_loss = mse_loss + self.alpha * nd_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # --- Debug: Calcolo della norma L2 dei gradienti per parametro per il batch corrente ---
        batch_gradient_norms = {}
        batch_weight_distributions = {}
        for name, p in self.named_parameters():
            if p.grad is not None:
                batch_gradient_norms[name] = p.grad.data.norm(2).item()
            batch_weight_distributions[name] = p.data.detach().cpu().numpy()
        # --- Fine Debug ---


        self.optimizer.step()

        # Restituisce la loss e la somma delle magnitudini dei gradienti per questo batch
        # La media a livello di epoca dovrà essere calcolata nel loop di training esterno
        return total_loss.detach().cpu().numpy(), batch_gradient_norms, batch_weight_distributions, mse_loss, nd_loss
