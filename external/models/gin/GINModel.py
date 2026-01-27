"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC
from torch_geometric.nn import GINConv

import torch
import torch_geometric
from collections import OrderedDict
import numpy as np
import random


class GINModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 n_layers,
                 nn_weight_size,
                 eps,
                 train_eps,
                 adj,
                 random_seed,
                 name="GIN",
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.adj = adj

        self.nn_weight_size_list = [self.embed_k] + nn_weight_size
        self.nn_num_layers = len(self.nn_weight_size_list) - 1
        self.eps = eps
        self.train_eps = train_eps
        self.n_layers = n_layers

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        self.Bu = torch.nn.Embedding(self.num_users, 1)
        torch.nn.init.xavier_normal_(self.Bu.weight)
        self.Bu.to(self.device)
        self.Bi = torch.nn.Embedding(self.num_items, 1)
        torch.nn.init.xavier_normal_(self.Bi.weight)
        self.Bi.to(self.device)

        self.Mu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((1, 1))))
        self.Mu.to(self.device)

        propagation_network_list = []

        for layer in range(self.n_layers):
            nn_list = []
            for layer_nn in range(self.nn_num_layers):
                nn_list.append(
                    ('nn_' + str(layer) + '_' + str(layer_nn),
                     torch.nn.Linear(in_features=self.nn_weight_size_list[layer_nn],
                                     out_features=self.nn_weight_size_list[layer_nn + 1],
                                     bias=True)))
            current_nn = torch.nn.Sequential(OrderedDict(nn_list))
            current_nn.to(self.device)

            propagation_network_list.append((GINConv(
                nn=current_nn,
                eps=self.eps,
                train_eps=self.train_eps
            ), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss = torch.nn.MSELoss()

    def propagate_embeddings(self, evaluate=False):
        current_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)

        for layer in range(self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    current_embeddings = torch.relu(list(
                        self.propagation_network.children()
                    )[layer](current_embeddings.to(self.device), self.adj.to(self.device)))
            else:
                current_embeddings = torch.relu(list(
                    self.propagation_network.children()
                )[layer](current_embeddings.to(self.device), self.adj.to(self.device)))

        if evaluate:
            self.propagation_network.train()

        gu, gi = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi

    def forward(self, inputs, **kwargs):
        gu, gi, bu, bi, = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        beta_u = torch.squeeze(bu).to(self.device)
        beta_i = torch.squeeze(bi).to(self.device)

        mu = torch.squeeze(self.Mu).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1) + beta_u + beta_i + mu

        return xui

    def predict(self, gu, gi, users, items, **kwargs):
        return self.forward(inputs=(gu, gi, self.Bu.weight[users], self.Bi.weight[items]))

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, item, r = batch
        rui = self.forward(inputs=(gu[user], gi[item], self.Bu.weight[user], self.Bi.weight[item]))

        loss = self.loss(torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()
