from tqdm import tqdm
import numpy as np
import torch
import random
import pandas as pd
import os
from ast import literal_eval as make_tuple
from torch_geometric.utils import degree
from torch.utils.tensorboard import SummaryWriter

from .pointwise_pos_neg_sampler import Sampler

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .RoGERModel import RoGERModel
from datetime import datetime

class RoGER(RecMixin, BaseRecommenderModel):
    """
    Reviews on Graph Edges for Recommendation
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Initialize the RoGER recommender.
        Loads parameters, prepares dataframes for validation and test,
        sets up the sampler and model.
        """
        # Parameter list for automatic configuration
        self._params_list = [
            ("_lr", "lr", "lr", 0.0005, float, None),
            ("_emb", "emb", "emb", 64, int, None),
            ("_batch_eval", "batch_eval", "bch_ev", 512, int, None),
            ("_n_layers", "n_layers", "n_ly", 3, int, None),
            ("_lambda", "lambda", "lmd", 0.1, float, None),
            ("_drop", "drop", "drop", 0.1, float, None),
            ("_aggr", "aggr", "aggr", 'sim', str, None),
            ("_dense", "dense", "dense", "(32,16,8)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_loader", "loader", "load", 'InteractionsTextualAttributes', str, None),
            ("_alpha", "alpha", "nda", 0.01, float, None),
            ("_factor", "factor", "fct", 0.1, float, None),
            ("_patience", "patience", "ptn", 0, int, None),
            ("_node_dropout", "node_dropout", "nd", 0.1, float, None),
            ("_weight_decay", "weight_decay", "wd", 1e-4, float, None),
            ("_save_adj", "save_adj", "s_adj", False, bool, None)
        ]
        self.autoset_params()

        # TensorBoard writer for logging
        self.writer = SummaryWriter(log_dir=f'./log/runs/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}/')

        np.random.seed(self._seed)
        random.seed(self._seed)

        # Sampler for training batches
        self._sampler = Sampler(self._batch_size, self._data.transactions)

        # Prepare validation and test dataframes
        self.df_val_rat = pd.DataFrame(columns=['user', 'item', 'rating'])
        self.df_test_rat = pd.DataFrame(columns=['user', 'item', 'rating'])

        # create dataframes for validation and test set with user-item-rating
        idx = 0
        for k, v in data.val_dict.items():
            for kk, vv in v.items():
                self.df_val_rat = pd.concat([self.df_val_rat,
                                             pd.DataFrame({'user': k, 'item': int(kk), 'rating': vv}, index=[idx])])
                idx += 1

        idx = 0
        for k, v in data.test_dict.items():
            for kk, vv in v.items():
                self.df_test_rat = pd.concat([self.df_test_rat,
                                              pd.DataFrame({'user': k, 'item': int(kk), 'rating': vv}, index=[idx])])
                idx += 1

        # Ensure correct types
        self.df_val_rat = self.df_val_rat.astype({'user': int, 'item': int, 'rating': float})
        self.df_test_rat = self.df_test_rat.astype({'user': int, 'item': int, 'rating': float})

        # Map public user/item IDs
        self.df_val_rat['user'] = self.df_val_rat['user'].map(data.public_users)
        self.df_val_rat['item'] = self.df_val_rat['item'].map(data.public_items)
        self.df_test_rat['user'] = self.df_test_rat['user'].map(data.public_users)
        self.df_test_rat['item'] = self.df_test_rat['item'].map(data.public_items)

        # remove interactions whose user and/or item is not in the training set
        self.df_val_rat = self.df_val_rat[self.df_val_rat['user'] <= self._num_users - 1]
        self.df_test_rat = self.df_test_rat[self.df_test_rat['user'] <= self._num_users - 1]
        self.df_val_rat = self.df_val_rat[self.df_val_rat['item'] <= self._num_items - 1]
        self.df_test_rat = self.df_test_rat[self.df_test_rat['item'] <= self._num_items - 1]

        # Load edge textual features
        self._side_edge_textual = self._data.side_information.InteractionsTextualAttributes

        # Build edge index for the graph
        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        edge_index = torch.tensor(np.array([list(row) + col, col + list(row)]))

        edge_features = self._side_edge_textual.object.get_all_features()

        # Initialize the RoGER model
        self._model = RoGERModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._lr,
            embed_k=self._emb,
            n_layers=self._n_layers,
            edge_features=edge_features,
            edge_index=edge_index,
            lm=self._lambda,
            aggr=self._aggr,
            drop=self._drop,
            dense=self._dense,
            random_seed=self._seed,
            alpha=self._alpha,
            factor=self._factor,
            patience=self._patience,
            weight_decay=self._weight_decay,
            save_adj=self._save_adj
        )

    @property
    def name(self):
        """
        Returns the model name with parameter shortcuts.
        """
        return "RoGER" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def norm(self, edge_index):
        """
        Normalize edge weights for GCN propagation.
        """
        row, col = edge_index
        deg = degree(col, self._num_users + self._num_items)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return torch.stack([row, col, norm], dim=0)

    def train(self):
        """
        Main training loop for RoGER.
        Handles batching, loss computation, logging, and learning rate scheduling.
        """
        if self._restore:
            return self.restore_weights()

        row, col = self._data.sp_i_train.nonzero()
        ratings = self._data.sp_i_train_ratings.data
        edge_index = np.array([row, col, ratings]).transpose()

        for it in self.iterate(self._epochs):
            if (it % 2 == 0)and(self._save_adj==True):
                gu, gi = self._model.propagate_embeddings(evaluate=True)
                all_embeddings = torch.cat((gu, gi), 0)
                updates = self._model.update_adjacency(all_embeddings, evaluate=True)
                final_values = self._model.lm * self._model.L0 + (1 - self._model.lm) * updates
                edge_index_full = torch.stack(
                    [self._model.edge_index[0], self._model.edge_index[1], final_values], dim=0
                )
                adj = self._model.edge_index_to_adj(edge_index_full)
                adj = adj.coalesce()
                n_users = self._num_users
                n_items = self._num_items

                # Estrai solo gli archi da utenti (righe 0:n_users) a item (colonne n_users:n_users+n_items)
                row, col, values = adj.coo()

                # Mask: solo righe utenti e colonne item
                mask = (row < n_users) & (col >= n_users) & (col < n_users + n_items)
                user_idx = row[mask]
                item_idx = col[mask] - n_users  # shift per portare le colonne da [n_users, n_users+n_items) a [0, n_items)
                bi_values = values[mask]

                # Crea la matrice sparsa bi-adiacenza n_users x n_items
                bi_adj = torch.sparse_coo_tensor(
                    torch.stack([user_idx, item_idx], dim=0),
                    bi_values,
                    (n_users, n_items)
                )
                #controlla che la cartella esista altrimenti la crea
                if not os.path.exists(f"./adj/{self._config.dataset}"):
                    os.makedirs(f"./adj/{self._config.dataset}")
                # Salva la matrice
                torch.save(bi_adj, f"./adj/{self._config.dataset}/bi_adj_epoch_{it}.pt")

            loss = 0
            steps = 0
            grad_norm = 0
            mse_loss = 0
            nd_loss = 0
            grad_norm = {}
            weight_distributions = {}

            # Shuffle edge index for each epoch
            np.random.shuffle(edge_index)
            edge_index = edge_index.astype(int)
            
            # Create dropout masks for contrastive views
            mask_user_1, mask_item_1 = self.create_adj_mask(edge_index)
            mask_user_2, mask_item_2 = self.create_adj_mask(edge_index)

            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(edge_index):
                    steps += 1
                    # Training step returns loss and gradient statistics
                    loss_t, grad_norm_dict, weight_distributions_dict, mse_loss_t, nd_loss_t = self._model.train_step(batch, mask=[mask_user_1, mask_item_1 , mask_user_2, mask_item_2] )
                    loss += loss_t
                    # Aggregate gradient norms
                    if isinstance(grad_norm_dict, dict):
                        for key, value in grad_norm_dict.items():
                            if key not in grad_norm:
                                grad_norm[key] = np.array([value])
                            else:
                                grad_norm[key] = np.append(grad_norm[key],value)
                    # Aggregate weight distributions
                    if isinstance(weight_distributions_dict, dict):
                        for key, value in weight_distributions_dict.items():
                            if key not in weight_distributions:
                                weight_distributions[key] = np.array([value])  # Inizializza un nuovo array con il primo valore
                            else:
                                weight_distributions[key] = np.append(weight_distributions[key],value)  # Aggiungi il valore all'array esistente
                    # else:
                        # Optionally, handle the case where grad_norm_dict is not a dictionary,
                        # for example, if train_step might return a scalar or None for grad_norm.
                        # If grad_norm_dict is guaranteed to be a dict, this else is not needed.
                        # Example: if grad_norm_dict is a scalar and grad_norm is not a dict (still 0)
                        # elif isinstance(grad_norm_dict, (int, float)) and not isinstance(grad_norm, dict):
                        #    grad_norm += grad_norm_dict
                    mse_loss += mse_loss_t
                    nd_loss += nd_loss_t
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()
                    
            #if isinstance(grad_norm, dict) and steps > 0:
            #    for key in grad_norm:
            #        grad_norm[key] /= steps
            #self.logger.info(f"Epoch {it + 1}: {grad_norm}")
            # Log gradient norms and weight distributions to TensorBoard
            for key, value in grad_norm.items():
                self.writer.add_histogram(f'GradNorm/{key}', value, it)
            for key, value in weight_distributions.items():
                self.writer.add_histogram(f'WeightDistribution/{key}', value, it)
            self.logger.info(f"Epoch {it + 1}: Loss: {loss / steps:.5f} - MSE Loss: {mse_loss / steps:.5f} - ND Loss: {nd_loss / steps:.5f}")
            self.writer.add_scalar('Tot_Loss', loss/steps, it)
            self.writer.add_scalar('MSE_Loss', mse_loss/steps, it)
            self.writer.add_scalar('ND_Loss', nd_loss/steps, it)
            self.evaluate(it, loss / (it + 1))

            # Learning rate scheduler step based on validation metric
            val_metric_value = self._results[-1][0]["val_results"]["MSE"]
            if val_metric_value is not None:
                old_lr = self._model.optimizer.param_groups[0]['lr']
                self._model.scheduler.step(val_metric_value)
                new_lr = self._model.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    self.logger.info(f"Epoch {it + 1}: Learning rate reduced from {old_lr:.8f} to {new_lr:.8f} based on {self._validation_metric}: {val_metric_value:.5f}")
            else:
                self.logger.warning(f"Epoch {it + 1}: Validation metric '{self._validation_metric}' is None. Skipping scheduler step.")
            self.writer.add_scalar('LR', self._model.optimizer.param_groups[0]['lr'], it)
            self.writer.add_scalar('MSE_val', val_metric_value, it)
            self.writer.add_scalar('MSE_test', self._results[-1][0]["test_results"]["MSE"], it)
            self.writer.flush()
        self.writer.close()

    def create_adj_mask(self, edge_index):
        """
        Create dropout masks for users and items for contrastive learning.
        """
        users_to_drop = random.sample(self._data.users, round(self._data.num_users * self._node_dropout))
        items_to_drop = random.sample(self._data.items, round(self._data.num_items * self._node_dropout))
        mask_user = ~np.isin(edge_index[:, 0], list(users_to_drop))
        mask_item = ~np.isin(edge_index[:, 1], list(items_to_drop))
    
        return mask_user, mask_item

    def get_recommendations(self, k: int = 100):
        """
        Generate recommendations for validation and test sets.
        """
        predictions_test = []
        predictions_val = []
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        val_len = len(self.df_val_rat)
        with tqdm(total=int(val_len // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, val_len, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, val_len)
                current_df = self.df_val_rat[offset:offset_stop]
                p = self._model.predict(gu[current_df['user'].tolist()], gi[current_df['item'].tolist()],
                                        current_df['user'].tolist(), current_df['item'].tolist())
                predictions_val += p.detach().cpu().numpy().tolist()
                t.update()
        test_len = len(self.df_test_rat)
        with tqdm(total=int(test_len // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, test_len, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, test_len)
                current_df = self.df_test_rat[offset:offset_stop]
                p = self._model.predict(gu[current_df['user'].tolist()], gi[current_df['item'].tolist()],
                                        current_df['user'].tolist(), current_df['item'].tolist())
                predictions_test += p.detach().cpu().numpy().tolist()
                t.update()
        return predictions_val, predictions_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        """
        Get top-k recommendations for a single batch of users.
        """
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0.0):
        """
        Evaluate the model on validation and test sets.
        Logs metrics and saves the best model if needed.
        """
        if (it is None) or (not (it + 1) % self._validation_rate):
            predictions_val, predictions_test = self.get_recommendations()
            true_val, true_test = self.df_val_rat['rating'].to_numpy(), self.df_test_rat['rating'].to_numpy()
            result_dict = self.evaluator.eval_error(np.array(predictions_val), true_val, np.array(predictions_test),
                                                    true_test)
    
            predictions_val = (np.array(predictions_val) >= 4).astype(int)
            predictions_test = (np.array(predictions_test) >= 4).astype(int)
            true_val = (true_val >= 4).astype(int)
            true_test = (true_test >= 4).astype(int)

            result_dict_prova = self.evaluator.eval_auc(predictions_val, true_val, predictions_test, true_test)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss:.5f}')
            else:
                self.logger.info(f'Finished')

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][0]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def get_loss(self):
        """
        Returns the best loss value for optimization.
        """
        if self._optimize_internal_loss:
            return min(self._losses)
        else:
            return min([r[0]["val_results"][self._validation_metric] for r in self._results])

    def get_best_arg(self):
        """
        Returns the index of the best epoch according to the validation metric.
        """
        if self._optimize_internal_loss:
            val_results = np.argmin(self._losses)
        else:
            val_results = np.argmin([r[0]["val_results"][self._validation_metric] for r in self._results])
        return val_results

    def restore_weights(self):
        """
        Restore model and optimizer weights from a checkpoint file.
        """
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False
