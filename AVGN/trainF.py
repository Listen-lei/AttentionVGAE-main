import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sknetwork.clustering import Louvain
from sklearn.cluster import SpectralClustering, KMeans
from tqdm import tqdm


class Train:
    def __init__(self,
                 processed_data,
                 graph_dict,
                 model,
                 pre_epochs,
                 epochs,
                 corrupt=0.001,
                 lr=5e-4,
                 weight_decay=1e-4,
                 domains=None,
                 kl_weight=100,
                 mse_weight=10,
                 bce_kld_weight=0.1,
                 domain_weight=1,
                 use_gpu=True,
                 ):

        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.processed_data = processed_data
        self.data = torch.FloatTensor(processed_data.copy()).to(self.device)
        self.adj = graph_dict['adj_norm'].to(self.device)
        self.adj_label = graph_dict['adj_label'].to(self.device)
        self.norm = graph_dict['norm_value']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr=lr, weight_decay=weight_decay)
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.num_spots = self.data.shape[0]
        self.dec_tol = 0
        self.kl_weight = kl_weight
        self.q_stride = 20
        self.mse_weight = mse_weight
        self.bce_kld_weight = bce_kld_weight
        self.domain_weight = domain_weight
        self.corrupt = corrupt
        if domains is not None:
            self.domains = torch.from_numpy(domains).to(self.device)
            print("Initializing with domains.")
        else:
            self.domains = domains

    def pretrain(self, grad_down=5):
        with tqdm(total=int(self.pre_epochs), desc="Pretraining an initial model",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.pre_epochs):
                inputs_corr = masking_noise(self.data, self.corrupt)
                inputs_coor = inputs_corr.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()

                if self.domains is None:
                    z, mu, logvar, de_feat, _, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
                    preds = self.model.dc(z)
                else:
                    z, mu, logvar, de_feat, _, feat_x, gnn_z, domain_pred = self.model(Variable(inputs_coor), self.adj)
                    preds = self.model.model.dc(z)

                loss = self.model.AVGN_loss(
                    decoded=de_feat, x=self.data, preds=preds, labels=self.adj_label, mu=mu, logvar=logvar,
                    n_nodes=self.num_spots, norm=self.norm, mask=self.adj_label, mse_weight=self.mse_weight,
                    bce_kld_weight=self.bce_kld_weight,
                )

                if self.domains is not None:
                    loss_function = nn.CrossEntropyLoss()
                    domain_loss = loss_function(domain_pred, self.domains)
                    loss1 = loss
                    loss = domain_loss * self.domain_weight + loss1
                else:
                    loss = loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_down)
                self.optimizer.step()
                pbar.update(1)

    @torch.no_grad()
    def process(self):
        self.model.eval()
        if self.domains is None:
            z, _, _, _, q, *_ = self.model(self.data, self.adj)
        else:
            z, _, _, _, q, _, _, _ = self.model(self.data, self.adj)
        z = z.cpu().detach().numpy()
        q = q.cpu().detach().numpy()

        return z, q

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def fit(self, cluster_n=20, cluster_type='Louvain', res=1.0, pretrain=True):
        if pretrain:
            self.pretrain()
            pre_z, _ = self.process()
        if cluster_type == 'KMeans':
            cluster_method = KMeans(n_clusters=cluster_n, n_init=cluster_n * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
        elif cluster_type == 'Louvain':
            cluster_data = sc.AnnData(pre_z)
            sc.pp.neighbors(cluster_data, n_neighbors=cluster_n)
            sc.tl.louvain(cluster_data, resolution=res)
            y_pred_last = cluster_data.obs['louvain'].astype(int).to_numpy()
            n_clusters = len(np.unique(y_pred_last))
            features = pd.DataFrame(pre_z, index=np.arange(0, pre_z.shape[0]))
            Group = pd.Series(y_pred_last, index=np.arange(0, features.shape[0]), name="Group")
            Mergefeature = pd.concat([features, Group], axis=1)
            cluster_centers_ = np.asarray(Mergefeature.groupby("Group").mean())
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)

        with tqdm(total=int(self.pre_epochs), desc="Training a final model",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.epochs):
                if epoch % self.q_stride == 0:
                    _, q = self.process()
                    q = self.model.target_distribution(torch.Tensor(q).clone().detach())
                    y_pred = q.cpu().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    self.model.train()
                    if epoch > 0 and delta_label < self.dec_tol:
                        print('Delta_label: {:.4f}'.format(delta_label), '< Tolerance:', self.dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                torch.set_grad_enabled(True)
                self.model.train()
                self.optimizer.zero_grad()
                inputs_coor = self.data.to(self.device)

                if self.domains is None:
                    z, mu, logvar, de_feat, out_q, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
                    preds = self.model.dc(z)
                else:
                    z, mu, logvar, de_feat, out_q, feat_x, gnn_z, domain_pred = self.model(Variable(inputs_coor),
                                                                                          self.adj)
                    loss_function = nn.CrossEntropyLoss()
                    domain_loss = loss_function(domain_pred, self.domains)
                    preds = self.model.model.dc(z)

                avgn_loss = self.model.AVGN_loss(
                    decoded=de_feat, x=self.data, preds=preds, labels=self.adj_label, mu=mu, logvar=logvar,
                    n_nodes=self.num_spots, norm=self.norm, mask=self.adj_label, mse_weight=self.mse_weight,
                    bce_kld_weight=self.bce_kld_weight
                )

                loss_kl = F.kl_div(out_q.log(), q.to(self.device))
                if self.domains is None:
                    loss = self.kl_weight * loss_kl + avgn_loss
                else:
                    loss = self.kl_weight * loss_kl + avgn_loss + domain_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                pbar.update(1)


def masking_noise(data, frac):
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand < frac] = 0
    return data_noise
