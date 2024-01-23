import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch_geometric
from torch_geometric.nn import Sequential, BatchNorm

class AVGN_model(nn.Module):
    def __init__(self, 
                input_dim,              
                Conv_type='GCNConv',
                linear_encoder_hidden=[32, 20],
                linear_decoder_hidden=[32],
                conv_hidden=[32, 8],
                p_drop=0.01,
                dec_cluster_n=15,
                alpha=0.9,
                activate="relu",
                ):
        super(AVGN_model, self).__init__()
        self.input_dim = input_dim
        self.Conv_type = Conv_type
        self.alpha = alpha
        self.conv_hidden = conv_hidden
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n

        current_encoder_dim = self.input_dim
        num_head = 4
        
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 20)
        self.multihead_attention1 = nn.MultiheadAttention(embed_dim=32, num_heads=num_head)
        self.multihead_attention2 = nn.MultiheadAttention(embed_dim=20, num_heads=num_head)
        self.batchnorm1 = nn.BatchNorm1d(32, momentum=0.01, eps=0.001)
        self.batchnorm2 = nn.BatchNorm1d(20, momentum=0.01, eps=0.001)
        if activate == "relu":
            self.activation = nn.ELU()
        elif activate == "sigmoid":
            self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p_drop)

        self.encoder = nn.Sequential()
        for le in range(len(linear_encoder_hidden)):
            self.encoder.add_module(f'encoder_L{le}', 
                                    buildNetwork(current_encoder_dim, self.linear_encoder_hidden[le], self.activate, self.p_drop))
            current_encoder_dim = linear_encoder_hidden[le]
        current_decoder_dim = linear_encoder_hidden[-1] + conv_hidden[-1]

        self.linear3 = nn.Linear(28, 32)
        self.linear4 = nn.Linear(32, 128)
        self.multihead_attention3 = nn.MultiheadAttention(embed_dim=32, num_heads=num_head)
        self.batchnorm3 = nn.BatchNorm1d(32, momentum=0.01, eps=0.001)
        if activate == "relu":
            self.activation = nn.ELU()
        elif activate == "sigmoid":
            self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p_drop)

        conv_types = [
            'MessagePassing', 'SimpleConv', 'GCNConv', 'ChebConv', 'SAGEConv', 'CuGraphSAGEConv', 'GraphConv',
            'GravNetConv', 'GatedGraphConv', 'ResGatedGraphConv', 'GATConv', 'CuGraphGATConv', 'FusedGATConv',
            'GATv2Conv', 'TransformerConv', 'AGNNConv', 'TAGConv', 'GINConv', 'GINEConv', 'ARMAConv', 'SGConv',
            'SSGConv', 'APPNP', 'MFConv', 'RGCNConv', 'FastRGCNConv', 'CuGraphRGCNConv', 'RGATConv', 'SignedConv',
            'DNAConv', 'PointNetConv', 'GMMConv', 'SplineConv', 'NNConv', 'CGConv', 'EdgeConv', 'DynamicEdgeConv',
            'XConv', 'PPFConv', 'FeaStConv', 'PointTransformerConv', 'HypergraphConv', 'LEConv', 'PNAConv',
            'ClusterGCNConv', 'GENConv', 'GCN2Conv', 'PANConv', 'WLConv', 'WLConvContinuous', 'FiLMConv', 'SuperGATConv',
            'FAConv', 'EGConv', 'PDNConv', 'GeneralConv', 'HGTConv', 'FastHGTConv', 'HEATConv', 'HeteroConv', 'HANConv',
            'LGConv', 'PointGNNConv', 'GPSConv', 'AntiSymmetricConv'
        ]

        for conv_type in conv_types:
            if self.Conv_type == conv_type:
                conv_module = getattr(torch_geometric.nn, conv_type)
                self.conv = Sequential('x, edge_index', [
                    (conv_module(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                    BatchNorm(conv_hidden[0] * 2),
                    nn.ReLU(inplace=True),
                ])
                self.conv_mean = Sequential('x, edge_index', [
                    (conv_module(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                ])
                self.conv_logvar = Sequential('x, edge_index', [
                    (conv_module(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                ])
                
        self.dc = InnerProductDecoder(p_drop)
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.linear_encoder_hidden[-1] + self.conv_hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encoder1(self, x):
        for i, module in enumerate(self.encoder):
            for j, l1_module in enumerate(module):
                if j == 1:
                    x, _ = l1_module(x, x, x)
                else:
                    x = l1_module(x)
        return x
    
    def encode(self, x, adj):
        feat_x = self.encoder1(x)
        conv_x = self.conv(feat_x, adj)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj), feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def target_distribution(self, target):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def AVGN_loss(self, decoded, x, preds, labels, mu, logvar, n_nodes, norm, mask=None, mse_weight=10, bce_kld_weight=0.1):
        mse_fun = torch.nn.MSELoss()
        mse_loss = mse_fun(decoded, x)

        if mask is not None:
            preds = preds * mask
            labels = labels * mask

        bce_logits_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)

        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return mse_weight * mse_loss + bce_kld_weight * (bce_logits_loss + KLD)

    def decoder(self, z):
        d = self.linear3(z)
        d = self.batchnorm3(d)
        d = self.activation(d)
        d = self.dropout(d)
        de_feat = self.linear4(d)
        return de_feat

    def forward(self, x, adj):
        mu, logvar, feat_x = self.encode(x, adj)

        gnn_z = self.reparameterize(mu, logvar)

        z = torch.cat((feat_x, gnn_z), 1)

        de_feat = self.decoder(z)

        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z
    
def MHA_block(net,in_features,out_features):
    net.append(nn.Linear(in_features, out_features))
    net.append(nn.MultiheadAttention(out_features, num_heads=4))
    return net

def buildNetwork(in_features, out_features, activate="relu", p_drop=0.0):
    net = []
    net = MHA_block(net,in_features,out_features)
    net.append(nn.Linear(out_features, out_features))
    net.append(BatchNorm(out_features, momentum=0.01, eps=0.001))

    if activate == "relu":
        net.append(nn.ELU())
    elif activate == "sigmoid":
        net.append(nn.Sigmoid())
    
    if p_drop > 0:
        net.append(nn.Dropout(p_drop))
    return nn.Sequential(*net) 

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t())) 
        return adj 

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x) * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -1 * ctx.weight), None

class AdversarialNetwork(nn.Module):
    def __init__(self, model, n_domains=2, weight=1, n_layers=2):
        super(AdversarialNetwork, self).__init__()
        self.model = model
        self.n_domains = n_domains
        self.n_layers = n_layers
        self.weight = weight

        hidden_layers = [
            nn.Linear(self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1], 
                        self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1]),
            nn.ReLU(),
        ] * n_layers

        self.domain_clf = nn.Sequential(
            *hidden_layers,
            nn.Linear(in_features=(self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1]), out_features=n_domains),
        )

    def set_rev_grad_weight(self, weight):
        self.weight = weight

    def target_distribution(self, target):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def AVGN_loss(self, decoded, x, preds, labels, mu, logvar, n_nodes, norm, mask=None, mse_weight=10, bce_kld_weight=0.1):
        mse_fun = torch.nn.MSELoss()
        mse_loss = mse_fun(decoded, x)

        if mask is not None:
            preds = preds * mask
            labels = labels * mask

        bce_logits_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)

        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return mse_weight * mse_loss + bce_kld_weight * (bce_logits_loss + KLD)

    def forward(self, x, edge_index):
        z, mu, logvar, de_feat, q, feat_x, gnn_z = self.model(x, edge_index)
        x_rev = GradientReverseLayer.apply(z, self.weight)
        domain_pred = self.domain_clf(x_rev)
        return z, mu, logvar, de_feat, q, feat_x, gnn_z, domain_pred
