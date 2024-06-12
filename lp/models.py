import torch
import torch.nn.functional as F
import dgl.nn as dglnn
import tqdm
import torch.nn as nn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl.nn import GATv2Conv
from dgl.nn.pytorch.conv import GINConv
from torch.nn import Linear

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        # n-layer GraphSAGE-mean
        for i in range(num_layers - 1):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.layers):
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
        # n-layer GraphConv
        for i in range(num_layers - 1):
            self.layers.append(
                dglnn.GraphConv(hid_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.layers):
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class MLP(nn.Module):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__()
        self.hid_size = hid_size
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hid_size))
        # n-layer GraphSAGE-mean
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hid_size, hid_size))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.LayerNorm(hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.LayerNorm(hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

        h_pos = self.predictor(torch.cat((h[pos_src], h[pos_dst]), dim=1))
        h_neg = self.predictor(torch.cat((h[neg_src], h[neg_dst]), dim=1))
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float().to(device)
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.layers):
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[output_nodes]
                h = layer(x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class Dot(nn.Module):
    ### no encoder because the finetuned feature should be robust enough
    ### directly decoder using an elementwise product + 3 layer MLP
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(in_size, hid_size),
        #     nn.LayerNorm(hid_size),
        #     nn.ReLU(),
        #     nn.Linear(hid_size, hid_size))

        # self.decoder = nn.Linear(hid_size, 1)
        self.decoder = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.LayerNorm(hid_size),
            nn.ReLU(),
            nn.Linear(in_size, hid_size),
            nn.LayerNorm(hid_size),
            nn.ReLU(),
            nn.Linear(in_size, hid_size),
            nn.LayerNorm(hid_size),
            nn.ReLU(),
            nn.Linear(in_size, hid_size),
            nn.LayerNorm(hid_size),
            nn.ReLU(),
            nn.Linear(in_size, hid_size),
            nn.LayerNorm(hid_size),
            nn.ReLU(),
            nn.Linear(in_size, hid_size),
            nn.LayerNorm(hid_size),
            nn.ReLU(),
            nn.Linear(in_size, hid_size),
            nn.LayerNorm(hid_size),
            nn.ReLU(),
            nn.Linear(in_size, hid_size),
            nn.LayerNorm(hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        # h = self.encoder(x)
        h = x
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        # h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_pos = self.decoder(h[pos_src] * h[pos_dst])
        h_neg = self.decoder(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float().to(device)
        buffer_device = torch.device('cpu')
        # h = self.encoder(feat).to(buffer_device)
        h = feat
        return h



class GATv2(nn.Module):
    def __init__(self, in_size, hid_size, num_layers, heads, activation, feat_drop, attn_drop,
                 negative_slope, residual):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        self.heads = heads
        self.hid_size = hid_size
        self.layer_norms = torch.nn.ModuleList()
        # input projection (no residual)
        self.gatv2_layers.append(
            GATv2Conv(in_size, hid_size, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation,
                      bias=False, share_weights=True)
        )
        # hidden layers
        for l in range(num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layer_norms.append(nn.LayerNorm(hid_size * heads[l]))
            self.gatv2_layers.append(
                GATv2Conv(hid_size * heads[l], hid_size, heads[l + 1], feat_drop, attn_drop, negative_slope, residual,
                          self.activation, bias=False, share_weights=True)
            )
        # output projection
        self.predictor = nn.Sequential(
            nn.Linear(hid_size * heads[-1], hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gatv2_layers, blocks)):
            h = layer(block, h).flatten(1)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.gatv2_layers):
            y = torch.empty(g.num_nodes(), self.hid_size * self.heads[l], device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x).flatten(1)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class GINMLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.hid_size = hid_size
        self.batch_norms = nn.ModuleList()
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = GINMLP(in_size, hid_size, hid_size)
            else:
                mlp = GINMLP(hid_size, hid_size, hid_size)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hid_size))

        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        # list of hidden representation at each layer (including the input layer)
        h = x
        for l, (layer, block) in enumerate(zip(self.ginlayers, blocks)):
            h = layer(block, h)
            h = self.batch_norms[l](h)
            if l != len(self.ginlayers) - 1:
                h = F.relu(h)

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.ginlayers):
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                h = self.batch_norms[l](h)
                if l != len(self.ginlayers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y