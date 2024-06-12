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
    def __init__(self, in_size, hid_size, out_size, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, 'mean'))
        elif num_layers == 2:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        elif num_layers == 3:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y



class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(dglnn.GraphConv(in_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
        elif num_layers == 2:
            self.layers.append(dglnn.GraphConv(in_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(hid_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
        elif num_layers == 3:
            self.layers.append(dglnn.GraphConv(in_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(hid_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(hid_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y




class MLP(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_size, out_size))
        elif num_layers == 2:
            self.layers.append(nn.Linear(in_size, hid_size))
            self.layers.append(nn.Linear(hid_size, out_size))
        elif num_layers == 3:
            self.layers.append(nn.Linear(in_size, hid_size))
            self.layers.append(nn.Linear(hid_size, hid_size))
            self.layers.append(nn.Linear(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[output_nodes]
                h = layer(x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y



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


class MMGCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, num_nodes, dropout=0.5):
        super().__init__()

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_nodes, hid_size), requires_grad=True)).cuda()
        self.result = nn.init.xavier_normal_(torch.rand((num_nodes, hid_size))).cuda()


        self.v_gcn = nn.ModuleList()
        if num_layers == 1:
            self.v_gcn.append(dglnn.GraphConv(in_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
        elif num_layers == 2:
            self.v_gcn.append(dglnn.GraphConv(in_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
            self.v_gcn.append(dglnn.GraphConv(hid_size+hid_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
        elif num_layers == 3:
            self.v_gcn.append(dglnn.GraphConv(in_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
            self.v_gcn.append(dglnn.GraphConv(hid_size+hid_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
            self.v_gcn.append(dglnn.GraphConv(hid_size+hid_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))

        self.t_gcn = nn.ModuleList()
        if num_layers == 1:
            self.t_gcn.append(dglnn.GraphConv(in_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
        elif num_layers == 2:
            self.t_gcn.append(dglnn.GraphConv(in_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
            self.t_gcn.append(dglnn.GraphConv(hid_size+hid_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
        elif num_layers == 3:
            self.t_gcn.append(dglnn.GraphConv(in_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
            self.t_gcn.append(dglnn.GraphConv(hid_size+hid_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))
            self.t_gcn.append(dglnn.GraphConv(hid_size+hid_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True))

        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        x = F.normalize(x)
        h_v = x
        for l, (layer, block) in enumerate(zip(self.v_gcn, blocks)):
            h_temp = layer(block, h_v)
            x_hat =  layer(block, h_v) + self.id_embedding[block.dstdata['_ID']]
            h_v = torch.cat((h_v, x_hat), dim=1)
            if l != len(self.layers) - 1:
                h_v =  F.leaky_relu(h_v)

        h_t = x
        for l, (layer, block) in enumerate(zip(self.t_gcn, blocks)):
            x_hat =  layer(block, h_t) + self.id_embedding[block.dstdata['_ID']]
            h_t = torch.cat((h_t, x_hat), dim=1)
            if l != len(self.layers) - 1:
                h_t =  F.leaky_relu(h_t)

        h = (h_v + h_t)/2
        breakpoint()
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.v_gcn):
            y_v = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.v_gcn) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = g.ndata["feat"].to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h_v = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h_v = F.leaky_relu(h_v) + self.id_embedding
                # by design, our output nodes are contiguous
                y_v[output_nodes[0] : output_nodes[-1] + 1] = h_v.to(buffer_device)
            feat = y_v

        for l, layer in enumerate(self.t_gcn):
            y_t = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.t_gcn) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = g.ndata["feat"].to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                breakpoint()
                h_t = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h_t = F.leaky_relu(h_t) + + self.id_embedding
                # by design, our output nodes are contiguous
                y_t[output_nodes[0] : output_nodes[-1] + 1] = h_t.to(buffer_device)
            feat = y_t

        y = (y_v + y_t)/2
        breakpoint()
        return y

