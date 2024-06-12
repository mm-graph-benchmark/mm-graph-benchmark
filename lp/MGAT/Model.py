import math
# from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from GraphGAT import GraphGAT

class MGAT(torch.nn.Module):
    def __init__(self, features, edge_index, batch_size, num_nodes, num_layers: int, dim_x=64):
        super(MGAT, self).__init__()
        self.batch_size = batch_size
        self.num_nodes = num_nodes

        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        
        v_feat, a_feat, t_feat = features
        self.v_feat = torch.tensor(v_feat).cuda()
        self.a_feat = None
        self.t_feat = torch.tensor(t_feat).cuda()

        self.v_gnn = GNN(self.v_feat, self.edge_index, batch_size, num_nodes, dim_x, num_layers, dim_latent=256)
        self.a_gnn = None
        self.t_gnn = GNN(self.t_feat, self.edge_index, batch_size, num_nodes, dim_x, num_layers, dim_latent=100)

        self.id_embedding = nn.Embedding(num_nodes, dim_x)
        nn.init.xavier_normal_(self.id_embedding.weight)

        #self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).cuda()
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_nodes, dim_x))).cuda()

    def forward(self, user_nodes, pos_items, neg_items):
        v_rep = self.v_gnn(self.id_embedding)
        a_rep = None
        t_rep = self.t_gnn(self.id_embedding)
        if a_rep is None:
            representation = (v_rep + t_rep) / 2
        else:
            representation = (v_rep + a_rep + t_rep) / 3 #torch.max_pool2d((v_rep, a_rep, t_rep))#max()#torch.cat((v_rep, a_rep, t_rep), dim=1)
        self.result_embed = representation
        user_tensor = representation[user_nodes]
        pos_tensor = representation[pos_items]
        neg_tensor = representation[neg_items]
        pos_scores = torch.sum(user_tensor * pos_tensor, dim=1)
        neg_tensor = torch.sum(user_tensor * neg_tensor, dim=1)
        return pos_scores, neg_tensor


    def loss(self, data):
        user, pos_items, neg_items = data
        pos_scores, neg_scores = self.forward(user.cuda(), pos_items.cuda(), neg_items.cuda())
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        return loss_value


class GNN(torch.nn.Module):
    def __init__(self, features, edge_index, batch_size, num_nodes, dim_id, num_layers: int, dim_latent=None):
        super(GNN, self).__init__()
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features
        self.num_layers = num_layers

        if self.dim_latent:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)

            self.conv_embed_1 = GraphGAT(self.dim_latent, self.dim_latent, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 
        else:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            self.conv_embed_1 = GraphGAT(self.dim_feat, self.dim_feat, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight)

        if num_layers >= 2:
            self.conv_embed_2 = GraphGAT(self.dim_id, self.dim_id, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_2.weight)
            self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer2.weight)
            self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer2.weight)
        if num_layers >= 3:
            self.conv_embed_3 = GraphGAT(self.dim_id, self.dim_id, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_3.weight)
            self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer3.weight)
            self.g_layer3 = nn.Linear(self.dim_id, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer3.weight)
            

    def forward(self, id_embedding):
        temp_features = torch.tanh(self.MLP(self.features)) if self.dim_latent else self.features
        x = temp_features
        x = F.normalize(x).cuda()

        #layer-1
        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
        x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)
        
        if self.num_layers == 1:
            x = x_1
            return x_1
        
        # layer-2
        h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
        x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)

        if self.num_layers == 2:
            x = torch.cat((x_1, x_2), dim=1)
            return x
            
        h = F.leaky_relu(self.conv_embed_2(x_2, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer2(x_2)) + id_embedding.weight
        x_3 = F.leaky_relu(self.g_layer3(h)+x_hat)
        
        if self.num_layers == 3:
            x = torch.cat((x_1, x_2, x_3), dim=1)
            return x
        