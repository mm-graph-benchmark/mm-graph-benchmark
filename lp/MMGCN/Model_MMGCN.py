import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from BaseModel import BaseModel

class GCN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_node, dim_feat, dim_id, aggr_mode, concate, has_id, num_layers, dim_latent=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_node = num_node
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.has_id = has_id
        self.num_layers = num_layers

        if self.dim_latent:
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 

        else:
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_feat, self.dim_id)     
            nn.init.xavier_normal_(self.g_layer1.weight)              
          
        if num_layers >= 2:
            self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_2.weight)
            self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer2.weight)
            self.g_layer2 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)

        if num_layers >= 3:
            self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_3.weight)
            self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer3.weight)
            self.g_layer3 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)
        
        if num_layers >= 4:
            self.conv_embed_4 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_4.weight)
            self.linear_layer4 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer4.weight)
            self.g_layer4 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)
        
        if num_layers >= 5:
            self.conv_embed_5 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_5.weight)
            self.linear_layer5 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer5.weight)
            self.g_layer5 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)

    def forward(self, features, id_embedding):
        temp_features = self.MLP(features) if self.dim_latent else features

        x = temp_features
        x = F.normalize(x).cuda()

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer1(x))#equation 5 
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer1(h)+x_hat)

        if self.num_layers >= 2:
            h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer2(x))#equation 5
            x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer2(h)+x_hat)

        if self.num_layers >= 3:
            h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer3(x))#equation 5
            x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer3(h)+x_hat)
        
        if self.num_layers >= 4:
            h = F.leaky_relu(self.conv_embed_4(x, self.edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer4(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer4(x))#equation 6
            x = F.leaky_relu(self.g_layer4(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer4(h)+x_hat)
        
        if self.num_layers >= 5:
            h = F.leaky_relu(self.conv_embed_5(x, self.edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer5(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer5(x))#equation 7
            x = F.leaky_relu(self.g_layer5(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer5(h)+x_hat)

        return x


class Net(torch.nn.Module):
    def __init__(self, v_feat, a_feat, t_feat, words_tensor, edge_index, batch_size, num_node, aggr_mode, concate, num_layers, has_id, src_tgt_dict, reg_weight, dim_x):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.num_node = num_node
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.src_tgt_dict = src_tgt_dict
        self.weight = torch.tensor([[1.0],[-1.0]]).cuda()
        self.reg_weight = reg_weight
        
        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        self.num_modal = 0

        self.v_feat = torch.tensor(v_feat,dtype=torch.float).cuda()
        self.v_gcn = GCN(self.edge_index, batch_size, num_node, self.v_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layers=num_layers, has_id=has_id, dim_latent=256)

        self.a_feat = None
        self.a_gcn = None

        self.t_feat = torch.tensor(t_feat,dtype=torch.float).cuda()
        self.t_gcn = GCN(self.edge_index, batch_size, num_node, self.t_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layers=num_layers, has_id=has_id)

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_node, dim_x), requires_grad=True)).cuda()
        self.result = nn.init.xavier_normal_(torch.rand((num_node, dim_x))).cuda()


    def forward(self):
        v_rep = self.v_gcn(self.v_feat, self.id_embedding)
        a_rep = None

        # # self.t_feat = torch.tensor(scatter_('mean', self.word_embedding(self.words_tensor[1]), self.words_tensor[0])).cuda()
        t_rep = self.t_gcn(self.t_feat, self.id_embedding)
        
        if a_rep is not None:
            representation = (v_rep+a_rep+t_rep)/3
        else:
            representation = (v_rep+t_rep)/2

        self.result = representation
        return representation

    def loss(self, src_tensor, tgt_tensor):
        src_tensor = src_tensor.view(-1)
        tgt_tensor = tgt_tensor.view(-1)
        out = self.forward()
        src_score = out[src_tensor]
        tgt_score = out[tgt_tensor]
        score = torch.sum(src_score*tgt_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        reg_embedding_loss = (self.id_embedding[src_tensor]**2 + self.id_embedding[tgt_tensor]**2).mean()
        reg_loss = self.reg_weight * (reg_embedding_loss)
        return loss+reg_loss, reg_loss, loss, reg_embedding_loss, reg_embedding_loss
