import time
import pickle
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def data_load(train_edge, feat_path):
    
    v_t_feat = torch.load(feat_path, map_location='cpu')
    t_feat = v_t_feat[:, :int(v_t_feat.shape[1] / 2)]
    v_feat = v_t_feat[:, int(v_t_feat.shape[1] / 2):]
    a_feat = None
    
    num_nodes = v_feat.shape[0]
    
    src_tgt_dict = {i: [] for i in range(num_nodes)}
    for pair in tqdm(train_edge, total=train_edge.shape[0]):
        src, tgt = pair[0].item(), pair[1].item()
        if src in src_tgt_dict.keys():
            src_tgt_dict[src].append(tgt)
        else:
            src_tgt_dict[src] = [tgt]
            
    
    return num_nodes, train_edge, src_tgt_dict, v_feat, a_feat, t_feat

def data_load_old(train_edge, v_feat_path, t_feat_path):
    
    v_feat = torch.load(v_feat_path).to('cuda')
    t_feat = torch.load(t_feat_path).to('cuda')
    a_feat = None
    
    num_nodes = v_feat.shape[0]
    
    src_tgt_dict = {i: [] for i in range(num_nodes)}
    for pair in tqdm(train_edge, total=train_edge.shape[0]):
        src, tgt = pair[0].item(), pair[1].item()
        if src in src_tgt_dict.keys():
            src_tgt_dict[src].append(tgt)
        else:
            src_tgt_dict[src] = [tgt]
            
    
    return num_nodes, train_edge, src_tgt_dict, v_feat, a_feat, t_feat


class TrainingDataset(Dataset):
    def __init__(self, num_nodes, train_nodes, labels):
        self.num_nodes = num_nodes
        self.train_nodes = torch.LongTensor(train_nodes)
        self.labels = labels

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, index):
        src = self.train_nodes[index].item()
        label = self.labels[src].item()
        return torch.LongTensor([src]), torch.LongTensor([label])
