import os
import sys
import pickle

import pandas as pd
import numpy as np
import torch
import dgl

class LinkPredictionDataset(object):
    def __init__(self, root: str, feat_name: str, edge_split_type: str, verbose: bool=True, device: str='cpu'):
        """
        Args:
            root (str): root directory to store the dataset folder.
            feat_name (str): the name of the node features, e.g., "t5vit".
            edge_split_type (str): the type of edge split, can be "random" or "hard".
            verbose (bool): whether to print the information.
            device (str): device to use.
        """
        root = os.path.normpath(root)
        self.name = os.path.basename(root)
        self.verbose = verbose
        self.root = root
        self.feat_name = feat_name
        self.edge_split_type = edge_split_type
        self.device = device
        if self.verbose:
            print(f"Dataset name: {self.name}")
            print(f'Feature name: {self.feat_name}')
            print(f'Edge split type: {self.edge_split_type}')
            print(f'Device: {self.device}')
        
        edge_split_path = os.path.join(root, f'lp-edge-split.pt')
        self.edge_split = torch.load(edge_split_path, map_location=self.device)
        feat_path = os.path.join(root, f'{self.feat_name}_feat.pt')
        feat = torch.load(feat_path, map_location='cpu')
        self.num_nodes = feat.shape[0]
        self.graph = dgl.graph((
            self.edge_split['train']['source_node'],
            self.edge_split['train']['target_node'],
        ), num_nodes=self.num_nodes).to('cpu')
        self.graph.ndata['feat'] = feat

    def get_edge_split(self):
        return self.edge_split

    def __getitem__(self, idx: int):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph

    def __len__(self):
        return 1
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
    

# borrowed from OGB
class LinkPredictionEvaluator(object):
    def __init__(self):
        """
        Calculate MRR, H@1, H@3, and H@10 
        """
        return
    
    def _parse_and_check_input(self, input_dict):
        if not 'y_pred_pos' in input_dict:
            raise RuntimeError('Missing key of y_pred_pos')
        if not 'y_pred_neg' in input_dict:
            raise RuntimeError('Missing key of y_pred_neg')

        y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

        '''
            y_pred_pos: numpy ndarray or torch tensor of shape (num_edges, )
            y_pred_neg: numpy ndarray or torch tensor of shape (num_edges, num_nodes_negative)
        '''

        # convert y_pred_pos, y_pred_neg into either torch tensor or both numpy array
        # type_info stores information whether torch or numpy is used

        type_info = None

        # check the raw tyep of y_pred_pos
        if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
            raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

        # check the raw type of y_pred_neg
        if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
            raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

        # if either y_pred_pos or y_pred_neg is torch tensor, use torch tensor
        if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
            # converting to torch.Tensor to numpy on cpu
            if isinstance(y_pred_pos, np.ndarray):
                y_pred_pos = torch.from_numpy(y_pred_pos)

            if isinstance(y_pred_neg, np.ndarray):
                y_pred_neg = torch.from_numpy(y_pred_neg)

            # put both y_pred_pos and y_pred_neg on the same device
            y_pred_pos = y_pred_pos.to(y_pred_neg.device)

            type_info = 'torch'


        else:
            # both y_pred_pos and y_pred_neg are numpy ndarray

            type_info = 'numpy'


        if not y_pred_pos.ndim == 1:
            raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

        if not y_pred_neg.ndim == 2:
            raise RuntimeError('y_pred_neg must to 2-dim arrray, {}-dim array given'.format(y_pred_neg.ndim))

        return y_pred_pos, y_pred_neg, type_info

    def eval(self, input_dict):
        y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
        return self._eval_mrr(y_pred_pos, y_pred_neg, type_info)

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator\n'
        desc += '{\'y_pred_pos\': y_pred_pos, \'y_pred_neg\': y_pred_neg}\n'
        desc += '- y_pred_pos: numpy ndarray or torch tensor of shape (num_edges, ). Torch tensor on GPU is recommended for efficiency.\n'
        desc += '- y_pred_neg: numpy ndarray or torch tensor of shape (num_edges, num_nodes_neg). Torch tensor on GPU is recommended for efficiency.\n'
        desc += 'y_pred_pos is the predicted scores for positive edges.\n'
        desc += 'y_pred_neg is the predicted scores for negative edges. It needs to be a 2d matrix.\n'
        desc += 'y_pred_pos[i] is ranked among y_pred_neg[i].\n'
        desc += 'Note: As the evaluation metric is ranking-based, the predicted scores need to be different for different edges.'

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator\n'
        desc += '{' + '\'hits@1_list\': hits@1_list, \'hits@3_list\': hits@3_list, \n\'hits@10_list\': hits@10_list, \'mrr_list\': mrr_list}\n'
        desc += '- mrr_list (list of float): list of scores for calculating MRR \n'
        desc += '- hits@1_list (list of float): list of scores for calculating Hits@1 \n'
        desc += '- hits@3_list (list of float): list of scores to calculating Hits@3\n'
        desc += '- hits@10_list (list of float): list of scores to calculating Hits@10\n'
        desc += 'Note: i-th element corresponds to the prediction score for the i-th edge.\n'
        desc += 'Note: To obtain the final score, you need to concatenate the lists of scores and take average over the concatenated list.'

        return desc

    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        '''
            compute mrr
            y_pred_neg is an array with shape (batch size, num_entities_neg).
            y_pred_pos is an array with shape (batch size, )
        '''

        if type_info == 'torch':
            # calculate ranks
            y_pred_pos = y_pred_pos.view(-1, 1)
            # optimistic rank: "how many negatives have a larger score than the positive?"
            # ~> the positive is ranked first among those with equal score
            optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
            # pessimistic rank: "how many negatives have at least the positive score?"
            # ~> the positive is ranked last among those with equal score
            pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            hits1_list = (ranking_list <= 1).to(torch.float)
            hits3_list = (ranking_list <= 3).to(torch.float)
            hits10_list = (ranking_list <= 10).to(torch.float)
            mrr_list = 1./ranking_list.to(torch.float)

            return {'hits@1_list': hits1_list,
                     'hits@3_list': hits3_list,
                     'hits@10_list': hits10_list,
                     'mrr_list': mrr_list}

        else:
            y_pred_pos = y_pred_pos.reshape(-1, 1)
            optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
            pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            hits1_list = (ranking_list <= 1).astype(np.float32)
            hits3_list = (ranking_list <= 3).astype(np.float32)
            hits10_list = (ranking_list <= 10).astype(np.float32)
            mrr_list = 1./ranking_list.astype(np.float32)

            return {'hits@1_list': hits1_list,
                     'hits@3_list': hits3_list,
                     'hits@10_list': hits10_list,
                     'mrr_list': mrr_list}
