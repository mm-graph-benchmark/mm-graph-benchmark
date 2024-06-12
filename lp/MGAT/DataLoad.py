import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class DataLoad(Dataset):
	def __init__(self, train_edge, num_nodes):
		super(DataLoad, self).__init__()

		src_tgt_dict = {i: [] for i in range(num_nodes)}
		for pair in tqdm(train_edge, total=train_edge.shape[0]):
			src, tgt = pair[0].item(), pair[1].item()
			if src in src_tgt_dict.keys():
				src_tgt_dict[src].append(tgt)
			else:
				src_tgt_dict[src] = [tgt]

		self.data = train_edge
		self.adj_lists = src_tgt_dict
		self.num_nodes = num_nodes

	def __getitem__(self, index):
		user, pos_item = self.data[index]
		user, pos_item = user.item(), pos_item.item()
  
		while True:
			neg_tgt = random.randint(0, self.num_nodes-1)
			if neg_tgt not in self.adj_lists[user]:
				break
		return [user, pos_item, neg_tgt]
				

	def __len__(self):
		return len(self.data)


