import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class DataLoad(Dataset):
	def __init__(self, num_nodes, train_nodes, labels):
		super(DataLoad, self).__init__()

		self.num_nodes = num_nodes
		self.train_nodes = torch.LongTensor(train_nodes)
		self.labels = labels

	def __getitem__(self, index):
		src = self.train_nodes[index].item()
		label = self.labels[src].item()
		return torch.LongTensor([src]), torch.LongTensor([label])


	def __len__(self):
		return len(self.train_nodes)



