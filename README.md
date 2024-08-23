# mm-graph-benchmark

This is the official repository of paper Multimodal Graph Benchmark.

# Installation

```bash
# Clone the repo
git clone https://github.com/mm-graph-benchmark/mm-graph-benchmark.git
cd mm-graph-benchmark

# Create the environment
conda create -n mm_bench python=3.10
conda activate mm_bench

# Install PyTorch and DGL. 
# Here we assume that the CUDA version is 11.8. You may need to modify this based on your CUDA version. 
# For more information, visit https://pytorch.org/get-started/previous-versions/ and https://www.dgl.ai/pages/start.html
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam/label/th21_cu118 dgl

# Install other dependencies
pip install pandas numpy scikit-learn
```

# Data Preparation

Download our datasets from [this link](https://huggingface.co/datasets/mm-graph-org/mm-graph). You may save them to any directory you like, such as `./Multimodal-Graph-Completed-Graph`. The structure should look like the following tree diagram. You can easily add new datasets following this format.

```bash
.
├── books-lp
│   ├── lp-edge-split-random.pt
│   ├── clip_feat.pt
│   ├── imagebind_feat.pt
│   ├── t5vit_feat.pt
│   └── t5dino_feat.pt
├── sports-copurchase
│   ├── lp-edge-split-hard.pt
│   ├── clip_feat.pt
│   ├── imagebind_feat.pt
│   ├── t5vit_feat.pt
│   └── t5dino_feat.pt
├── cloth-copurchase
│   ├── lp-edge-split-hard.pt
│   ├── clip_feat.pt
│   ├── imagebind_feat.pt
│   ├── t5vit_feat.pt
│   └── t5dino_feat.pt
├── ele-fashion
│   ├── nc_edges-nodeid.pt
│   ├── split.pt
│   ├── labels-w-missing.pt
│   ├── clip_feat.pt
│   ├── imagebind_feat.pt
│   ├── t5vit_feat.pt
│   └── t5dino_feat.pt
└── books-nc
    ├── nc_edges-nodeid.pt
    ├── split.pt
    ├── labels-w-missing.pt
    ├── clip_feat.pt
    ├── imagebind_feat.pt
    ├── t5vit_feat.pt
    └── t5dino_feat.pt
```

# Examples

## NodeClassificationDataset

```python
import os
from nc_dataset import NodeClassificationDataset, NodeClassificationEvaluator

data_path = './Multimodal-Graph-Completed-Graph' # replace this with the path where you save the datasets
dataset_name = 'books-nc'
feat_name = 't5vit'
verbose = True
device = 'cpu' # use 'cuda' if GPU is available

dataset = NodeClassificationDataset(
	root=os.path.join(data_path, dataset_name),
	feat_name=feat_name,
	verbose=verbose,
	device=device
)

graph = dataset.graph
# type(graph) would be dgl.DGLGraph
# use graph.ndata['feat'] to get the features
# use graph.ndata['label'] to get the labels (i.e., classes)
# use graph.ndata['train_mask'], graph.ndata['val_mask'], and graph.ndata['test_mask'] to get the corresponding masks

#########################

eval_metric = 'rocauc' # 'acc' is also supported
evaluator = NodeClassificationEvaluator(eval_metric=eval_metric)
# use evaluator.expected_input_format and evaluator.expected_output_format to see the details about the format

input_dict = {'y_true': ..., 'y_pred': ...} # get input_dict using the model you trained
result = evaluator.eval(input_dict=input_dict)
```

## LinkPredictionDataset

```python
import os
from lp_dataset import LinkPredictionDataset, LinkPredictionEvaluator

data_path = './Multimodal-Graph-Completed-Graph' # replace this with the path where you save the datasets
dataset_name = 'sports-copurchase'
feat_name = 't5vit'
edge_split_type = 'hard'
verbose = True
device = 'cpu' # use 'cuda' if GPU is available

dataset = LinkPredictionDataset(
	root=os.path.join(data_path, dataset_name),
	feat_name=feat_name,
	edge_split_type=edge_split_type,
	verbose=verbose,
	device=device
)

graph = dataset.graph
# type(graph) would be dgl.DGLGraph
# use graph.ndata['feat'] to get the features

edge_split = dataset.get_edge_split()
# edge_split = {
#     'train': {
#         'source_node': ...,
#         'target_node': ...,
#     },
#     'valid': {
#         'source_node': ...,
#         'target_node': ...,
#         'target_node_neg': ...,
#     }
#     'test': {
#         'source_node': ...,
#         'target_node': ...,
#         'target_node_neg': ...,
#     }
# }

#########################

evaluator = LinkPredictionEvaluator()
# these metrics will be automatically calculated: MRR, Hits@1, Hits@3, and Hits@10
# use evaluator.expected_input_format and evaluator.expected_output_format to see the details about the format

input_dict = {'y_pred_pos': ..., 'y_pred_neg': ...} # get input_dict using the model you trained
result = evaluator.eval(input_dict=input_dict)
```
#########################
## Raw Images

Raw images can be downloaded by using `node_mapping.pt` (which provides 1-1 mapping for node id and raw file id) for each dataset. A reference code for downloading can be found in `download_img.py`.
The products metadata can be obtained from: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews and https://mengtingwan.github.io/data/goodreads.html. The zipped image folder can be found here. 

## Raw text

Raw images can be downloaded by using `node_mapping.pt` (which provides 1-1 mapping for node id and raw file id) for each dataset. 
The products metadata can be obtained from: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews and https://mengtingwan.github.io/data/goodreads.html. The zipped text folder can be found here. 
