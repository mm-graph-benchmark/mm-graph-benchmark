import torch
from utils import Logger, to_bidirected_with_reverse_mapping, load_usair_dataset, load_esci_dataset, \
    remove_collab_dissimilar_edges
import torch.nn.functional as F
import dgl
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler, as_edge_prediction_sampler, \
    negative_sampler
import tqdm
from models import SAGE, GCN, MLP, Dot, GATv2
import os
import numpy as np
from sklearn import metrics
import pickle
import pdb
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import time
import optuna
from lp_dataset import LinkPredictionDataset, LinkPredictionEvaluator

PROJETC_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(PROJETC_DIR, '../')
CONFIG_DIR = os.path.join(PROJETC_DIR, "configs")
log = logging.getLogger(__name__)

def compute_mrr_esci(
        model, 
        node_emb, 
        src, 
        dst, 
        neg_dst, 
        device, 
        batch_size=500, 
        preload_node_emb=True,
        use_concat = False, 
        use_dot = False
    ):
    """Compute Mean Reciprocal Rank (MRR) in batches in esci dataset."""

    # gpu may be out of memory for large datasets
    if preload_node_emb:
        node_emb = node_emb.to(device)

    rr = torch.zeros(src.shape[0])
    hits_at_10 = torch.zeros(src.shape[0])
    hits_at_1 = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc='Evaluate'):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        if use_concat:
            h_src = h_src.repeat(1, 1001, 1)
            pred = model.predictor(torch.cat((h_src, h_dst), dim=2)).squeeze(-1)
        elif use_dot:
            pred = model.decoder(h_src * h_dst).squeeze(-1)
        else:
            pred = model.predictor(h_src * h_dst).squeeze(-1)
        #import pdb; pdb.set_trace()
        y_pred_pos = pred[:, 0]
        y_pred_neg = pred[:, 1:]
        y_pred_pos = y_pred_pos.view(-1, 1)
        # optimistic rank: "how many negatives have at least the positive score?"
        # ~> the positive is ranked first among those with equal score
        optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        # pessimistic rank: "how many negatives have a larger score than the positive?"
        # ~> the positive is ranked last among those with equal score
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hits_at_10[start:end] = ranking_list<=10
        hits_at_1[start:end] = ranking_list<=1
        mrr_list = 1. / ranking_list.to(torch.float)
        rr[start:end] = mrr_list
    MRR = rr.mean()
    Hits_10 = hits_at_10.sum()/src.shape[0]
    Hits_1 = hits_at_1.sum()/src.shape[0]

    return MRR, Hits_10, Hits_1




def train(cfg, device, g, reverse_eids, seed_edges, model, edge_split, logger, run, eval_batch_size=1000):
    # create sampler & dataloader
    total_it = 1000 * 512 / cfg.batch_size
    if not os.path.exists(cfg.checkpoint_folder):
        os.makedirs(cfg.checkpoint_folder)
    checkpoint_path = cfg.checkpoint_folder + cfg.model_name + "_" + cfg.dataset + "_" + "batch_size_" + str(
        cfg.batch_size) + "_n_layers_" + str(cfg.num_layers) + "_hidden_dim_" + str(cfg.hidden_dim) + "_lr_" + str(
        cfg.lr) + "_exclude_degree_" + str(cfg.exclude_target_degree) + "_full_neighbor_" + str(
        cfg.full_neighbor) + "_accu_num_" + str(cfg.accum_iter_number) + "_trail_" + str(run) + "_best.pth"
    if cfg.full_neighbor:
        log.info("We use the full neighbor of the target node to train the models. ")
        sampler = MultiLayerFullNeighborSampler(num_layers=cfg.num_layers, prefetch_node_feats=['feat'])
    else:
        log.info("We sample the neighbor node of the target node to train the models. ")
        sampler = NeighborSampler([cfg.num_of_neighbors] * cfg.num_layers, prefetch_node_feats=['feat'])
    log.info("We exclude the training target. ")
    sampler = as_edge_prediction_sampler(
        sampler, exclude="reverse_id", reverse_eids=reverse_eids, negative_sampler=negative_sampler.Uniform(1))
    use_uva = (cfg.mode == 'mixed')
    dataloader = DataLoader(
        g, seed_edges, sampler,
        device=device, batch_size=cfg.batch_size, shuffle=True,
        drop_last=False, num_workers=0, use_uva=use_uva)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt, 
        step_size=cfg.lr_scheduler_step_size,
        gamma=cfg.lr_scheduler_gamma,
    )
    optuna_acc = 0
    for epoch in range(cfg.n_epochs):
        model.train()
        total_loss = 0
        # batch accumulation parameter
        accum_iter = cfg.accum_iter_number

        log.info('Training...')
        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader):
            # pair_graph: all positive edge pairs in this batch, stored  as a graph
            # neg_pair_graph: all negative edge pairs in this batch, stored as a graph
            # blocks: each block is the aggregated graph as input for each layer
            x = blocks[0].srcdata['feat'].float()
            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
            score = torch.cat([pos_score, neg_score])
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            (loss / accum_iter).backward()
            if ((it + 1) % accum_iter == 0) or (it + 1 == len(dataloader)) or (it + 1 == total_it):
                # Update Optimizer
                opt.step()
                opt.zero_grad()
            total_loss += loss.item()
            if (it + 1) == total_it:
                break

        lr_scheduler.step()

        log.info("Epoch {:05d} | Loss {:.4f}".format(epoch, total_loss / (it + 1)))
        if (epoch + 1) % cfg.log_steps == 0:
            model.eval()
            log.info('Validation/Testing...')
            with torch.no_grad():
                node_emb = model.inference(g, device, eval_batch_size)
                results = []

                log.info("do evaluation on training examples: check if can be overfitted")
                torch.manual_seed(12345)
                num_sampled_nodes = edge_split['valid']['target_node_neg'].size(dim=0)
                idx = torch.randperm(edge_split['train']['source_node'].numel())[:num_sampled_nodes]
                edge_split['eval_train'] = {
                    'source_node': edge_split['train']['source_node'][idx],
                    'target_node': edge_split['train']['target_node'][idx],
                    'target_node_neg': edge_split['valid']['target_node_neg'],
                }

                src = edge_split['eval_train']['source_node'].to(node_emb.device)
                dst = edge_split['eval_train']['target_node'].to(node_emb.device)
                neg_dst = edge_split['eval_train']['target_node_neg'].to(node_emb.device)

                use_concat = cfg.use_concat
                use_dot = cfg.model_name == "Dot"
                mrr, hits_at_10, hits_at_1 = compute_mrr_esci(model, node_emb, src, dst, neg_dst, device, preload_node_emb=cfg.preload_node_emb, use_concat=use_concat, use_dot=use_dot)
                log.info('Train MRR {:.4f} '.format(mrr.item()))
                if cfg.no_eval is False:
                    valid_mrr = []
                    valid_h_10 = []
                    valid_h_1 = []
                    test_mrr = []
                    test_h_10 = []
                    test_h_1 = []
                    for split in ['valid', 'test']:
                        if cfg.dataset == "ogbl-citation2":
                            evaluator = Evaluator(name=cfg.dataset)
                            src = edge_split[split]['source_node'].to(node_emb.device)
                            dst = edge_split[split]['target_node'].to(node_emb.device)
                            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)
                            results.append(compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device))
                        else:
                            src = edge_split[split]['source_node'].to(node_emb.device)
                            dst = edge_split[split]['target_node'].to(node_emb.device)
                            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)
                            results.append(
                                compute_mrr_esci(model, node_emb, src, dst, neg_dst, device, preload_node_emb=cfg.preload_node_emb, use_concat=use_concat, use_dot=use_dot)
                            )
                    valid_mrr.append(results[0][0].item())
                    valid_h_10.append(results[0][1].item())
                    valid_h_1.append(results[0][2].item())
                    test_mrr.append(results[1][0].item())
                    test_h_10.append(results[1][1].item())
                    test_h_1.append(results[1][2].item())

                    # save best checkpoint
                    valid_result, test_result = results[0][0].item(), results[1][0].item()
                    
                    # we want to find the best previous checkpoint
                    # if there is no previous checkpoint, set it to 0
                    # Warning: it only works for MRR and Hit@N.
                    if len(logger.results[run]) > 0:
                        previous_best_valid_result = torch.tensor(logger.results[run])[:, 0].max().item()
                    else:  # length = 0
                        previous_best_valid_result = 0.0
                    
                    if valid_result > previous_best_valid_result:
                        log.info("Saving checkpoint. ")
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                        }, checkpoint_path)
                        optuna_acc = test_result

                    logger.add_result(run, [valid_result, test_result])
                    log.info('Validation MRR {:.4f}, Test MRR {:.4f}'.format(valid_result, test_result))
                    log.info('Validation Hits@10 {:.4f}, Test Hits@10 {:.4f}'.format(results[0][1].item(), results[1][1].item()))
                    log.info('Validation Hits@1 {:.4f}, Test Hits@1 {:.4f}'.format(results[0][2].item(), results[1][2].item()))
    logger.print_statistics(run)
    return results[1][0].item(), results[1][1].item(), results[1][2].item()


@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base='1.2')
def main(cfg: DictConfig):
    log.info('Loading data')
    data_path = '/nfs/turbo/coe-dkoutra/jing/Multimodal-Graph-Completed-Graph' # replace this with the path where you save the datasets
    dataset_name = 'sports-copurchase'
    feat_name = 't5vit'
    edge_split_type = 'hard'
    verbose = True
    device = ('cuda' if cfg.mode == 'puregpu' else 'cpu') # use 'cuda' if GPU is available

    dataset = LinkPredictionDataset(
        root=os.path.join(data_path, dataset_name),
        feat_name=feat_name,
        edge_split_type=edge_split_type,
        verbose=verbose,
        device=device
    )

    g = dataset.graph
    # type(graph) would be dgl.DGLGraph
    # use graph.ndata['feat'] to get the features

    edge_split = dataset.get_edge_split()
    g = dgl.remove_self_loop(g)
    log.info("remove isolated nodes")
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    g = g.to('cuda' if cfg.mode == 'puregpu' else 'cpu')
    num_nodes = g.number_of_nodes()
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)
    
    in_size = g.ndata['feat'].shape[1]
    logger = Logger(cfg.runs)

    mrrs = []
    h1s = []
    h10s = []
    for run in range(cfg.runs):
        log.info("Run {}/{}".format(run + 1, cfg.runs))
        if cfg.model_name == "SAGE":
            model = SAGE(in_size, cfg.hidden_dim, cfg.num_layers).to(device)
        elif cfg.model_name == "GCN":
            model = GCN(in_size, cfg.hidden_dim, cfg.num_layers).to(device)
            if cfg.add_self_loop:
                g = dgl.add_self_loop(g)
        elif cfg.model_name == "MLP":
            model = MLP(in_size, cfg.hidden_dim, cfg.num_layers).to(device)
        elif cfg.model_name == "Dot":
            model = Dot(in_size, cfg.hidden_dim, cfg.num_layers).to(device)
        elif cfg.model_name == "GAT":
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            if cfg.dataset == "ogbl-citation2":
                num_heads = 8
            else:
                num_heads = 8
            num_out_heads = 1
            heads = ([num_heads] * (cfg.num_layers - 1)) + [num_out_heads]
            activation = F.elu
            feat_drop = 0
            attn_drop = 0
            negative_slope = 0.2
            residual = True
            model = GATv2(in_size, cfg.hidden_dim, cfg.num_layers, heads, activation, feat_drop, attn_drop,
                        negative_slope, residual).to(device)
        elif cfg.model_name == "GIN":
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            print("Add self loop")
            model = GIN(in_size, cfg.hidden_dim, cfg.num_layers).to(device)
        else:
            raise ValueError(f"Model '{cfg.model_name}' is not supported")
        # model training
        log.info('Training...')
        # log.info(edge_split['test'].keys())
        mrr, h10, h1 = train(cfg, device, g, reverse_eids, seed_edges, model, edge_split, logger, run)
        mrrs.append(mrr)
        h10s.append(h10)
        h1s.append(h1)
    logger.print_statistics()
    mean_mrr = torch.mean(torch.tensor(mrrs)).item()
    std_mrr = torch.std(torch.tensor(mrrs)).item()
    mean_h1 = torch.mean(torch.tensor(h1s)).item()
    std_h1 = torch.std(torch.tensor(h1s)).item()
    mean_h10 = torch.mean(torch.tensor(h10s)).item()
    std_h10 = torch.std(torch.tensor(h10s)).item()
    print(mean_mrr)
    print(std_mrr)
    print(mean_h1)
    print(std_h1)
    print(mean_h10)
    print(std_h10)    
    return mean_mrr


if __name__=='__main__':
    main()