import argparse
import sys
import os
import time
import random
import numpy as np
import torch
from Dataset import TrainingDataset, data_load, data_load_old
from Model_MMGCN import Net
from torch.utils.data import DataLoader, TensorDataset
from Train import train
from Full_vt import full_vt
import wandb
from tqdm import tqdm
import optuna


def get_rank(pred, n=0, descending=True):
    """
    Calculate the rank of the nth element in pred
    descending=True means large values ranks higher,
    descending=False means small values ranks higher.
    """
    arg = torch.argsort(torch.tensor(pred), descending=descending)
    rank = torch.where(arg == n)[0] + 1
    return rank.tolist()[0]


def get_rank_2d(pred, n=0, descending=True):
    """
    Calculate the rank of the nth element in each row of pred.
    `pred` is a 2D tensor where each row contains scores for a single sample.
    `n` specifies the index of the element to find the rank for in each row.
    `descending` controls the order of ranking.
    """
    batch_size, num_elements = pred.shape
    # Create a tensor of the same shape as pred filled with the index n
    n_tensor = torch.full_like(pred, n, dtype=torch.long)

    # Sort pred in the specified order
    sorted_indices = torch.argsort(pred, dim=1, descending=descending)

    # Find the rank (position) of the nth element in each row
    ranks = torch.nonzero(sorted_indices == n_tensor, as_tuple=False)[:, 1] + 1  # Adding 1 to make ranks start from 1

    return ranks


def calculate_mrr(rank_ls):
    """
    Return the MRR (Mean Reciprocal Rank) of a list of ranks.
    """
    if type(rank_ls) is list:
        rk = np.array(rank_ls)
    else:
        rk = rank_ls
    return (1 / rk).mean()


def calculate_hit(rank_ls, n):
    """
    Return the Hit@n of a list of ranks.
    """
    if type(rank_ls) is list:
        rk = np.array(rank_ls)
    else:
        rk = rank_ls
    rk = np.array(rank_ls)
    return rk[rk <= n].shape[0] / rk.shape[0]


def batch_evaluate_model(representation, triple_data, batch_size=128, verbose=False):
    dataset = TensorDataset(
        torch.tensor(triple_data['source_node']),
        torch.tensor(triple_data['target_node']),
        torch.tensor(triple_data['target_node_neg'], dtype=torch.long))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    rank_ls = []
    for src, pos_tgt, neg_tgt_ls in tqdm(dataloader, disable=not verbose):
        src_ls = src.repeat_interleave(neg_tgt_ls.size(1) + 1, dim=0)
        tgt_ls = torch.cat((pos_tgt.unsqueeze(1), neg_tgt_ls), dim=1).view(-1)

        src_score = representation[src_ls]
        tgt_score = representation[tgt_ls]

        pred = torch.sum(src_score * tgt_score, dim=1).detach().cpu()
        rank = get_rank_2d(pred.view(src.size(0), -1))
        rank_ls.extend(rank.numpy().tolist())

    hit_10 = calculate_hit(rank_ls, 10)
    hit_1 = calculate_hit(rank_ls, 1)
    mrr = calculate_mrr(rank_ls)
    return mrr, hit_1, hit_10


def train(dataloader, model, optimizer, batch_size, max_step=None):
    model.train()
    sum_loss = 0.0
    sum_model_loss = 0.0
    sum_reg_loss = 0.0
    sum_ent_loss = 0.0
    sum_weight_loss = 0.0
    step = 0.0
    num_pbar = 0
    for idx, batch in enumerate(tqdm(dataloader)):
        src_tensor, tgt_tensor = batch
        optimizer.zero_grad()
        loss, model_loss, reg_loss, weight_loss, entropy_loss = model.loss(src_tensor, tgt_tensor)
        loss.backward(retain_graph=True)
        optimizer.step()
        num_pbar += batch_size
        sum_loss += loss.cpu().item()
        sum_model_loss += model_loss.cpu().item()
        sum_reg_loss += reg_loss.cpu().item()
        sum_ent_loss += entropy_loss.cpu().item()
        sum_weight_loss += weight_loss.cpu().item()
        step += 1.0
        if max_step is not None and idx > max_step:
            break

    return loss


def hyperparameter_search(train_dataloader, args, v_feat, a_feat, t_feat, train_edge, num_nodes, src_tgt_dict,
                          weight_decay, dim_E, val_triple_data):
    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        num_layers = trial.suggest_int('num_layers', 1, 3)

        model = Net(v_feat, a_feat, t_feat, None, train_edge, args.batch_size, num_nodes, 'mean', 'False', num_layers,
                    True, src_tgt_dict, weight_decay, dim_E).cuda()
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': lr}])

        # loss = train(train_dataloader, model, optimizer, args.batch_size, max_step=args.hyperparameter_search_max_step)
        #
        # with torch.no_grad():
        #     model.eval()
        #     representation = model.forward()
        #     val_mrr, val_hit_1, val_hit_10 = batch_evaluate_model(representation, val_triple_data, verbose=False)
        #
        # return val_mrr
        val_max_mrr = 0
        for epoch in range(args.num_epoch):
            train(train_dataloader, model, optimizer, args.batch_size, max_step=None)
            torch.cuda.empty_cache()

            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    representation = model.forward()
                    val_mrr, val_hit_1, val_hit_10 = batch_evaluate_model(representation, val_triple_data,
                                                                          verbose=False)
                    if val_mrr > val_max_mrr:
                        val_max_mrr = val_mrr
                    print(f"{lr}, {num_layers}")
                    print(f"Epoch: {epoch}, MRR: {val_max_mrr}, Hit@1: {val_hit_1}, Hit@10: {val_hit_10}")
        return val_max_mrr

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    print(study.best_params)

    if args.PATH_best_hyperparameter_save is not None:
        with open(args.PATH_best_hyperparameter_save, 'w') as f:
            f.write(study.best_params)
            f.write('\n')
            f.write(study.best_value)


def main():
    parser = argparse.ArgumentParser()

    # using default values is recommended
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='movielens', help='Dataset path')
    parser.add_argument('--save_file', default='', help='Filename')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')
    parser.add_argument('--aggr_mode', default='add', help='Aggregation Mode.')
    parser.add_argument('--topK', type=int, default=10, help='Workers number.')
    parser.add_argument('--has_entropy_loss', default='False', help='Has Cross Entropy loss.')
    parser.add_argument('--has_weight_loss', default='False', help='Has Weight Loss.')
    parser.add_argument('--has_v', default='True', help='Has Visual Features.')
    parser.add_argument('--has_a', default='True', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='True', help='Has Textual Features.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')

    # your data here
    parser.add_argument('--edge_split_path', required=True)
    parser.add_argument('--v_feat_path', default=None)
    parser.add_argument('--t_feat_path', default=None)
    parser.add_argument('--feat_path', default=None)

    # your setting here
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--repeat_times', type=int, default=3, help='Repeat times.')

    parser.add_argument('--PATH_best_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--PATH_best_metrics_save', default=None)

    # if want to do hyperparameter search
    parser.add_argument('--do_hyperparameter_search', action="store_true")
    parser.add_argument('--PATH_best_hyperparameter_save', default=None)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--hyperparameter_search_max_step', type=int, default=10000)

    parser.add_argument('--project_name', default='MGAT')
    parser.add_argument('--wandb_run_name', default='untitled_run')
    parser.add_argument('--wandb_key', default=None)
    parser.add_argument('--report_to', default=None)

    args = parser.parse_args()

    if args.wandb_key is not None and args.report_to == 'wandb':
        wandb_key = 'ab1e2fd95c62273341f8fe8bbe29b9a6ee33725a'
        wandb.login(key=wandb_key)

    seed = args.seed
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    ##########################################################################################################################################
    data_path = args.data_path
    save_file = args.save_file

    learning_rate = args.l_r
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    topK = args.topK
    prefix = args.prefix
    aggr_mode = args.aggr_mode

    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False

    has_entropy_loss = True if args.has_entropy_loss == 'True' else False
    has_weight_loss = True if args.has_weight_loss == 'True' else False
    dim_E = args.dim_E
    writer = None

    edge_split = torch.load(args.edge_split_path)
    train_edge = torch.concat(
        [edge_split['train']['source_node'].reshape(-1, 1), edge_split['train']['target_node'].reshape(-1, 1)], dim=1)
    train_edge.shape

    ##########################################################################################################################################
    print('Data loading ...')
    if args.feat_path is None:
        num_nodes, train_edge, src_tgt_dict, v_feat, a_feat, t_feat = data_load_old(train_edge, args.v_feat_path,
                                                                                    args.t_feat_path)
    else:
        num_nodes, train_edge, src_tgt_dict, v_feat, a_feat, t_feat = data_load(train_edge, args.feat_path)

    train_dataset = TrainingDataset(num_nodes, src_tgt_dict, train_edge)

    val_triple_data = edge_split['valid']
    test_triple_data = edge_split['test']

    print('Data has been loaded.')

    if args.do_hyperparameter_search:
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
        hyperparameter_search(train_dataloader, args, v_feat, a_feat, t_feat, train_edge, num_nodes, src_tgt_dict,
                              weight_decay, dim_E, val_triple_data)
        return

    num_layers = args.num_layers
    learning_rate = args.l_r
    num_epoch = args.num_epoch
    batch_size = args.batch_size

    global_step = 0
    mrrs, h1s, h10s = [], [], []
    for try_time in range(args.repeat_times):
        model = Net(v_feat, a_feat, t_feat, None, train_edge, batch_size, num_nodes, 'mean', 'False', num_layers, True,
                    src_tgt_dict, weight_decay, dim_E).cuda()
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])

        if args.report_to == 'wandb':
            wandb.init(
                project=args.project_name,
                name=args.wandb_run_name,
                config={
                    "learning_rate": learning_rate,
                    'num_layers': num_layers,
                    "epochs": num_epoch,
                }
            )

        max_mrr = 0.0
        max_hit_1 = 0.0
        max_hit_10 = 0.0
        val_max_mrr = 0.0
        num_decreases = 0
        for epoch in range(num_epoch):
            train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
            loss = train(train_dataloader, model, optimizer, batch_size, max_step=None)
            torch.cuda.empty_cache()

            with torch.no_grad():
                representation = model.forward()
                val_mrr, val_hit_1, val_hit_10 = batch_evaluate_model(representation, val_triple_data, verbose=False)
                test_mrr, test_hit_1, test_hit_10 = batch_evaluate_model(representation, test_triple_data,
                                                                         verbose=False)

            print({
                'loss': loss,
                "val_mrr": val_mrr,
                'val_hit_1': val_hit_1,
                'val_hit_10': val_hit_10,
                'test_mrr': test_mrr,
                'test_hit_1': test_hit_1,
                'test_hit_10': test_hit_10,
            })

            if val_mrr > val_max_mrr:
                val_max_mrr = val_mrr
                max_mrr = test_mrr
                max_hit_1 = test_hit_1
                max_hit_10 = test_hit_10
                num_decreases = 0

                if args.PATH_best_weight_save is not None:
                    torch.save(model.state_dict(), args.PATH_best_weight_save)

                if args.PATH_best_metrics_save is not None:
                    with open(args.PATH_best_metrics_save, 'w') as f:
                        f.write(f"MRR: {test_mrr}")
                        f.write(f"Hit@1: {test_hit_1}")
                        f.write(f"Hit@10: {test_hit_10}")
            global_step += 1

        mrrs.append(test_mrr)
        h1s.append(test_hit_1)
        h10s.append(test_hit_10)

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


if __name__ == '__main__':
    main()
