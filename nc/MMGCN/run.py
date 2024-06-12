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
# from Train import train
from Full_vt import full_vt
import wandb
from tqdm import tqdm
import optuna



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
    step = 0.0
    num_pbar = 0
    for idx, batch in enumerate(tqdm(dataloader)):
        node, label = batch
        optimizer.zero_grad()
        loss = model.loss(node, label)
        loss.backward(retain_graph=True)
        optimizer.step()
        num_pbar += batch_size
        sum_loss += loss.cpu().item()
        step += 1.0
        if max_step is not None and idx > max_step:
            break

    return loss


def hyperparameter_search(train_dataloader, args, v_feat, a_feat, t_feat, train_edge, num_nodes,
                          weight_decay, dim_E, valid_nodes, test_nodes, labels):
    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        num_layers = trial.suggest_int('num_layers', 1, 3)

        model = Net(v_feat, t_feat, train_edge, args.batch_size, num_nodes, 'mean', 'False', num_layers,
                    True, weight_decay, dim_E).cuda()
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': lr}])


        val_max_mrr = 0
        for epoch in range(args.num_epoch):
            train(train_dataloader, model, optimizer, args.batch_size, max_step=None)
            torch.cuda.empty_cache()

            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    representation = model.forward()
                    val_rep = representation[valid_nodes].cpu()
                    test_rep = representation[test_nodes].cpu()
                    val_label = labels[valid_nodes]
                    val_pred = torch.argmax(val_rep, dim=1)
                    val_acc = (val_pred == val_label).sum()/val_label.shape[0]
                    test_label = labels[test_nodes]
                    test_pred = torch.argmax(test_rep, dim=1)
                    test_acc = (test_pred == test_label).sum()/test_label.shape[0]

                    if val_acc > val_max_mrr:
                        val_max_mrr = val_acc
                    print(f"{lr}, {num_layers}")
                    print(f"Epoch: {epoch}, Val Acc: {val_max_mrr}, Test Acc: {test_acc}")

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

    BOOKS_PATH = "/nfs/turbo/coe-dkoutra/jing/Next-GraphGPT/amazon_nc/fashion"
    edges = torch.load(os.path.join(BOOKS_PATH, "nc_edges-nodeid.pt"))
    labels = torch.LongTensor(torch.load(os.path.join(BOOKS_PATH, "labels-w-missing.pt")))
    splits = torch.load(os.path.join(BOOKS_PATH, "split.pt"))
    train_edge = torch.LongTensor(edges)
    num_classes = 12
    num_nodes = labels.shape[0]
    clip_feat = torch.load(args.feat_path)
    t_feat = clip_feat[:, :int(clip_feat.shape[1] / 2)]
    v_feat = clip_feat[:, int(clip_feat.shape[1] / 2):]

    ##########################################################################################################################################
    print('Data loading ...')
    # if args.feat_path is None:
    #     num_nodes, train_edge, src_tgt_dict, v_feat, a_feat, t_feat = data_load_old(train_edge, args.v_feat_path,
    #                                                                                 args.t_feat_path)
    # else:
    #     num_nodes, train_edge, src_tgt_dict, v_feat, a_feat, t_feat = data_load(train_edge, args.feat_path)

    train_dataset = TrainingDataset(num_nodes, splits['train_idx'], labels)

    print('Data has been loaded.')

    if args.do_hyperparameter_search:
        valid_nodes = splits['val_idx']                
        test_nodes = splits['test_idx']
        a_feat = None
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
        hyperparameter_search(train_dataloader, args, v_feat, a_feat, t_feat, train_edge, num_nodes,
                              weight_decay, dim_E, valid_nodes, test_nodes, labels)
        return

    num_layers = args.num_layers
    learning_rate = args.l_r
    num_epoch = args.num_epoch
    batch_size = args.batch_size

    global_step = 0
    mrrs, h1s, h10s = [], [], []
    for try_time in range(args.repeat_times):
        model = Net(v_feat, t_feat, train_edge, batch_size, num_nodes, 'mean', 'False', num_layers, True, weight_decay, dim_E).cuda()
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

        max_acc = 0.0
        max_hit_1 = 0.0
        max_hit_10 = 0.0
        val_max_acc = 0.0
        num_decreases = 0
        for epoch in range(num_epoch):
            train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
            loss = train(train_dataloader, model, optimizer, batch_size, max_step=None)
            torch.cuda.empty_cache()

            with torch.no_grad():
                representation = model.forward()
                valid_nodes = splits['val_idx']
                val_rep = representation[valid_nodes].cpu()
                test_nodes = splits['test_idx']
                test_rep = representation[test_nodes].cpu()
                val_label = labels[valid_nodes]
                val_pred = torch.argmax(val_rep, dim=1)
                val_acc = (val_pred == val_label).sum()/val_label.shape[0]
                test_label = labels[test_nodes]
                test_pred = torch.argmax(test_rep, dim=1)
                test_acc = (test_pred == test_label).sum()/test_label.shape[0]

            print({
                'loss': loss,
                "val_acc": val_acc,
                "test_acc": test_acc,
            })

            if val_acc > val_max_acc:
                val_max_acc = val_acc
                max_acc = test_acc
                num_decreases = 0

                if args.PATH_best_weight_save is not None:
                    torch.save(model.state_dict(), args.PATH_best_weight_save)

                if args.PATH_best_metrics_save is not None:
                    with open(args.PATH_best_metrics_save, 'w') as f:
                        f.write(f"MRR: {max_acc}")
            global_step += 1

        mrrs.append(test_acc)

    mean_mrr = torch.mean(torch.tensor(mrrs)).item()
    std_mrr = torch.std(torch.tensor(mrrs)).item()
    print(mean_mrr)
    print(std_mrr)


if __name__ == '__main__':
    main()
