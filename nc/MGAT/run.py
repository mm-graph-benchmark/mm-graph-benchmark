import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from DataLoad import DataLoad
from Model import MGAT
import wandb
import pickle
import optuna


def batch_evaluate_model(representation: torch.Tensor, triple_data, batch_size=256, verbose=False):
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


def train(dataloader, model, optimizer, max_step=None):
    model.train()
    sum_loss = 0.0
    for idx, data in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        optimizer.step()
        sum_loss += loss.cpu().item()

        if max_step is not None and idx > max_step:
            break
    return sum_loss / idx



def hyperparameter_search(train_dataloader, args, features, edge_index, num_nodes, valid_nodes, test_nodes, labels):
    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        num_layers = trial.suggest_int('num_layers', 1, 3)

        model = MGAT(features, edge_index, args.batch_size, num_nodes, num_layers,
                     args.dim_latent).cuda()
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': lr}],
                                     weight_decay=args.weight_decay)
        val_max_mrr = 0

        for epoch in range(args.num_epoch):
            loss = train(train_dataloader, model, optimizer, max_step=None)
            torch.cuda.empty_cache()

            print({"loss": loss})
            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    v_rep = model.v_gnn(model.id_embedding)
                    t_rep = model.t_gnn(model.id_embedding)
                    representation = (v_rep + t_rep) / 2
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
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='MGAT', help='Model name.')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--dim_latent', type=int, default=32, help='Latent dimension.')
    parser.add_argument('--num_workers', type=int, default=4, help='Workers number')

    # your data here
    parser.add_argument('--v_feat_path', default=None)
    parser.add_argument('--t_feat_path', default=None)
    parser.add_argument('--feat_path', default=None)

    # your setting here
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number')
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
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

    print('Data loading ...')
    BOOKS_PATH = "/nfs/turbo/coe-dkoutra/jing/Next-GraphGPT/amazon_nc/fashion"
    edges = torch.load(os.path.join(BOOKS_PATH, "nc_edges-nodeid.pt"))
    labels = torch.LongTensor(torch.load(os.path.join(BOOKS_PATH, "labels-w-missing.pt")))
    splits = torch.load(os.path.join(BOOKS_PATH, "split.pt"))
    train_edge = torch.LongTensor(edges)
    num_classes = 12
    num_nodes = labels.shape[0]
    if args.feat_path is None:
        v_feat = torch.load(args.v_feat_path).to('cuda')
        t_feat = torch.load(args.t_feat_path).to('cuda')
    else:
        feat = torch.load(args.feat_path).to('cuda')
        v_feat = feat[:, int(feat.shape[1] / 2):]
        t_feat = feat[:, :int(feat.shape[1] / 2)]
    a_feat = None
    features = [v_feat, a_feat, t_feat]
    print(f"number of nodes: {num_nodes}")
    edge_index = train_edge
    train_dataset = DataLoad(num_nodes, splits['train_idx'], labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)

    valid_nodes = splits['val_idx']                
    test_nodes = splits['test_idx']
    print('Data has been loaded.')

    if args.do_hyperparameter_search:
        hyperparameter_search(train_dataloader, args, features, edge_index, num_nodes, valid_nodes, test_nodes, labels)
        return

    num_epoch = args.num_epoch
    learning_rate = args.l_r
    weight_decay = args.weight_decay

    if args.report_to == 'wandb':
        wandb.init(
            project=args.project_name,
            name=args.wandb_run_name,
            config={
                "learning_rate": learning_rate,
                "epochs": num_epoch,
            }
        )

    global_cur_epoch = 0
    mrrs, h1s, h10s = [], [], []

    for _ in range(args.repeat_times):
        model = MGAT(features, edge_index, args.batch_size, num_nodes, args.num_layers,
                     args.dim_latent).cuda()
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}],
                                     weight_decay=weight_decay)

        max_acc = 0.0
        max_hit_1 = 0.0
        max_hit_10 = 0.0
        val_max_acc = 0.0
        num_decreases = 0

        for epoch in range(num_epoch):
            loss = train(train_dataloader, model, optimizer, max_step=None)

            torch.cuda.empty_cache()

            # print({"loss": loss})

            with torch.no_grad():
                v_rep = model.v_gnn(model.id_embedding)
                t_rep = model.t_gnn(model.id_embedding)
                representation = (v_rep + t_rep) / 2

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

                if val_acc > val_max_acc:
                    val_max_acc = val_acc
                    max_acc = test_acc

                    num_decreases = 0

                    if args.PATH_best_weight_save is not None:
                        torch.save(model.state_dict(), args.PATH_best_weight_save)

                    if args.PATH_best_metrics_save is not None:
                        with open(args.PATH_best_metrics_save, 'w') as f:
                            f.write(f"MRR: {max_acc}")

            global_cur_epoch += 1

            print({
                'loss': loss,
                "val_acc": val_acc,
                "test_acc": test_acc,
            })
        mrrs.append(max_acc)

    mean_mrr = torch.mean(torch.tensor(mrrs)).item()
    std_mrr = torch.std(torch.tensor(mrrs)).item()
    print(mean_mrr)
    print(std_mrr)

if __name__ == '__main__':
    main()
