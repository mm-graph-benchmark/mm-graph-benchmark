
python run.py \
    --l_r 0.002 \
    --weight_decay 1e-5 \
    --batch_size 1024 \
    --num_epoch 100 \
    --num_workers 4 \
    --aggr_mode mean \
    --num_layers 2 \
    --has_a False \
    --edge_split_path /nfs/turbo/coe-dkoutra/jing/Next-GraphGPT/Patton/sports/lp-edge-split.pt \
    --feat_path /nfs/turbo/coe-dkoutra/jing/Next-GraphGPT/Patton/cloth/clip_feat.pt \
    --project_name mmgcn_cloth \
    --wandb_run_name hard_clip\
    --PATH_best_weight_save './mmgcn_cloth_ckpt.pt' \
    --PATH_best_metrics_save './mmgcn_cloth_best_metrics.txt' \
    --PATH_best_hyperparameter_save './mmgcn_cloth_best_hyperparameter.txt'
