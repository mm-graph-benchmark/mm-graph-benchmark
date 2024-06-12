
python run.py \
    --l_r 0.002 \
    --weight_decay 1e-1 \
    --batch_size 2048 \
    --num_epoch 100 \
    --edge_split_path /nfs/turbo/coe-dkoutra/jing/Next-GraphGPT/Patton/cloth/lp-edge-split-random.pt \
    --feat_path /nfs/turbo/coe-dkoutra/jing/Next-GraphGPT/Patton/cloth/t5vit_feat.pt \
    --project_name mmgat_cloth \
    --wandb_run_name random_t5vit \
    --num_layers 2 \
    --PATH_best_weight_save './mmgat_cloth_ckpt.pt' \
    --PATH_best_metrics_save './mmgat_cloth_best_metrics.txt' \
    --PATH_best_hyperparameter_save './mmgat_cloth_best_hyperparameter.txt' \
