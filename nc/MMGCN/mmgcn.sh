
python run.py \
    --l_r 0.01233773647859825 \
    --weight_decay 1e-5 \
    --batch_size 1024 \
    --num_epoch 30 \
    --num_workers 4 \
    --aggr_mode mean \
    --num_layers 1 \
    --has_a False \
    --feat_path /nfs/turbo/coe-dkoutra/jing/Next-GraphGPT/amazon_nc/fashion/t5dino_feat.pt \
    --project_name mmgcn_nc \
    --wandb_run_name hard_clip\
    --PATH_best_weight_save './mmgcn_amazon_nc_ckpt.pt' \
    --PATH_best_metrics_save './mmgcn_amazon_nc_best_metrics.txt' \
    --PATH_best_hyperparameter_save './mmgcn_amazon_nc_best_hyperparameter.txt' \
    # --do_hyperparameter_search