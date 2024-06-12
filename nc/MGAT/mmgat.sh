


python run.py \
    --l_r 0.0004549756183905287 \
    --weight_decay 1e-5 \
    --batch_size 2048 \
    --num_epoch 30 \
    --num_workers 4 \
    --num_layers 2 \
    --feat_path /nfs/turbo/coe-dkoutra/jing/Next-GraphGPT/amazon_nc/fashion/t5vit_feat.pt \
    --project_name mgat_nc \
    --wandb_run_name hard_clip\
    --PATH_best_weight_save './mgat_amazon_nc_ckpt.pt' \
    --PATH_best_metrics_save './mgat_amazon_nc_best_metrics.txt' \
    --PATH_best_hyperparameter_save './mgat_amazon_nc_best_hyperparameter.txt' \
    # --do_hyperparameter_search


