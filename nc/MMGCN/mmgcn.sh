
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

#srun --qos=huge-long --partition=clip --account=clip \
#--time=60:00:00 --gres=gpu:rtxa6000:1 --cpus-per-task=4 --mem=20g \
#python run.py --weight_decay 1e-5 --batch_size 2048 --num_epoch 30 --num_workers 4 --aggr_mode mean --has_a False \
#--edge_split_path ../../../../Patton/books-lp/lp-edge-split-random.pt \
#--feat_path ../../../../Patton/books-lp/t5vit_feat.pt --project_name mmgcn_cloth --wandb_run_name hard_clip \
#--PATH_best_weight_save './mmgcn_book_ckpt_clip.pt' --PATH_best_metrics_save './mmgcn_book_best_metrics_clip.txt' \
#--PATH_best_hyperparameter_save './mmgcn_book_best_hyperparameter_clip.txt' \
#--num_layers 2 --l_r 0.002