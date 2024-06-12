


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


#srun --time=72:00:00 --qos=huge-long --partition=clip --account=clip \
#--gres=gpu:rtxa6000:1 --cpus-per-task=4 --mem=20g \
#python run.py --weight_decay 1e-1 --batch_size 2048 --num_epoch 20 \
#--edge_split_path ../../../../Patton/books-lp/lp-edge-split-random.pt \
#--feat_path ../../../../Patton/books-lp/t5vit_feat.pt \
#--project_name mmgat_book \
#--wandb_run_name random_t5vit \
#--PATH_best_weight_save './mmgat_book_ckpt_t5.pt' \
#--PATH_best_metrics_save './mmgat_book_best_metrics_t5.txt' \
#--PATH_best_hyperparameter_save './mmgat_book_best_hyperparameter_t5.txt' \
#--num_layers 2 \
#--l_r 0.002 \
#--do_hyperparameter_search \
