
export PYTHONPATH=../:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3


python3 tools/train.py \
--dataroot datasets \
--backbone resnext101 \
--dataset_list DiverseDepth \
--batchsize 1 \
--base_lr 0.001 \
--use_tfboard \
--thread 4 \
--loss_mode _ranking-edge_pairwise-normal-regress-edge_msgil-normal_meanstd-tanh_pairwise-normal-regress-plane_ranking_ \
--epoch 50 \
--lr_scheduler_multiepochs 10 25 40 \
--val_step 5000 \
--snapshot_iters 5000 \
--log_interval 10 \
--resume --load_ckpt resnext101.pth
