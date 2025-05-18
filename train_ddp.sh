# num_workers is set to 6 * 8 = 48
# batch_size is the micro batch for each gpu, 
# the total_batch_size = batch_size * len(gpu_ids)

python train.py \
 --batch_size 6 --multi_gpu --gpu_ids 0 1 2 3 4 5 6 7 \
 --num_workers 48 \
 --task_name "ft_ddp_b6x8" \
 --checkpoint "ckpt/sam_med3d_turbo.pth" \
 --lr 8e-5