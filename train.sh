python train.py --batch_size 2 --num_epochs 200 --task_name baseline3d --model_type vit_b_ori --num_workers 4
python train.py --batch_size 2 --num_epochs 200 --task_name pe2d --model_type vit_b_pe2d --num_workers 4
python train.py --batch_size 2 --num_epochs 200 --task_name att2d --model_type vit_b_att2d --num_workers 4
python train.py --batch_size 2 --num_epochs 200 --task_name pe2d_att2d --model_type vit_b_pe2d_att2d --num_workers 4