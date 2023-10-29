python validation.py --seed 2023\
 -vp ./results/vis_sam_vit_b \
 -cp ./ckpt/sam_vit_b.pth \
 -tdp ./data/validation_test1 -nc 10 \
 --image_size 1024 -mt vit_b --dim 2 --save_name ./results/sam_vit_b.py 