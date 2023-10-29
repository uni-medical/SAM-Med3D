python validation.py --seed 2023\
 -vp ./results/vis_sam_med2d \
 -cp ./ckpt/sam_med2d.pth \
 -tdp ./data/validation_test1 -nc 10 \
 --image_size 256 -mt vit_b --dim 2 --save_name ./results/sam_med2d.py --ft2d 