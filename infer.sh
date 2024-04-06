python inference.py --seed 2024\
 -cp ./ckpt/sam_med3d_turbo.pth \
 -tdp ./data/medical_preprocessed -nc 1 \
 --output_dir ./results  \
 --task_name infer_turbo  
 #--sliding_window
 #--save_image_and_gt
