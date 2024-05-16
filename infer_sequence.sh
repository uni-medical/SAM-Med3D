python infer_sequence.py --seed 2023 \
 -tdp ./data/inference -nc 1 \
 -cp ./work_dir/fine_tune_experimental_augmented/sam_model_latest.pth \
 --output_dir ./results/sequence  \
 --task_name sequence
