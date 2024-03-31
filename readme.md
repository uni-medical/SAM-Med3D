# SAM-Med3D \[[Paper](https://arxiv.org/abs/2310.15161)]

<a src="https://img.shields.io/badge/cs.CV-2310.15161-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2310.15161"> 
<img src="https://img.shields.io/badge/cs.CV-2310.15161-b31b1b?logo=arxiv&logoColor=red">
<a src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat" href="https://github.com/uni-medical/SAM-Med3D/tree/main?tab=readme-ov-file#-discussion-group"> <img src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat">
</a>


<div align="center">
  <img src="assets/motivation.png">
</div>

## 🔥🌻📰 News 📰🌻🔥
- **[New Checkpoints Release]** A newer version of finetuned SAM-Med3D named `SAM-Med3D-turbo` is released now. We fine-tuned it on 44 datasets ([list](https://github.com/uni-medical/SAM-Med3D/issues/2#issuecomment-1849002225)) to improve the performance. Hope this update can help you 🙂.
- **[New Checkpoints Release]** Finetuned SAM-Med3D for organ/brain segmentation is released now! Hope you enjoy the enhanced performance for specific tasks 😉. Details are in [results](https://github.com/uni-medical/SAM-Med3D/blob/main/readme.md#-dice-on-different-anatomical-architecture-and-lesions) and [ckpt](https://github.com/uni-medical/SAM-Med3D#-checkpoint).
- **[Recommendation]** If you are interested in computer vision, 
we recommend checking out [OpenGVLab](https://github.com/OpenGVLab) for more exciting projects like [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D/tree/main)!

## 🌟 Highlights
- 📚 Curated the most extensive volumetric medical dataset to date for training, boasting 131K 3D masks and 247 categories.
- 🚤 Achieved efficient promptable segmentation, requiring 10 to 100 times fewer prompt points for satisfactory 3D outcomes.
- 🏆 Conducted a thorough assessment of SAM-Med3D across 15 frequently used volumetric medical image segmentation datasets.

## 🔨 Usage
### Training / Fine-tuning
(we recommend fine-tuning with SAM-Med3D pre-trained weights from [link](https://github.com/uni-medical/SAM-Med3D#-checkpoint))

To train the SAM-Med3D model on your own data, follow these steps:

#### 0. **(Recommend) Prepare the Pre-trained Weights**

Download the checkpoint from [ckpt section](https://github.com/uni-medical/SAM-Med3D#-checkpoint) and move the pth file into `SAM_Med3D/ckpt/` (We recommand to use `SAM-Med3D-turbo.pth`.).


#### 1. Prepare Your Training Data (from nnU-Net-style dataset): 

Ensure that your training data is organized according to the structure shown in the `data/medical_preprocessed` directories. The target file structures should be like the following:
```
data/medical_preprocessed
      ├── adrenal
      │ ├── ct_WORD
      │ │ ├── imagesTr
      │ │ │ ├── word_0025.nii.gz
      │ │ │ ├── ...
      │ │ ├── labelsTr
      │ │ │ ├── word_0025.nii.gz
      │ │ │ ├── ...
      ├── ...
```

> If the original data are in the **nnU-Net style**, follow these steps:
> 
> For a nnU-Net style dataset, the original file structure should be:
> ```
> Task010_WORD
>      ├── imagesTr
>      │ ├── word_0025_0000.nii.gz
>      │ ├── ...
>      ├── labelsTr
>      │ ├── word_0025.nii.gz
>      │ ├── ...
> ```
> Then you should resample and convert the masks into binary. (You can use [script](https://github.com/uni-medical/SAM-Med3D/blob/b77585070b2f520ecd204b551a3f27715f5b3b43/utils/prepare_data_from_nnUNet.py) for nnU-Net folder)
> ```
> data/train
>       ├── adrenal
>       │ ├── ct_WORD
>       │ │ ├── imagesTr
>       │ │ │ ├── word_0025.nii.gz
>       │ │ │ ├── ...
>       │ │ ├── labelsTr
>       │ │ │ ├── word_0025.nii.gz (binary label)
>       │ │ │ ├── ...
>       ├── liver
>       │ ├── ct_WORD
>       │ │ ├── imagesTr
>       │ │ │ ├── word_0025.nii.gz
>       │ │ │ ├── ...
>       │ │ ├── labelsTr
>       │ │ │ ├── word_0025.nii.gz (binary label)
>       │ │ │ ├── ...
>       ├── ...
> ```

Then, modify the `utils/data_paths.py` according to your own data.
```
img_datas = [
"data/train/adrenal/ct_WORD",
"data/train/liver/ct_WORD",
...
]
```


#### 2. **Run the Training Script**: 
Run `bash train.sh` to execute the following command in your terminal:

```
python train.py --multi_gpu --task_name ${tag}
```
This will start the training process of the SAM-Med3D model on your prepared data. If you use only one GPU, remove the `--multi_gpu` flag.

The key options are listed below:

- task_name: task name
- checkpoint: pre-trained checkpoint
- work_dir: results folder for log and ckpt
- multi_gpu: use multiple GPU with DDP
- gpu_ids: set gpu ids used for training
- num_epochs: number of epoches
- batch_size: batch size for training
- lr: learning rate for training


**Hint**: Use the `--checkpoint` to set the pre-trained weight path, the model will be trained from scratch if no ckpt in the path is found!

### Evaluation & Inference
Prepare your own dataset and refer to the samples in `data/validation` to replace them according to your specific scenario. 
Then you can simply run `bash val.sh` to **quickly validate** SAM-Med3D on your data. Or you can use `bash infer.sh` to **generate full-volume results** for your application.
Make sure the masks are processed into the one-hot format (have only two values: the main image (foreground) and the background). We highly recommend using the spacing of `1.5mm` for the best experience.

```
python validation.py --seed 2023\
 -vp ./results/vis_sam_med3d \
 -cp ./ckpt/sam_med3d_turbo.pth \
 -tdp ./data/medical_preprocessed -nc 1 \
 --save_name ./results/sam_med3d.py
```

- vp: visualization path, dir to save the final visualization files
- cp: checkpoint path
- tdp: test data path, where your data is placed
- nc: number of clicks of prompt points
- save_name: filename to save evaluation results 
- (optional) skip_existing_pred: skip and not predict if output file is found existing

**Sliding-window Inference (experimental)**: To extend the application scenario of SAM-Med3D and support more choices for full-volume inference. We provide the sliding-window mode here within `inference.py`. 
```
python inference.py --seed 2024\
 -cp ./ckpt/sam_med3d_turbo.pth \
 -tdp ./data/medical_preprocessed -nc 1 \
 --output_dir ./results  --task_name test_amos_move \
 #--sliding_window
 #--save_image_and_gt
```
- cp: checkpoint path
- tdp: test data path, where your data is placed
- output_dir&task_name: all your output will be saved to `<output_dir>/<task_name>`
- (optional) sliding_window: enable the sliding-window mode. model will infer 27 patches with improved accuracy and slower responce.
- (optional) save_image_and_gt: enable saving the full-volume image and ground-truth into `output_dir`, plz ensure your disk is okay when you turn on this

For validation of SAM and SAM-Med2D on 3D volumetric data, you can refer to `scripts/val_sam.sh` and `scripts/val_med2d.sh` for details.

Hint: We also provide a simple script `sum_result.py` to help summarize the results from files like `./results/sam_med3d.py`. 

## 🔗 Checkpoint
**Our most recommended version is SAM-Med3D-turbo**

| Model | Google Drive | Baidu NetDisk |
|----------|----------|----------|
| SAM-Med3D | [Download](https://drive.google.com/file/d/1PFeUjlFMAppllS9x1kAWyCYUJM9re2Ub/view?usp=drive_link) | [Download (pwd:r5o3)](https://pan.baidu.com/s/18uhMXy_XO0yy3ODj66N8GQ?pwd=r5o3) |
| SAM-Med3D-organ    | [Download](https://drive.google.com/file/d/1kKpjIwCsUWQI-mYZ2Lww9WZXuJxc3FvU/view?usp=sharing) | [Download (pwd:5t7v)](https://pan.baidu.com/s/1Dermdr-ZN8NMWELejF1p1w?pwd=5t7v) |
| SAM-Med3D-brain    | [Download](https://drive.google.com/file/d/1otbhZs9uugSWkAbcQLLSmPB8jo5rzFL2/view?usp=sharing) | [Download (pwd:yp42)](https://pan.baidu.com/s/1S2-buTga9D4Nbrt6fevo8Q?pwd=yp42) |
| SAM-Med3D-turbo    | [Download](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view?usp=sharing) | [Download (pwd:l6ol)](https://pan.baidu.com/s/1OEVtiDc6osG0l9HkQN4hEg?pwd=l6ol) |

Other checkpoints are available with their official link: [SAM](https://drive.google.com/file/d/1_U26MIJhWnWVwmI5JkGg2cd2J6MvkqU-/view?usp=drive_link) and [SAM-Med2D](https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link).

## 🗼 Method
<div align="center">
  <img src="assets/comparison.png">
</div>
<div align="center">
  <img src="assets/architecture.png">
</div>

## 🏆 Results
### 💡 Overall Performance
| **Model**    | **Prompt**   | **Resolution**                 | **Inference Time (s)** | **Overall Dice** |
|--------------|--------------|--------------------------------|------------------|------------------|
| SAM          | N points     | 1024×1024×N                    | 13               | 16.15            |
| SAM-Med2D    | N points     | 256×256×N                      | 4                | 36.83            |
| SAM-Med3D    | 1 point      | 128×128×128                    | 2                | 38.65            |
| SAM-Med3D    | 10 points    | 128×128×128                    | 6                | 49.02            |
| **SAM-Med3D-turbo** | 1 points | 128×128×128                 | 6                | 76.27            |
| **SAM-Med3D-turbo** | 10 points | 128×128×128                | 6                | 80.71            |

> **Note:** Quantitative comparison of different methods on our evaluation dataset. Here, N denotes the count of slices containing the target object (10 ≤ N ≤ 200). Inference time is calculated with N=100, excluding the time for image processing and simulated prompt generation.



### 💡 Dice on Different Anatomical Architecture and Lesions
| **Model**    | **Prompt**   | **A&T** | **Bone** | **Brain** | **Cardiac** | **Muscle** | **Lesion** | **Unseen Organ** | **Unseen Lesion** |
|--------------|--------------|---------|----------|-----------|-------------|------------|------------|-----------------|-------------------|
| SAM          | N points     | 19.93   | 17.85    | 29.73     | 8.44        | 3.93       | 11.56      | 12.14           | 8.88   |
| SAM-Med2D    | N points     | 50.47   | 32.70    | 36.00     | 40.18       | 43.85      | 24.90      | 19.36           | 44.87  |
| SAM-Med3D    | 1 point      | 46.12   | 33.30    | 49.14     | 61.04       | 53.78      | 39.56      | 23.85           | 40.53  |
| SAM-Med3D    | 10 points    | 58.61   | 43.52    | 54.01     | 68.50       | 69.45      | 47.87      | 29.05           | 48.44  |
| **SAM-Med3D-turbo** |  1 points | 80.76 | 83.38  | 43.74     | 87.12       | 89.74      | 58.06      | 35.99           | 44.22  |
| **SAM-Med3D-turbo** | 10 points | 85.42 | 85.34  | 61.27     | 90.97       | 91.62      | 64.80      | 48.10           | 62.72  |


> **Note:** Comparison from the perspective of anatomical structure and lesion. A&T represents Abdominal and Thorax targets. N denotes the count of slices containing the target object (10 ≤ N ≤ 200).


### 💡 Visualization
<div align="center">
  <img src="assets/vis_anat.png">
</div>
<div align="center">
  <img src="assets/vis_modal.png">
</div>


<!-- ## 🗓️ Ongoing 
- [ ] Dataset release
- [x] Train code release
- [x] [Feature] Evaluation on 3D data with 2D models (slice-by-slice)
- [x] Evaluation code release
- [x] Pre-trained model release
- [x] Paper release -->

## 📬 Citation
```
@misc{wang2023sammed3d,
      title={SAM-Med3D}, 
      author={Haoyu Wang and Sizheng Guo and Jin Ye and Zhongying Deng and Junlong Cheng and Tianbin Li and Jianpin Chen and Yanzhou Su and Ziyan Huang and Yiqing Shen and Bin Fu and Shaoting Zhang and Junjun He and Yu Qiao},
      year={2023},
      eprint={2310.15161},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE). 

## 💬 Discussion Group
<p align="center"><img width="100" alt="image" src="assets/QRCode.jpg"></p> 
(If the QRCode is expired, please contact the WeChat account: EugeneYonng or Small_dark8023，please note with "add sammed3d wechat"/请备注“sammed3d交流群”.)

BTW, welcome to follow our [Zhihu official account](https://www.zhihu.com/people/gmai-38), we will share more information on medical imaging there.

## 🙏 Acknowledgement
- We thank all medical workers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects:
  - [Segment Anything](https://github.com/facebookresearch/segment-anything) &#8194;
  - [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D/tree/main)

## 👋 Hiring & Global Collaboration
- **Hiring:** We are hiring researchers, engineers, and interns in General Vision Group, Shanghai AI Lab. If you are interested in Medical Foundation Models and General Medical AI, including designing benchmark datasets, general models, evaluation systems, and efficient tools, please contact us.
- **Global Collaboration:** We're on a mission to redefine medical research, aiming for a more universally adaptable model. Our passionate team is delving into foundational healthcare models, promoting the development of the medical community. Collaborate with us to increase competitiveness, reduce risk, and expand markets.
- **Contact:** Junjun He(hejunjun@pjlab.org.cn), Jin Ye(yejin@pjlab.org.cn), and Tianbin Li (litianbin@pjlab.org.cn).
