# SAM-Med3D \[[Paper](https://arxiv.org/abs/2310.15161)] \[[Suppl](https://github.com/uni-medical/SAM-Med3D/blob/main/paper/SAM_Med3D_ECCV_Supplementary.pdf)\] \[[Data](https://huggingface.co/datasets/blueyo0/SA-Med3D-140K)\]
[![x](https://img.shields.io/badge/cs.CV-2310.15161-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2310.15161)
[![x](https://img.shields.io/badge/WeChat-Group-green?logo=wechat)](https://github.com/uni-medical/SAM-Med3D/tree/main?tab=readme-ov-file#-discussion-group)
[![x](https://img.shields.io/badge/Python-3.9|3.10-A7D8FF)]()
[![x](https://img.shields.io/badge/PyTorch-2.6-FCD299)]()

The official repo of "SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images".

<div align="center">
  <img src="assets/motivation.png">
</div>

## 🔥🌻📰 News 📰🌻🔥
- **[Data]** We have now released all of our dataset SA-Med3D-140K. Full Dataset Download Link: [Huggingface](https://huggingface.co/datasets/blueyo0/SA-Med3D-140K). Labels Download Link: [Baidu Netdisk](https://pan.baidu.com/s/12Nxwd10uVZs57O8WP8Y-Hg?pwd=cv6t) and [Google Drive](https://drive.google.com/file/d/1F7lRWM5mdEKSRQtvJ8ExEyNrWIEkXc-G/view?usp=drive_link).
- **[Challenge]** SAM-Med3D is invited as a baseline of [CVPR-MedSegFMCompetition](https://www.codabench.org/competitions/5263/) and the tutorial is [here](https://github.com/uni-medical/SAM-Med3D/tree/CVPR25_3DFM). We kindly invite you to join the challenge and build better foundation models for 3D medical image segmentation!
- **[Examples]** SAM-Med3D is now supported in [MedIM](https://github.com/uni-medical/MedIM), you can easily get our model with one-line Python code. Details can be found in [`medim_val_single.py`](https://github.com/uni-medical/SAM-Med3D/blob/main/medim_val_single.py).
- **[Paper]** SAM-Med3D is accepted as [ECCV BIC 2024 Oral](https://www.bioimagecomputing.com/program/selected-contributions/)
- **[Model]** A newer version of finetuned SAM-Med3D named `SAM-Med3D-turbo` is released now. We fine-tuned it on 44 datasets ([list](https://github.com/uni-medical/SAM-Med3D/issues/2#issuecomment-1849002225)) to improve the performance. Hope this update can help you 🙂.
- **[Repos]** If you are interested in computer vision, 
we recommend checking out [OpenGVLab](https://github.com/OpenGVLab) for more exciting projects like [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D/tree/main)!

## 🌟 Highlights
- 📚 Curated the most extensive volumetric medical dataset to date for training, boasting 143K 3D masks and 245 categories.
- 🚤 Achieved efficient promptable segmentation, requiring 10 to 100 times fewer prompt points for satisfactory 3D outcomes.
- 🏆 Conducted a thorough assessment of SAM-Med3D across 16 frequently used volumetric medical image segmentation datasets.

## 🔗 Checkpoint

**SAM-Med3D-turbo**: [Hugging Face](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth) | [Google Drive](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view?usp=sharing) | [Baidu NetDisk (pwd:l6ol)](https://pan.baidu.com/s/1OEVtiDc6osG0l9HkQN4hEg?pwd=l6ol)


## 🔨 Usage
### 1. Quick Start for SAM-Med3D
> **Note:**
> **Ground-truth labels are required** to generate prompt points.
> If you want to test an image without ground-truth, please generate a fake ground-truth with the target region for prompt annotated.

First, set up your environment with the following commands:
```
conda create --name sammed3d python=3.10 
conda activate sammed3d
pip install uv
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
uv pip install torchio opencv-python-headless matplotlib prefetch_generator monai edt surface-distance medim
```

Then, use [`medim_val_single.py`](https://github.com/uni-medical/SAM-Med3D/blob/main/medim_val_single.py) to test the model:
```
python medim_val_single.py
```
You could set your custom data in the code like:
```
img_path = "./test_data/Seg_Exps/ACDC/ACDC_test_cases/patient101_frame01_0000.nii.gz"
gt_path =  "./test_data/Seg_Exps/ACDC/ACDC_test_gts/patient101_frame01.nii.gz"
out_path = "./test_data/Seg_Exps/ACDC/ACDC_test_SAM_Med3d/patient101_frame01.nii.gz"
```


### 2. Steps for Training / Fine-tuning
(we recommend fine-tuning with SAM-Med3D pre-trained weights from [link](https://github.com/uni-medical/SAM-Med3D#-checkpoint))

To train the SAM-Med3D model on your own data, follow these steps:

#### 0. **(Recommend) Prepare the Pre-trained Weights**

> Note: You can easily get PyTorch SAM-Med3D model with pre-trained weights from huggingface use `MedIM`.
> ```
> ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
> model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
> ```

Download the checkpoint from [ckpt section](https://github.com/uni-medical/SAM-Med3D#-checkpoint) and move the pth file into `SAM_Med3D/ckpt/` (We recommand to use `SAM-Med3D-turbo.pth`).


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

Then, modify `img_datas` in `utils/data_paths.py` according to your own data.
```
img_datas = [
"data/train/adrenal/ct_WORD",
"data/train/liver/ct_WORD",
...
]
```
or
```
PROJ_DIR = <YOUR PROJ DIR>
img_datas = glob(os.path.join(PROJ_DIR, "data", "train", "*", "*"))
```


#### 2. **Run the Training Script**: 
You can refer to [`train.sh`](https://github.com/uni-medical/SAM-Med3D/blob/main/train.sh) and [train_ddp.sh](https://github.com/uni-medical/SAM-Med3D/blob/main/train_ddp.sh) for training.



**Hint**: Use the `--checkpoint` to set the pre-trained weight path, the model will be trained from scratch if no ckpt in the path is found!

## 🗼 Method
<div align="center">
  <img src="assets/comparison.png">
</div>
<div align="center">
  <img src="assets/architecture.png">
</div>


## 📬 Citation
```
@misc{wang2024sammed3dgeneralpurposesegmentationmodels,
      title={SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images}, 
      author={Haoyu Wang and Sizheng Guo and Jin Ye and Zhongying Deng and Junlong Cheng and Tianbin Li and Jianpin Chen and Yanzhou Su and Ziyan Huang and Yiqing Shen and Bin Fu and Shaoting Zhang and Junjun He and Yu Qiao},
      year={2024},
      eprint={2310.15161},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2310.15161}, 
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
