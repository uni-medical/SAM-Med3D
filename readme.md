# SAM-Med3D Baseline for [CVPR 2025 Challenge: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/)
## 📖 Introduction
[SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) is selected as the baseline model for [CVPR 2025 Challenge: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/). 
And here's a quick tutorial to help you start from our baseline.

## ➡️ Quick tutorial to reproduce the SAM-Med3D baseline
### How to infer/evaluate with SAM-Med3D
#### From the docker image
You can download the docker image from [Google Drive](https://drive.google.com/file/d/1NO6sPJT9dQXSYNK_y2_L7V109yOnAgi7/view?usp=drive_link) and run it with this command:
``` bash
docker container run --gpus "device=0" -m 8G --name sammed3d --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ sammed3d_baseline:latest /bin/bash -c "sh predict.sh
```
This docker will predict all data in `inputs` and save the results in the `outputs`. Details of the challenge evaluation code can be found in [link](https://github.com/JunMa11/CVPR-MedSegFMCompetition/tree/main).

#### From the source code and checkpoint
Follow the following steps to run the prediction code:
1. Download the finetuned version of ckpt `sam_med3d_turbo_bbox_cvpr.pth` from [huggingface](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo_bbox_cvpr.pth) and then put it into `ckpt`;
2. Put your test data into `inputs`;
3. Follow the [readme](https://github.com/uni-medical/SAM-Med3D?tab=readme-ov-file#quick-start-for-sam-med3d-inference) in SAM-Med3D to install dependency:
``` bash
conda create --name sammed3d python=3.10 
conda activate sammed3d
pip install light-the-torch && ltt install torch
pip install torchio opencv-python-headless matplotlib prefetch_generator monai edt
```
4. Install the special version of `medim` from https://github.com/uni-medical/MedIM/tree/CVPR25_3DFM with these commands:
``` bash
git clone https://github.com/uni-medical/MedIM.git --branch CVPR25_3DFM
pushd MedIM
pip install -e .
popd
```
5. Run the script to predict images in `inputs`
``` bash 
bash predict.sh
```

### How to train a SAM-Med3D baseline
Follow the following steps to run the training code:
1. Download the training data and preprocess them with [data/resample_preprocess_for_CVPR_3DFM.py](data/resample_preprocess_for_CVPR_3DFM.py). And check the [utils/data_paths.py](utils/data_paths.py) to set correct `img_datas`:
```python
from glob import glob 
import os.path as osp
PROJ_DIR=osp.dirname(osp.dirname(__file__))
img_datas = glob(osp.join(PROJ_DIR, "data", "resampled_3D_train_npz_random_10percent_16G", "*", "*"))
```
2. Download the pre-trained checkpoint from [huggingface](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo_bbox_init.pth) (to support potential bbox input, we add randomly initialized weights for bbox embedding above [the original checkpoint](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth)).
3. Follow the [readme](https://github.com/uni-medical/SAM-Med3D?tab=readme-ov-file#quick-start-for-sam-med3d-inference) in SAM-Med3D to install dependency:
``` bash
conda create --name sammed3d python=3.10 
conda activate sammed3d
pip install light-the-torch && ltt install torch
pip install torchio opencv-python-headless matplotlib prefetch_generator monai edt
```
4. Run the script to start the training
```bash
bash train.sh
```
5. If you get CUDA OOM or you want to adjust some hyperparameters, change `train.sh` as u wish.
```bash
python train.py \
 --batch_size 6 \
 --task_name "ft_3DFM_b6x1" \
 --checkpoint "ckpt/sam_med3d_turbo_bbox_init.pth" \
 --lr 8e-5
```
### Minimum Training Requirements
- **GPU**: 16 GB DRAM (`batch_size=2`), 48 GB DRAM (`batch_size=6`)
- **Memory**: 64 GB
- **Expected Time Cost**: ~6 GPU days for a 200-epoch experiment (1 epoch takes 0.8 GPU hours) for the Coreset Data Track (10% of all data)

## 📬 Citation
```bibtex
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

