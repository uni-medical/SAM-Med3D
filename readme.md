# SAM-Med3D for [CVPR 2025 Challenge: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/)
## Introduction
[SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) is selected as the baseline model for [CVPR 2025 Challenge: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/). 
And here's a quick tutorial to help you start from our baseline.

## Quick tutorial to reproduce the SAM-Med3D baseline
### From the docker image
You can download the docker image from [TBD]() and run it with this command:
``` bash
docker container run --gpus "device=0" -m 8G --name sammed3d --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ sammed3d_baseline:latest /bin/bash -c "sh predict.sh
```
Details can be found in [link](https://github.com/JunMa11/CVPR-MedSegFMCompetition/tree/main).

### From the source code and checkpoint
Follow the following steps to run the source code:
1. download the ckpt from [TBD]() and then put it into `ckpt`;
2. put your test data into `inputs`;
3. follow the [readme](https://github.com/uni-medical/SAM-Med3D?tab=readme-ov-file#quick-start-for-sam-med3d-inference) in SAM-Med3D to install packages;
4. install the special version of `medim` from https://github.com/uni-medical/MedIM/tree/CVPR25_3DFM with these commands:
``` bash
git clone https://github.com/uni-medical/MedIM.git --branch CVPR25_3DFM
cd MedIM
pip install -e .
```
5. run the script to predict images in `inputs` into `outputs`
``` bash 
bash predict.sh
```
