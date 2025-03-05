# SAM-Med3D for [CVPR 2025 Challenge: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/)
## Introduction
[SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) is selected as the baseline model for [CVPR 2025 Challenge: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/). 

## Quick tutorial to reproduce the SAM-Med3D baseline
### From the docker image
You can download the docker image from [TBD]() and run it with this command:
``` bash
docker container run --gpus "device=0" -m 8G --name sammed3d --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ sammed3d_baseline:latest /bin/bash -c "sh predict.sh
```
Details can be found in [link](https://github.com/JunMa11/CVPR-MedSegFMCompetition/tree/main).

### From the source code and checkpoint
Follow the following steps to run the source code:
1. download the ckpt from [TBD]() and then put it into `ckpt`.
2. put your test data into `inputs`.
3. just run the script:
``` bash 
bash predict.sh
```
