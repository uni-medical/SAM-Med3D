"""
The code was adapted from the CVPR24 Segment Anything in Medical Images on a Laptop Challenge
https://www.codabench.org/competitions/1847/ 

pip install connected-components-3d
pip install cupy-cuda12x
pip install cucim-cu12


The testing images will be evaluated one by one.

Folder structure:
CVPR25_iter_eval.py
--docker_folder path # submitted docker containers from participants
    - docker_dir
        - teamname_1.tar.gz 
        - teamname_2.tar.gz
        - ...
--test_img_path # test images
    - imgs
        - case1.npz  # test image
        - case2.npz  
        - ...   
--save_path  # segmentation results
    - output
        - case1.npz  # segmentation file name is the same as the testing image name
        - case2.npz  
        - ...
--validation_gts_path # path to validation / test set GT files
    -   Contains the npz files with the same name as the images but only 'gts' key is available in each file instead of storing it in the image itself. This is done to prevent label leakage during the challenge.
    - validation_gts
        - case1.npz  # file containing only the 'gts' key
        - case2.npz  
        - ...
--verbose
    -   Whether to have a more detailed output, e.g. coordinates of generated clicks


This script is designed for evaluating docker submissions for the CVPR25: Foundation Models for Interactive 3D Biomedical Image Segmentation Challenge Challenge

##########################################################
######### Docker Submission Evaluation Process ###########
##########################################################
Submissions for the CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation Challenge will be evaluated using an iterative refinement approach. 
Each participant's Docker container will be tested on a set of medical images provided as .npz files. 
The evaluation process follows these key steps:
    -   Initial Prediction: Image +  Bounding Box Prompt (1 prediction)
        -   Each test case begins with a bounding box prompt, specified in the 'bbox' key of the test image. This serves as the starting point for the segmentation.
    -   Iterative Click Refinements: Image + Bounding Box + 1-5 Clicks (5 predictions)
        -   After the initial segmentation, we iteratively simulate 5 refinement clicks to address segmentation errors. These clicks are automatically generated based on the center of the largest error region in the current prediction:
            -   If the center of the largest error is an undersegmentation, we simulate and place a foreground click.
            -   If the center of the largest error is an oversegmentation, we simulate and place a background click.
        -   The clicks are stored in the clicks key of the 'npz' file and progressively updated during the second step of the evaluation.

###############################################################
######### How are interactions (bbox, clicks) stored? #########
###############################################################
The interactions are stored in the 'bbox' and 'clicks' keys of each input .npz image.
    - The bounding box is stored in the 'bbox' key as a list of dictionaries [{'z_min': 27, 'z_max': 396, 'z_mid': 311, 'z_mid_x_min': 175, 'z_mid_y_min': 94, 'z_mid_x_max': 278, 'z_mid_y_max': 233}, ...] containing bbox coordinates for each class.
    - The clicks are provided in the 'clicks' key as a list of dictionaries [{'fg': [click_fg_1, clicks_fg_2,...], 'bg': [click_bg_1, click_bg_2,...]}, ...] 
where click_fg_i and click_bg_i are 3-element arrays with the 3D click coordinates [x, y, z].

#######################################
######### Performance Metrics #########
#######################################
For each image, multi-class segmentation quality is evaluated using:
-   Dice Similarity Coefficient (DSC) and Normalized Surface Dice (NSD), calculated iteratively over the 6 steps (bounding box + 5 clicks).
-   AUC (Area Under the Curve) for DSC and NSD to measure cumulative improvement with more interactions.
-   Final DSC and NSD after all interactions.
-   Inference Time averaged over all 6 steps.

##########################
######### Output #########
##########################
Results are saved in .npz format with metrics compiled into a CSV file for each submission. 5 metrics are stored: DSC_AUC, NSD_AUC, Final_DSC, Final_NSD, Inference Time.


################################
######### Script Steps #########
################################
This script executes the following steps:
1. Docker Submission Handling:
   - Loads docker containers submitted by participants.
   - Executes inference for each test image using the participant's docker container. Images are infered one by one.

2. Iterative Refinement:
   - The initial bounding box prediction is refined iteratively by simulating user clicks at the centers of segmentation errors for each class in the image.
   - The Euclidean Distance Transform (EDT) is computed for error regions to identify the distance to the boundary of each error component,
     ensuring clicks are placed at locations at the center of the largest error for refinement.
   - For each image, the docker is run 6 times for inference:
        - 1) Bounding Box initial prediction
        - 2)-6) Click refinement predictions (each new click is placed in the center of the largest error component)
            - If the center of the largest error is part of the background --> a background click is placed
            - Otherwise, a foreground click is placed
        - Steps 1)-6) are done in parallel for all segmentation classes in 6 interaction steps (6 docker runs) 

3. GPU vs. CPU Computation:
   - If a GPU is available, the script uses `cupy` and `cucim` for accelerated EDT computation.
   - For CPU-only environments, `scipy.ndimage.distance_transform_edt` is used as a fallback.

4. Metrics Calculation:
   - Computes multi-class DSC and NSD for each image.
   - For the final metrics, the AUC (Area Under the Curve) for the DSC and NSD are computed for iterative improvement across the 6 interactive iterations. 
        - The AUC quantifies the cumulative performance improvement over the 6 successive iterations (bbox + 5 clicks) providing a holistic view of the segmentation refinement process.
   - The final DSC and NSD after all 6 interactive steps are also computed.
        - These metrics reflect the final segmentation quality achieved after all refinements, indicating the model's final performance.
   - The last metric is the inference time which is the average inference time over the 6 interactive steps.

5. Output:
   - Segmentation results are saved in the specified output directory. 
        -   Final prediction in the 'segs' key
        -   Intermediate prediction in the 'all_segs' key
   - Metrics for each test case are compiled into a CSV file.

#################################
############## Misc##############
#################################
- The input image also contains the 'prev_pred' key which stores the prediction from the previous iteration. This is used only to help with submission that are using the previous prediction as an additional input and is not
a mandatory input.
"""

import os
join = os.path.join
import shutil
import time
import torch
import argparse
from collections import OrderedDict
import pandas as pd
import numpy as np


from scipy.ndimage import distance_transform_edt 
import cc3d
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from scipy import integrate
from tqdm import tqdm

# Taken from CVPR24 challenge code with change to np.unique
def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.unique(gt)[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)

# Taken from CVPR24 challenge code with change to np.unique
def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in np.unique(gt)[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
        print(f"cls:{i}\t{nsd[-1]}")
    return np.mean(nsd)

parser = argparse.ArgumentParser('Segmentation iterative refinement with clicks eavluation for docker containers', add_help=False)
parser.add_argument('-i', '--test_img_path', default='3D_val_npz', type=str, help='testing data path')
parser.add_argument('-o','--save_path', default='./seg', type=str, help='segmentation output path')
parser.add_argument('-d','--docker_folder_path', default='./team_docker', type=str, help='team docker path')
parser.add_argument('-val_gts','--validation_gts_path', default='3D_val_gt_interactive_seg', type=str, help='path to validation set (or final test set) GT files')
parser.add_argument('-v','--verbose', default=False, action='store_true', help="Verbose output, e.g., print coordinates of generated clicks")

args = parser.parse_args()

test_img_path = args.test_img_path
save_path = args.save_path
docker_path = args.docker_folder_path
validation_gts_path = args.validation_gts_path
verbose = args.verbose

input_temp = './inputs/'
output_temp = './outputs'
os.makedirs(save_path, exist_ok=True)

# dockers = sorted(os.listdir(docker_path))
test_cases = sorted(os.listdir(test_img_path))

for docker in ["test_local"]:
    try:
        # create temp folers for inference one-by-one
        if os.path.exists(input_temp):
            shutil.rmtree(input_temp)
        if os.path.exists(output_temp):
            shutil.rmtree(output_temp)
        os.makedirs(input_temp)
        os.makedirs(output_temp)

        # load docker and create a new folder to save segmentation results
        teamname = docker.split('.')[0].lower()
        print('teamname docker: ', docker)
        # os.system('docker image load -i {}'.format(join(docker_path, docker)))
        team_outpath = join(save_path, teamname)
        if os.path.exists(team_outpath):
            shutil.rmtree(team_outpath)
        os.makedirs(team_outpath)
        # os.system(f'chmod -R 777 ./* >/dev/null 2>&1') # ignore output warnings/errors of this command with >/dev/null 2>&1
        
        # Evaluation Metrics
        metric = OrderedDict()
        metric['CaseName'] = []
        # 5 Metrics
        metric['TotalRunningTime'] = []
        metric['RunningTime_1'] = []
        metric['RunningTime_2'] = []
        metric['RunningTime_3'] = []
        metric['RunningTime_4'] = []
        metric['RunningTime_5'] = []
        metric['RunningTime_6'] = []
        metric['DSC_AUC'] = []
        metric['NSD_AUC'] = []
        metric['DSC_Final'] = []
        metric['NSD_Final'] = []
        n_clicks = 5
        time_warning = False

        # To obtain the running time for each case, testing cases are inferred one-by-one
        for case in tqdm(test_cases):
            real_running_time = 0
            dscs = []
            nsds = []
            all_segs = []
            no_bbox = False

            # copy input image to accumulate clicks in its dict
            shutil.copy(join(test_img_path, case), input_temp)
            if validation_gts_path is  None: # for training images
                gts = np.load(join(input_temp, case))['gts']
            else: # for validation or test images --> gts are in separate files to avoid label leakage during the course of the challenge
                gts = np.load(join(validation_gts_path, case))['gts']

            # foreground and background clicks for each class
            clicks_cls = [{'fg': [], 'bg': []} for _ in np.unique(gts)[1:]] # skip background class 0 
            
            for it in range(n_clicks + 1): # + 1 due to bbox pred at iteration 0
                if it == 0:
                    if "boxes" not in np.load(join(input_temp, case)).keys():
                        if verbose:
                            print(f'This sample does not use a Bounding Box for the initial iteration {it}') 
                        no_bbox = True
                        metric["RunningTime_1"] = 0
                        continue
                    if verbose:
                        print(f'Using Bounding Box for iteration {it}') 
                else:
                    if verbose:
                        print(f'Using Clicks for iteration {it}')
                    if os.path.isfile(join(output_temp, case)):
                        segs = np.load(join(output_temp, case))['segs'].astype(np.uint8) # previous prediction
                    else:
                        segs = np.zeros_like(gts).astype(np.uint8) # in case the bbox prediction did not produce a result
                    all_segs.append(segs.astype(np.uint8))

                    # Refinement clicks
                    for ind, cls in enumerate(sorted(np.unique(gts)[1:])):
                        if cls == 0:
                            continue # skip background

                        segs_cls = (segs == cls).astype(np.uint8)
                        gts_cls = (gts == cls).astype(np.uint8)

                        # Compute error mask
                        error_mask = (segs_cls != gts_cls).astype(np.uint8)
                        if np.sum(error_mask) > 0:
                            errors = cc3d.connected_components(error_mask, connectivity=26)  # 26 for 3D connectivity

                            # Calculate the sizes of connected error components
                            component_sizes = np.bincount(errors.flat)

                            # Ignore non-error regions 
                            component_sizes[0] = 0

                            # Find the largest error component
                            largest_component_error = np.argmax(component_sizes)

                            # Find the voxel coordinates of the largest error component
                            largest_component = (errors == largest_component_error)

                            # Get bounding box of the largest error component to limit computation
                            coords = np.argwhere(largest_component)
                            min_coords = coords.min(axis=0)
                            max_coords = coords.max(axis=0) + 1

                            # Crop error to the bounding box of the largest error component
                            cropped_mask = largest_component[
                                min_coords[0]:max_coords[0],
                                min_coords[1]:max_coords[1],
                                min_coords[2]:max_coords[2],
                            ]

                            # Compute distance transform only within the bounding box to save time
                            # if torch.cuda.is_available(): # GPU available
                            #     import cupy as cp
                            #     from cucim.core.operations import morphology
                            #     error_mask_cp = cp.array(cropped_mask)
                            #     edt_cp = morphology.distance_transform_edt(error_mask_cp)
                            #     center = cp.unravel_index(cp.argmax(edt_cp), edt_cp.shape)
                            #     center = np.array([int(center[0]), int(center[1]), int(center[2])])
                            # else: # CPU available only
                            edt = distance_transform_edt(cropped_mask)
                            # Find the center in the cropped mask
                            center = np.unravel_index(np.argmax(edt), edt.shape)

                            center = tuple(min_coords + center)


                            if gts_cls[center] == 0: # oversegmentation -> place background click
                                assert segs_cls[center] == 1
                                clicks_cls[ind]['bg'].append(list(center))
                            else: # undersegmentation -> place foreground click
                                assert segs_cls[center] == 0
                                clicks_cls[ind]['fg'].append(list(center))

                            assert largest_component[center] # click within error

                            if verbose:
                                print(f"Class {cls}: Largest error component center is at {center}")
                        else:
                            if verbose:
                                print(f"Class {cls}: No error connected components found. Prediction is perfect! No clicks were added.")
                    # import pdb; pdb.set_trace()
                    # update model input with new click
                    input_img = np.load(join(input_temp, case))

                    if validation_gts_path is None:
                        np.savez_compressed(
                            join(input_temp, case),
                            imgs=input_img['imgs'],
                            gts=input_img['gts'], # only for training images
                            spacing=input_img['spacing'],
                            clicks=clicks_cls, 
                            prev_pred=segs,
                        ) 
                    else:
                        np.savez_compressed(
                            join(input_temp, case),
                            imgs=input_img['imgs'],
                            spacing=input_img['spacing'],
                            clicks=clicks_cls, 
                            prev_pred=segs,
                        ) 
                

                # Model inference on the current input
                # if torch.cuda.is_available(): # GPU available
                #     cmd = 'docker container run --gpus "device=0" -m 8G --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname, teamname)
                # else:
                #     cmd = 'docker container run -m 8G --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname, teamname)
                cmd = 'bash predict.sh'
                if verbose:
                    print(teamname, ' docker command:', cmd, '\n', 'testing image name:', case)
                start_time = time.time()
                os.system(cmd)
                infer_time = time.time() - start_time
                real_running_time += infer_time # only add the inference time without the click generation time
                print(f"{case} finished! Inference time: {infer_time}")
                metric[f"RunningTime_{it + 1}"] = infer_time

                if not os.path.isfile(join(output_temp, case)):
                    print(f"[WARNING] Failed / Skipped prediction for iteration {ind}! Setting predcition to zeros...")
                    segs = np.zeros_like(gts).astype(np.uint8)
                else:
                    segs = np.load(join(output_temp, case))['segs']
                all_segs.append(segs.astype(np.uint8))

                dsc = compute_multi_class_dsc(gts, segs)
                # compute nsd
                if dsc > 0.2:
                    # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
                    nsd = compute_multi_class_nsd(gts, segs, np.load(join(input_temp, case))['spacing'])
                else:
                    nsd = 0.0 # Assume model performs poor on this sample
                dscs.append(dsc)
                nsds.append(nsd)
                print('Dice', dsc, 'NSD', nsd)
                seg_name = case


                # Copy temp prediction to the final folder
                try:
                    shutil.copy(join(output_temp, seg_name), join(team_outpath, seg_name))
                    segs = np.load(join(team_outpath, seg_name))['segs']
                    np.savez_compressed(
                        join(team_outpath, seg_name),
                        segs=segs,
                        all_segs=all_segs, # store all intermediate predictions
                    ) 
                except:
                    print(f"{join(output_temp, seg_name)}, {join(team_outpath, seg_name)}")
                    print("Final prediction could not be copied!")
            

            if real_running_time > 90 * (len(np.unique(gts)) - 1):
                print("[WARNING] Your model seems to take more than 90 seconds per class during inference! The final test set will have a time constraint of 90s per class --> Make sure to optimize your approach!")
                time_warning = True
            # Compute interactive metrics
            n_interactions = n_clicks if no_bbox else n_clicks + 1
            dsc_auc = integrate.cumulative_trapezoid(np.array(dscs), np.arange(n_interactions))[-1]
            nsd_auc = integrate.cumulative_trapezoid(np.array(nsds), np.arange(n_interactions))[-1]
            dsc_final = dscs[-1]
            nsd_final = nsds[-1]
            metric['CaseName'].append(case)
            metric['TotalRunningTime'].append(real_running_time)
            metric['DSC_AUC'].append(dsc_auc)
            metric['NSD_AUC'].append(nsd_auc)
            metric['DSC_Final'].append(dsc_final)
            metric['NSD_Final'].append(nsd_final)
            os.remove(join(input_temp, case))  

            metric_df = pd.DataFrame(metric)
            metric_df.to_csv(join(team_outpath, teamname + '_metrics.csv'), index=False)

        # Clean up for next docker
        torch.cuda.empty_cache()
        os.system("docker rmi {}:latest".format(teamname))
        shutil.rmtree(input_temp)
        shutil.rmtree(output_temp)
        if time_warning: # repeat warning at the end as well
            print("[WARNING] Your model seems to take more than 90 seconds per class during inference for some images! The final test set will have a time constraint of 90s per class --> Make sure to optimize your approach!")
    except Exception as e:
        print(e)