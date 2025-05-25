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
import traceback

from scipy.ndimage import distance_transform_edt 
import cc3d
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from scipy import integrate
from tqdm import tqdm

# Taken from CVPR24 challenge code with change to np.unique
def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.sort(pd.unique(gt.ravel()))[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)

# Taken from CVPR24 challenge code with change to np.unique
def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in np.sort(pd.unique(gt.ravel()))[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)

def patched_np_load(*args, **kwargs):
    with np.load(*args, **kwargs) as f:
        return dict(f) 

def sample_coord(edt):
    # Find all coordinates with max EDT value
    np.random.seed(42)

    max_val = edt.max()
    max_coords = np.argwhere(edt == max_val)

    # Uniformly choose one of them
    chosen_index = max_coords[np.random.choice(len(max_coords))]

    center = tuple(chosen_index)
    return center

# Compute the EDT with same shape as the image
def compute_edt(error_component):
    # Get bounding box of the largest error component to limit computation
    coords = np.argwhere(error_component)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1

    crop_shape = max_coords - min_coords

    # Compute padding (25% of crop size in each dimension)
    padding =  np.maximum((crop_shape * 0.25).astype(int), 1)


    # Define new padded shape
    padded_shape = crop_shape + 2 * padding

    # Create new empty array with padding
    center_crop = np.zeros(padded_shape, dtype=np.uint8)

    # Fill center region with actual cropped data
    center_crop[
        padding[0]:padding[0] + crop_shape[0],
        padding[1]:padding[1] + crop_shape[1],
        padding[2]:padding[2] + crop_shape[2]
    ] = error_component[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]

    large_roi = False
    if center_crop.shape[0] * center_crop.shape[1] * center_crop.shape[2] > 60000000:
        from skimage.measure import block_reduce
        print(f'ROI too large {center_crop.shape} --> 2x downsampling for EDT')
        center_crop = block_reduce(center_crop, block_size=(2, 2, 2), func=np.max)
        large_roi = True

    # Compute EDT on the padded array
    if torch.cuda.is_available() and not large_roi: # GPU available
        import cupy as cp
        from cucim.core.operations import morphology
        error_mask_cp = cp.array(center_crop)
        edt_cp = morphology.distance_transform_edt(error_mask_cp, return_distances=True)
        edt = cp.asnumpy(edt_cp)
    else: # CPU available only
        edt = distance_transform_edt(center_crop)
    
    if large_roi: # upsample
        edt = edt.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)

    # Crop out the center (remove padding)
    dist_cropped = edt[
        padding[0]:padding[0] + crop_shape[0],
        padding[1]:padding[1] + crop_shape[1],
        padding[2]:padding[2] + crop_shape[2]
    ]

    # Create full-sized EDT result array and splat back 
    dist_full = np.zeros_like(error_component, dtype=dist_cropped.dtype)
    dist_full[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ] = dist_cropped

    dist_transformed = dist_full

    return dist_transformed

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

if not os.path.exists(validation_gts_path):
    validation_gts_path = None
    print('[WARNING] Validation path does not exist for your GT data! Make sure you supplied the correct path or your .npz inputs have a gts key!')

input_temp = './inputs/'
output_temp = './outputs'
os.makedirs(save_path, exist_ok=True)

dockers = sorted(os.listdir(docker_path))
test_cases = sorted(os.listdir(test_img_path))

for docker in dockers:
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
        os.system('docker image load -i {}'.format(join(docker_path, docker)))
        team_outpath = join(save_path, teamname)
        if os.path.exists(team_outpath):
            shutil.rmtree(team_outpath)
        os.makedirs(team_outpath)
        os.system(f'chmod -R 777 ./* >/dev/null 2>&1') # ignore output warnings/errors of this command with >/dev/null 2>&1
        
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
        metric['DSC_1'] = []
        metric['DSC_2'] = []    
        metric['DSC_3'] = []
        metric['DSC_4'] = []
        metric['DSC_5'] = []
        metric['DSC_6'] = []
        metric['NSD_1'] = []
        metric['NSD_2'] = []
        metric['NSD_3'] = []
        metric['NSD_4'] = []
        metric['NSD_5'] = []
        metric['NSD_6'] = []
        metric['num_class'] = []
        metric['runtime_upperbound'] = []
        n_clicks = 5
        time_warning = False

        # To obtain the running time for each case, testing cases are inferred one-by-one
        for case in tqdm(test_cases):

            metric_temp = {}
            real_running_time = 0
            dscs = []
            nsds = []
            all_segs = []
            no_bbox = False

            # copy input image to accumulate clicks in its dict
            shutil.copy(join(test_img_path, case), input_temp)
            if validation_gts_path is  None: # for training images
                gts = patched_np_load(join(input_temp, case), allow_pickle=True)['gts']
            else: # for validation or test images --> gts are in separate files to avoid label leakage during the course of the challenge
                gts = patched_np_load(join(validation_gts_path, case), allow_pickle=True)['gts']
                
            unique_gts = np.sort(pd.unique(gts.ravel()))
            num_classes = len(unique_gts) - 1
            metric_temp['num_class'] = num_classes
            metric_temp['runtime_upperbound'] = num_classes * 90


            # foreground and background clicks for each class
            clicks_cls = [{'fg': [], 'bg': []} for _ in unique_gts[1:]] # skip background class 0 
            clicks_order = [[] for _ in unique_gts[1:]]
            if "boxes" in patched_np_load(join(input_temp, case), allow_pickle=True).keys():
                boxes = patched_np_load(join(input_temp, case), allow_pickle=True)['boxes']
            

            for it in range(n_clicks + 1): # + 1 due to bbox pred at iteration 0
                if it == 0:
                    if "boxes" not in patched_np_load(join(input_temp, case), allow_pickle=True).keys():
                        if verbose:
                            print(f'This sample does not use a Bounding Box for the initial iteration {it}') 
                        no_bbox = True
                        metric_temp["RunningTime_1"] = 0
                        metric_temp["DSC_1"] = 0
                        metric_temp["NSD_1"] = 0
                        dscs.append(0)
                        nsds.append(0)
                        continue
                    if verbose:
                        print(f'Using Bounding Box for iteration {it}') 
                else:
                    if verbose:
                        print(f'Using Clicks for iteration {it}')
                    if os.path.isfile(join(output_temp, case)):
                        segs = patched_np_load(join(output_temp, case), allow_pickle=True)['segs'].astype(np.uint8) # previous prediction
                    else:
                        segs = np.zeros_like(gts).astype(np.uint8) # in case the bbox prediction did not produce a result
                    all_segs.append(segs.astype(np.uint8))

                    # Refinement clicks
                    for ind, cls in enumerate(sorted(unique_gts[1:])):
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

                            edt = compute_edt(largest_component)
                            edt *= largest_component # make sure correct voxels have a distance of 0
                            if np.sum(edt) == 0: # no valid voxels to sample
                                if verbose:
                                    print("Error is extremely small --> Sampling uniformly instead of using EDT")
                                edt = largest_component # in case EDT is empty (due to artifacts in resizing, simply sample a random voxel from the component), happens only for extremely small errors

                            center = sample_coord(edt)

                            if gts_cls[center] == 0: # oversegmentation -> place background click
                                assert segs_cls[center] == 1
                                clicks_cls[ind]['bg'].append(list(center))
                                clicks_order[ind].append('bg')
                            else: # undersegmentation -> place foreground click
                                assert segs_cls[center] == 0
                                clicks_cls[ind]['fg'].append(list(center))
                                clicks_order[ind].append('fg')

                            assert largest_component[center] # click within error

                            if verbose:
                                print(f"Class {cls}: Largest error component center is at {center}")
                        else:
                            clicks_order[ind].append(None)
                            if verbose:
                                print(f"Class {cls}: No error connected components found. Prediction is perfect! No clicks were added.")
                    
                    # update model input with new click
                    input_img = patched_np_load(join(input_temp, case), allow_pickle=True)

                    if validation_gts_path is None:
                        if no_bbox:
                            np.savez_compressed(
                                join(input_temp, case),
                                imgs=input_img['imgs'],
                                gts=input_img['gts'], # only for training images
                                spacing=input_img['spacing'],
                                clicks=clicks_cls,
                                clicks_order=clicks_order, 
                                prev_pred=segs,
                            ) 
                        else:
                            np.savez_compressed(
                                join(input_temp, case),
                                imgs=input_img['imgs'],
                                gts=input_img['gts'], # only for training images
                                spacing=input_img['spacing'],
                                clicks=clicks_cls, 
                                clicks_order=clicks_order, 
                                prev_pred=segs,
                                boxes=boxes,
                            ) 
                    else:
                        if no_bbox:
                            np.savez_compressed(
                                join(input_temp, case),
                                imgs=input_img['imgs'],
                                spacing=input_img['spacing'],
                                clicks=clicks_cls, 
                                clicks_order=clicks_order, 
                                prev_pred=segs,
                            ) 
                        else:
                            np.savez_compressed(
                                join(input_temp, case),
                                imgs=input_img['imgs'],
                                spacing=input_img['spacing'],
                                clicks=clicks_cls, 
                                clicks_order=clicks_order, 
                                prev_pred=segs,
                                boxes=boxes,
                            ) 

                # Model inference on the current input
                if torch.cuda.is_available(): # GPU available
                    cmd = 'docker container run --gpus "device=0" -m 32G --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ blueyo0/{}:latest /bin/bash -c "sh predict.sh" '.format(teamname.replace('/', '_'), teamname)
                else:
                    cmd = 'docker container run -m 32G --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname.replace('/', '_'), teamname)
                if verbose:
                    print(teamname, ' docker command:', cmd, '\n', 'testing image name:', case)
                start_time = time.time()
                os.system(cmd)
                infer_time = time.time() - start_time
                real_running_time += infer_time # only add the inference time without the click generation time
                print(f"{case} finished! Inference time: {infer_time}")
                metric_temp[f"RunningTime_{it + 1}"] = infer_time

                if not os.path.isfile(join(output_temp, case)):
                    print(f"[WARNING] Failed / Skipped prediction for iteration {it}! Setting prediction to zeros...")
                    segs = np.zeros_like(gts).astype(np.uint8)
                else:
                    segs = patched_np_load(join(output_temp, case), allow_pickle=True)['segs']
                all_segs.append(segs.astype(np.uint8))

                dsc = compute_multi_class_dsc(gts, segs)
                # compute nsd
                if dsc > 0.2:
                    # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
                    nsd = compute_multi_class_nsd(gts, segs, patched_np_load(join(input_temp, case), allow_pickle=True)['spacing'])
                else:
                    nsd = 0.0 # Assume model performs poor on this sample
                dscs.append(dsc)
                nsds.append(nsd)
                metric_temp[f'DSC_{it + 1}'] = dsc
                metric_temp[f'NSD_{it + 1}'] = nsd
                print('Dice', dsc, 'NSD', nsd)
                seg_name = case


                # Copy temp prediction to the final folder
                try:
                    shutil.copy(join(output_temp, seg_name), join(team_outpath, seg_name))
                    segs = patched_np_load(join(team_outpath, seg_name), allow_pickle=True)['segs']
                    np.savez_compressed(
                        join(team_outpath, seg_name),
                        segs=segs,
                        all_segs=all_segs, # store all intermediate predictions
                    ) 
                except:
                    print(f"{join(output_temp, seg_name)}, {join(team_outpath, seg_name)}")
                    if os.path.exists(join(team_outpath, seg_name)):
                        os.remove(team_outpath, seg_name) # clean up cached files if model has failed
                    print("Final prediction could not be copied!")
            

            if real_running_time > 90 * (len(unique_gts) - 1):
                print("[WARNING] Your model seems to take more than 90 seconds per class during inference! The final test set will have a time constraint of 90s per class --> Make sure to optimize your approach!")
                time_warning = True
            # Compute interactive metrics
            dsc_auc = integrate.cumulative_trapezoid(np.array(dscs[-n_clicks:]), np.arange(n_clicks))[-1] # AUC is only over the point prompts since the bbox prompt is optional
            nsd_auc = integrate.cumulative_trapezoid(np.array(nsds[-n_clicks:]), np.arange(n_clicks))[-1] 
            dsc_final = dscs[-1]
            nsd_final = nsds[-1]
            if os.path.exists(join(team_outpath, seg_name)): # add to csv only if final prediction is successful
                for k, v in metric_temp.items():
                    metric[k].append(v)
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
        os.system("docker rmi {}:latest".format(teamname.split('_')[0]))
        shutil.rmtree(input_temp)
        shutil.rmtree(output_temp)
        if time_warning: # repeat warning at the end as well
            print("[WARNING] Your model seems to take more than 90 seconds per class during inference for some images! The final test set will have a time constraint of 90s per class --> Make sure to optimize your approach!")
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(f"Error processing {case} with docker {docker}. Skipping this docker.")