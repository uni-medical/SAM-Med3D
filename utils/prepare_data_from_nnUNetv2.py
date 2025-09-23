# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data_from_nnUNet.py
@Time    :   2023/12/10 23:07:39
@Author  :   Haoyu Wang
@Contact :   small_dark@sina.com
@Brief   :   pre-process nnUNet-style dataset into SAM-Med3D-style
'''

import json
import os
import os.path as osp
import shutil

import nibabel as nib
import torchio as tio
from tqdm import tqdm
from pathlib import Path

def resample_nii(
    input_path: str,
    output_path: str,
    target_spacing: tuple = (1.5, 1.5, 1.5),
    n=None,
    reference_image=None,
    mode="linear"):
    """
    Resamples a nii.gz file to a specified spacing using torchio.
    """

    # Load the nii.gz file using torchio
    subject = tio.Subject(img=tio.ScalarImage(input_path))
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)

    # This will be the variable we finally save
    image_to_save = None

    if n is not None:
        image = resampled_subject.img
        tensor_data = image.data.clone() # Use .clone() to avoid modifying the original tensor

        if isinstance(n, int):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1

        # Create a temporary torchio image object before cropping/padding
        temp_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        
        # Apply the crop/pad operation
        reference_size = reference_image.shape[1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)
        final_image_object = cropper_or_padder(temp_image)

        # --- THE FIX ---
        # 1. Get the tensor from the FINAL object
        final_tensor = final_image_object.data

        # 2. Ensure its dtype is float
        final_tensor_float = final_tensor.float()

        # 3. Create the clean, final ScalarImage object that we will save
        image_to_save = tio.ScalarImage(
            tensor=final_tensor_float,
            affine=final_image_object.affine,
        )
    else:
        # If no modifications, the image to save is just the resampled one
        image_to_save = resampled_subject.img

    # Save the final, correct torchio object
    if image_to_save is not None:
        image_to_save.save(output_path)
    else:
        print("Warning: No image was processed to be saved.")


dataset_root = "/home/sagemaker-user/data/nnUNet_raw"
dataset_list = [
    'Dataset538_OASIS_20',
]
target_dir = "./data/OASIS_20"

for dataset in dataset_list:
    dataset_dir = osp.join(dataset_root, dataset)
    meta_info = json.load(open(osp.join(dataset_dir, "dataset.json")))

    print(F"Dataset name: {dataset}")
    print(meta_info['channel_names'])
    num_classes = len(meta_info["labels"]) - 1
    print("num_classes:", num_classes, meta_info["labels"])
    resample_dir = osp.join(dataset_dir, "imagesTr_1.5")
    os.makedirs(resample_dir, exist_ok=True)
    for cls_name, idx in meta_info["labels"].items():
        cls_name = cls_name.replace(" ", "_")
        idx = int(idx)
        dataset_name = dataset.split("_", maxsplit=1)[1]
        target_cls_dir = osp.join(target_dir, cls_name, dataset_name)
        target_img_dir = Path(target_cls_dir) / "imagesTr"
        target_gt_dir = Path(target_cls_dir) / "labelsTr"
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_gt_dir, exist_ok=True)
        dataset_dir = Path(dataset_dir)
        source_img_dir = dataset_dir / "imagesTr"
        source_gt_dir = dataset_dir / "labelsTr"

        for img, gt in tqdm(
            zip(source_img_dir.iterdir(), source_gt_dir.iterdir()), 
            desc=f"{dataset_name}-{cls_name}",
            total=meta_info["numTraining"]):

            resample_img = osp.join(resample_dir, img.name)
            if (not osp.exists(resample_img)):
                resample_nii(img, resample_img)
            img = resample_img

            target_img_path = osp.join(
                target_img_dir,
                osp.basename(img).replace("_0000.nii.gz", ".nii.gz"))
            target_gt_path = osp.join(
                target_gt_dir,
                osp.basename(gt).replace("_0000.nii.gz", ".nii.gz"))

            gt_img = nib.load(gt)
            spacing = tuple(gt_img.header['pixdim'][1:4])
            spacing_voxel = spacing[0] * spacing[1] * spacing[2]
            gt_arr = gt_img.get_fdata()
            gt_arr[gt_arr != idx] = 0
            gt_arr[gt_arr != 0] = 1
            volume = gt_arr.sum() * spacing_voxel
            if (volume < 10):
                print("skip", target_img_path)
                continue

            reference_image = tio.ScalarImage(img)
            if (dataset_name == "kits23" and idx == 1):
                resample_nii(
                    gt,
                    target_gt_path,
                    n=[1, 2, 3],
                    reference_image=reference_image,
                    mode="nearest")
            else:
                resample_nii(
                    gt,
                    target_gt_path,
                    n=idx,
                    reference_image=reference_image,
                    mode="nearest")
            shutil.copy(img, target_img_path)
