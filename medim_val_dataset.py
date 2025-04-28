# -*- encoding: utf-8 -*-

import medim
import torch
import numpy as np
import torch.nn.functional as F
import torchio as tio
import os.path as osp
import os
import SimpleITK as sitk
from tqdm import tqdm
from glob import glob


def random_sample_next_click(prev_mask, gt_mask):
    """
    Randomly sample one click from ground-truth mask and previous seg mask

    Arguements:
        prev_mask: (torch.Tensor) [H,W,D] previous mask that SAM-Med3D predict
        gt_mask: (torch.Tensor) [H,W,D] ground-truth mask for this image
    """
    prev_mask = prev_mask > 0
    true_masks = gt_mask > 0

    if (not true_masks.any()):
        raise ValueError("Cannot find true value in the ground-truth!")

    fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_mask))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), prev_mask)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    all_points = torch.argwhere(to_point_mask)
    point = all_points[np.random.randint(len(all_points))]

    if fn_masks[point[0], point[1], point[2]]:
        is_positive = True
    else:
        is_positive = False

    sampled_point = point.clone().detach().reshape(1, 1, 3)
    sampled_label = torch.tensor([
        int(is_positive),
    ]).reshape(1, 1)

    return sampled_point, sampled_label


def sam_model_infer(model,
                    roi_image,
                    roi_gt=None,
                    prompt_generator=random_sample_next_click,
                    prev_low_res_mask=None):
    '''
    Inference for SAM-Med3D, inputs prompt points with its labels (positive/negative for each points)

    # roi_image: (torch.Tensor) cropped image, shape [1,1,128,128,128]
    # prompt_points_and_labels: (Tuple(torch.Tensor, torch.Tensor))
    '''
    
    if roi_gt is not None and (roi_gt==0).all():
        return torch.zeros_like(roi_image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("using device", device)
    model = model.to(device)

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

        points_coords, points_labels = torch.zeros(1, 0,
                                                   3).to(device), torch.zeros(
                                                       1, 0).to(device)
        new_points_co, new_points_la = torch.Tensor(
            [[[64, 64, 64]]]).to(device), torch.Tensor([[1]]).to(torch.int64)
        if (roi_gt is not None):
            prev_low_res_mask = prev_low_res_mask if (
                prev_low_res_mask is not None) else torch.zeros(
                    1, 1, roi_image.shape[2] // 4, roi_image.shape[3] //
                    4, roi_image.shape[4] // 4)
            new_points_co, new_points_la = prompt_generator(
                torch.zeros_like(roi_image)[0, 0], roi_gt[0, 0])
            new_points_co, new_points_la = new_points_co.to(
                device), new_points_la.to(device)
        points_coords = torch.cat([points_coords, new_points_co], dim=1)
        points_labels = torch.cat([points_labels, new_points_la], dim=1)

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[points_coords, points_labels],
            boxes=None,  # we currently not support bbox prompt
            masks=prev_low_res_mask.to(device),
            # masks=None,
        )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,  # (1, 384, 8, 8, 8)
            image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 384, 8, 8, 8)
            sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 384)
            dense_prompt_embeddings=dense_embeddings,  # (1, 384, 8, 8, 8)
        )

        prev_mask = F.interpolate(low_res_masks,
                                  size=roi_image.shape[-3:],
                                  mode='trilinear',
                                  align_corners=False)

    # convert prob to mask
    medsam_seg_prob = torch.sigmoid(prev_mask)  # (1, 1, 64, 64, 64)
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)

    return medsam_seg_mask

def read_and_resample_nifti(img: str,
                            cls_gt: str,
                            meta_info: dict,
                            target_spacing: tuple = (1.5, 1.5, 1.5)):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    # Load the nii.gz file using torchio
    subject = tio.Subject(
                    image=tio.ScalarImage(tensor=img[None]), 
                    label=tio.LabelMap(tensor=cls_gt[None]),
                )
    resampler = tio.Resample(target=target_spacing)
    resampled_subject = resampler(subject)

    meta_info["orig_image_shape"] = subject.image.shape[1:]
    meta_info["resampled_image_shape"] = resampled_subject.image.shape[1:]

    return resampled_subject, meta_info


def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info):
    # torchio turn 1xHxWxD -> DxWxH
    # so we need to squeeze and transpose back to HxWxD
    # ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(in_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["sitk_origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["sitk_direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["sitk_spacing"]))
    sitk.WriteImage(out, out_path)


def get_roi_from_subject(subject, meta_info):
    crop_transform = tio.CropOrPad(mask_name='label',
                                   target_shape=(128, 128, 128))
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(
        subject)
    if (cropping_params is None): cropping_params = (0, 0, 0, 0, 0, 0)
    if (padding_params is None): padding_params = (0, 0, 0, 0, 0, 0)

    infer_transform = tio.Compose([
        crop_transform,
        tio.ZNormalization(masking_method=lambda x: x > 0),
    ])
    subject_roi = infer_transform(subject)

    img3D_roi, gt3D_roi = subject_roi.image.data.clone().detach().unsqueeze(
        1), subject_roi.label.data.clone().detach().unsqueeze(1)
    ori_roi_offset = (
        cropping_params[0],
        cropping_params[0] + 128 - padding_params[0] - padding_params[1],
        cropping_params[2],
        cropping_params[2] + 128 - padding_params[2] - padding_params[3],
        cropping_params[4],
        cropping_params[4] + 128 - padding_params[4] - padding_params[5],
    )

    meta_info["padding_params"] = padding_params
    meta_info["cropping_params"] = cropping_params
    meta_info["ori_roi"] = ori_roi_offset

    return img3D_roi, gt3D_roi, meta_info,

def read_arr_from_nifti(nii_path, get_meta_info=False):
    sitk_image = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(sitk_image)

    if not get_meta_info:
        return arr

    sitk_spacing = sitk_image.GetSpacing()
    spacing = [sitk_spacing[2], sitk_spacing[0], sitk_spacing[1]]
    meta_info = {
        "sitk_origin": sitk_image.GetOrigin(),
        "sitk_direction": sitk_image.GetDirection(),
        "sitk_spacing": sitk_image.GetSpacing(),
        "spacing": spacing,
    }
        
    return arr, meta_info
    

def data_preprocess(img_path, gt_path, category_index):
    full_img, meta_info = read_arr_from_nifti(img_path, get_meta_info=True)

    full_gt = read_arr_from_nifti(gt_path)
    cls_gt = np.zeros_like(full_gt).astype(np.uint8)
    cls_gt[full_gt==category_index] = 1

    subject, meta_info = read_and_resample_nifti(full_img, cls_gt, meta_info, target_spacing=[t/o for o, t in zip(meta_info["spacing"], [1.5, 1.5, 1.5])])
    roi_image, roi_label, meta_info = get_roi_from_subject(subject, meta_info)
    return roi_image, roi_label, meta_info


def data_postprocess(roi_pred, meta_info):
    pred3D_full = np.zeros(meta_info["resampled_image_shape"])
    padding_params = meta_info["padding_params"]
    unpadded_pred = roi_pred[padding_params[0] : 128-padding_params[1],
                             padding_params[2] : 128-padding_params[3],
                             padding_params[4] : 128-padding_params[5]]
    ori_roi = meta_info["ori_roi"]
    pred3D_full[ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3],
                ori_roi[4]:ori_roi[5]] = unpadded_pred

    pred3D_full_ori = F.interpolate(
        torch.Tensor(pred3D_full)[None][None],
        size=meta_info["orig_image_shape"],
        mode='nearest').cpu().numpy().squeeze()
    return pred3D_full_ori


def get_category_list_and_zero_mask(gt_path):
    img = sitk.ReadImage(gt_path)
    arr = sitk.GetArrayFromImage(img)
    unique_label = np.unique(arr)
    unique_fg_labels = [l for l in unique_label if l!=0]
    return unique_fg_labels, np.zeros_like(arr)


def validate_paired_img_gt(img_path, gt_path, output_path):
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    exist_categories, final_pred = get_category_list_and_zero_mask(gt_path)
    # for category_index in tqdm(exist_categories, desc=f"infer {len(exist_categories)} categories"):
    for category_index in exist_categories:
        roi_image, roi_label, meta_info = data_preprocess(img_path, gt_path, category_index=category_index)
        
        roi_pred = sam_model_infer(model, roi_image, roi_gt=roi_label)

        cls_pred = data_postprocess(roi_pred, meta_info)
        final_pred[cls_pred!=0] = category_index

    save_numpy_to_nifti(final_pred, output_path, meta_info)
    # print("result saved to", output_path)


if __name__ == "__main__":
    ''' prepare the pre-trained model with local path or huggingface url '''
    ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    # or you can use a local path like: 
    # ckpt_path = "./ckpt/sam_med3d_turbo.pth"
    model = medim.create_model("SAM-Med3D",
                               pretrained=True,
                               checkpoint_path=ckpt_path)

    test_data_list = [
        dict(
            img_dir="./test_data/Seg_Exps/ACDC/ACDC_test_cases",
            gt_dir="./test_data/Seg_Exps/ACDC/ACDC_test_gts",
            out_dir="./test_data/Seg_Exps/ACDC_test_SAM_Med3D",
        ),
    ]
    for test_data in test_data_list:
        gt_fname_list = sorted(glob(osp.join(test_data["gt_dir"], "*.nii.gz")))
        for gt_fname in tqdm(gt_fname_list):
            # print(gt_fname)
            case_name = osp.basename(gt_fname).replace(".nii.gz", "")
            img_path = osp.join(test_data["img_dir"], f"{case_name}_0000.nii.gz")
            if test_data["img_dir"].endswith("CT"):
                img_path = osp.join(test_data["img_dir"], f"{case_name}_0001.nii.gz")
            gt_path = gt_fname
            out_path = osp.join(test_data["out_dir"], f"{case_name}.nii.gz")
            validate_paired_img_gt(img_path, gt_path, out_path)
