# -*- encoding: utf-8 -*-
'''
@File    :   infer_with_medim.py
@Time    :   2024/09/08 11:31:02
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   Example code for inference with MedIM
'''

import medim
import torch
import numpy as np
import torch.nn.functional as F
import torchio as tio
import os.path as osp
import os
from torchio.data.io import sitk_to_nib
import SimpleITK as sitk
from os.path import join
from glob import glob
from collections import defaultdict


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
                    prompt_generator=random_sample_next_click,
                    roi_gt=None,
                    prev_low_res_mask=None):
    '''
    Inference for SAM-Med3D, inputs prompt points with its labels (positive/negative for each points)

    # roi_image: (torch.Tensor) cropped image, shape [1,1,128,128,128]
    # prompt_points_and_labels: (Tuple(torch.Tensor, torch.Tensor))
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("using device", device)
    model = model.to(device)
    
    # import pdb; pdb.set_trace()

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
            prev_low_res_mask = F.interpolate(prev_low_res_mask,
                                              size=(roi_image.shape[2] // 4, roi_image.shape[3] // 4, roi_image.shape[4] // 4),
                                              mode='nearest').to(torch.float32)
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


def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info):
    # torchio turn 1xHxWxD -> DxWxH
    # so we need to squeeze and transpose back to HxWxD
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)


def resample_nii(imgs: np.array,
                 gts: np.array,
                 prev_seg: np.array,
                 target_spacing: tuple = (1.5, 1.5, 1.5),
                ):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    # Load the nii.gz file using torchio
    subject = tio.Subject(
                    image=tio.ScalarImage(tensor=imgs[None]), 
                    label=tio.LabelMap(tensor=gts[None]),
                    prev_seg=tio.LabelMap(tensor=prev_seg[None]),
                    )
    resampler = tio.Resample(target=target_spacing)
    resampled_subject = resampler(subject)
    return resampled_subject


def read_data_from_subject(subject):
    # sitk_image = sitk.ReadImage(img_path)
    # sitk_label = sitk.ReadImage(gt_path)

    # if sitk_image.GetOrigin() != sitk_label.GetOrigin():
    #     sitk_image.SetOrigin(sitk_label.GetOrigin())
    # if sitk_image.GetDirection() != sitk_label.GetDirection():
    #     sitk_image.SetDirection(sitk_label.GetDirection())

    # sitk_image_arr, _ = sitk_to_nib(sitk_image)
    # sitk_label_arr, _ = sitk_to_nib(sitk_label)

    # subject = tio.Subject(
    #     image=tio.ScalarImage(tensor=sitk_image_arr),
    #     label=tio.LabelMap(tensor=sitk_label_arr),
    # )
    # import pdb; pdb.set_trace()
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

    # import pdb; pdb.set_trace()
    img3D_roi, gt3D_roi = subject_roi.image.data.clone().detach().unsqueeze(
        1), subject_roi.label.data.clone().detach().unsqueeze(1)
    prev_seg3D_roi = subject_roi.prev_seg.data.clone().detach().unsqueeze(1)
    ori_roi_offset = (
        cropping_params[0],
        cropping_params[0] + 128 - padding_params[0] - padding_params[1],
        cropping_params[2],
        cropping_params[2] + 128 - padding_params[2] - padding_params[3],
        cropping_params[4],
        cropping_params[4] + 128 - padding_params[4] - padding_params[5],
    )

    meta_info = {
    #     "image_path": img_path,
    #     "origin": sitk_label.GetOrigin(),
    #     "direction": sitk_label.GetDirection(),
    #     "spacing": sitk_label.GetSpacing(),
        "padding_params": padding_params,
        "cropping_params": cropping_params,
        "ori_roi": ori_roi_offset,
    }
    return (
        img3D_roi,
        gt3D_roi,
        prev_seg3D_roi,
        meta_info,
    )


def data_preprocess(imgs, cls_gt, cls_prev_seg, orig_spacing, category_index):
    subject = resample_nii(imgs, cls_gt, cls_prev_seg, target_spacing=[t/o for o, t in zip(orig_spacing, [1.5, 1.5, 1.5])])
    roi_image, roi_label, roi_prev_seg, meta_info = read_data_from_subject(subject)
    
    meta_info["orig_shape"] = imgs.shape
    meta_info["resampled_shape"] = subject.spatial_shape,
    return roi_image, roi_label, roi_prev_seg, meta_info


def data_postprocess(roi_pred, meta_info, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    pred3D_full = np.zeros(*meta_info["resampled_shape"])
    padding_params = meta_info["padding_params"]
    unpadded_pred = roi_pred[padding_params[0] : 128-padding_params[1],
                             padding_params[2] : 128-padding_params[3],
                             padding_params[4] : 128-padding_params[5]]
    ori_roi = meta_info["ori_roi"]
    pred3D_full[ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3],
                ori_roi[4]:ori_roi[5]] = unpadded_pred

    # sitk_image = sitk.ReadImage(ori_img_path)
    # ori_meta_info = {
    #     "image_path": ori_img_path,
    #     "image_shape": sitk_image.GetSize(),
    #     "origin": sitk_image.GetOrigin(),
    #     "direction": sitk_image.GetDirection(),
    #     "spacing": sitk_image.GetSpacing(),
    # }
    pred3D_full_ori = F.interpolate(
        torch.Tensor(pred3D_full)[None][None],
        size=meta_info["orig_shape"],
        mode='nearest').cpu().numpy().squeeze()
    # save_numpy_to_nifti(pred3D_full_ori, output_path, meta_info)
    return pred3D_full_ori


def read_data_from_npz(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    imgs = data.get('imgs', None)
    gts = data.get('gts', None)

    sitk_spacing = data.get('spacing', None)
    # spacing = sitk_spacing
    spacing = [sitk_spacing[2], sitk_spacing[0], sitk_spacing[1]]

    # z-score normalize imgs
    imgs = imgs.astype(np.float32)

    # parsing boxes/clicks tensor, allow category to has more than 1 clicks
    all_clicks = defaultdict(list)
    # get bbox click first
    boxes = data.get('boxes', None)
    if (boxes is not None):
        for cls_idx, bbox in enumerate(boxes):
            all_clicks[cls_idx].append((
            ((bbox['z_min']+bbox['z_max'])/2, (bbox['z_mid_y_min']+bbox['z_mid_y_max'])/2, (bbox['z_mid_x_min']+bbox['z_mid_x_max'])/2,), # center of bbox
            [1], # positive click
            ))
        # all_clicks = [
        #     [
        #     #  ([bbox['z_mid'], (bbox['z_mid_y_min']+bbox['z_mid_y_max'])/2, (bbox['z_mid_x_min']+bbox['z_mid_x_max'])/2,], # center of bbox
        #      ([(bbox['z_min']+bbox['z_max'])/2, (bbox['z_mid_y_min']+bbox['z_mid_y_max'])/2, (bbox['z_mid_x_min']+bbox['z_mid_x_max'])/2,
        #     ], # center of bbox
        #     [1], # positive click
        #     )] for bbox in boxes
        # ] # [[(clickl_cls1, tag), (click2_cls1, tag), ...], []]

    # get point click then
    prev_pred = data.get('prev_pred', np.zeros_like(imgs, dtype=np.uint8))
    clicks = data.get('clicks', None)
    if (clicks is not None):
        for cls_idx, cls_click_dict in enumerate(clicks):
            for click in cls_click_dict['fg']:
            #     all_clicks[cls_idx].append(((click[2], click[1], click[0]), [1]))
                all_clicks[cls_idx].append((click, [1]))
            for click in cls_click_dict['bg']:
                all_clicks[cls_idx].append((click, [0]))
            # if len(cls_click_dict['fg'])<1 :
            #     center = np.unravel_index(np.argmax(prev_pred==(cls_idx+1)), prev_pred.shape)
            #     all_clicks[cls_idx].append((center, [1]))
        # print(all_clicks)

    # import pdb; pdb.set_trace()
    return imgs, spacing, all_clicks, prev_pred


def create_gt_arr(shape, point, category_index, square_size=20):
    # Create an empty array with the same shape as the input array
    gt_array = np.zeros(shape)
    
    # Extract the coordinates of the point
    z, y, x = point
    
    # Calculate the half size of the square
    half_size = square_size // 2
    
    # Calculate the coordinates of the square around the point
    z_min = max(int(z - half_size), 0)
    z_max = min(int(z + half_size) + 1, shape[0])
    y_min = max(int(y - half_size), 0)
    y_max = min(int(y + half_size) + 1, shape[1])
    x_min = max(int(x - half_size), 0)
    x_max = min(int(x + half_size) + 1, shape[2])
    
    # Set the values within the square to 1
    gt_array[z_min:z_max, y_min:y_max, x_min:x_max] = category_index
    
    return gt_array


if __name__ == "__main__":
    ''' 1. prepare the pre-trained model with local path or huggingface url '''
    # ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    # or you can use the local path like: 
    ckpt_path = "./ckpt/sam_med3d_turbo_bbox_cvpr.pth"
    model = medim.create_model("SAM-Med3D",
                               pretrained=True,
                               checkpoint_path=ckpt_path)

    ''' 2. read and pre-process your input data '''
    npz_file = glob("inputs/*.npz")[0]
    out_dir = "./outputs"
    imgs, spacing, all_clicks, prev_pred = read_data_from_npz(npz_file)
    final_pred = np.zeros_like(imgs, dtype=np.uint8)
    for idx, cls_clicks in all_clicks.items():
        category_index = idx + 1
        # import pdb; pdb.set_trace()
        pred_ori = prev_pred==category_index
        final_pred[pred_ori!=0] = category_index
        if (cls_clicks[-1][1][0] == 1):
            cls_gt = create_gt_arr(imgs.shape, cls_clicks[-1][0], category_index=category_index)
            # print(category_index, imgs.shape, spacing, cls_clicks, (cls_gt==category_index).sum())
            # continue
            cls_prev_seg = prev_pred==category_index
            roi_image, roi_label, roi_prev_seg, meta_info = data_preprocess(imgs, cls_gt, cls_prev_seg,
                                                            orig_spacing=spacing, 
                                                            category_index=category_index)

            # import pdb; pdb.set_trace()
            ''' 3. infer with the pre-trained SAM-Med3D model '''
            
            roi_pred = sam_model_infer(model, roi_image, roi_gt=roi_label, prev_low_res_mask=roi_prev_seg)

            # import pdb; pdb.set_trace()
            ''' 4. post-process and save the result '''
            pred_ori = data_postprocess(roi_pred, meta_info, out_dir)
            final_pred[pred_ori!=0] = category_index

    output_path = osp.join(out_dir, osp.basename(npz_file))
    np.savez_compressed(output_path, segs=final_pred)
    print("result saved to", output_path)
