import copy
import os
import os.path as osp

import edt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio


def random_sample_next_click(prev_mask, gt_mask, method='random'):
    """
    Randomly sample one click from ground-truth mask and previous seg mask

    Arguements:
        prev_mask: (torch.Tensor) [H,W,D] previous mask that SAM-Med3D predict
        gt_mask: (torch.Tensor) [H,W,D] ground-truth mask for this image
    """
    def ensure_3D_data(roi_tensor):
        if roi_tensor.ndim != 3:
            roi_tensor = roi_tensor.squeeze()
        assert roi_tensor.ndim == 3, "Input tensor must be 3D"
        return roi_tensor

    prev_mask = ensure_3D_data(prev_mask)
    gt_mask = ensure_3D_data(gt_mask)

    prev_mask = prev_mask > 0
    true_masks = gt_mask > 0

    if not true_masks.any():
        raise ValueError("Cannot find true value in the ground-truth!")

    fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_mask))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), prev_mask)

    if method.lower() == 'random':
        to_point_mask = torch.logical_or(fn_masks, fp_masks)  # error region

        if not to_point_mask.any():
            all_points = torch.argwhere(true_masks)
            point = all_points[np.random.randint(len(all_points))]
            is_positive = True
        else:
            all_points = torch.argwhere(to_point_mask)
            point = all_points[np.random.randint(len(all_points))]
            is_positive = bool(fn_masks[point[0], point[1], point[2]])

        sampled_point = point.clone().detach().reshape(1, 1, 3)
        sampled_label = torch.tensor([[int(is_positive)]], dtype=torch.long)

        return sampled_point, sampled_label

    elif method.lower() == 'ritm':
        # Pad masks and compute EDT
        fn_mask_single = F.pad(fn_masks[None, None], (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)[0, 0]
        fp_mask_single = F.pad(fp_masks[None, None], (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)[0, 0]

        fn_mask_dt = torch.tensor(edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
        fp_mask_dt = torch.tensor(edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]

        fn_max_dist = torch.max(fn_mask_dt)
        fp_max_dist = torch.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        max_dist = max(fn_max_dist, fp_max_dist)

        to_point_mask = (dt > (max_dist / 2.0))
        all_points = torch.argwhere(to_point_mask)

        if len(all_points) == 0:
            # fallback: center of volume
            point = torch.tensor([gt_mask.shape[0] // 2, gt_mask.shape[1] // 2, gt_mask.shape[2] // 2])
            is_positive = False
        else:
            point = all_points[np.random.randint(len(all_points))]
            is_positive = bool(fn_masks[point[0], point[1], point[2]])

        sampled_point = point.clone().detach().reshape(1, 1, 3)
        sampled_label = torch.tensor([[int(is_positive)]], dtype=torch.long)

        return sampled_point, sampled_label

    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'ritm' or 'random'.")


def sam_model_infer(model,
                    roi_image,
                    roi_gt=None,
                    prompt_generator=random_sample_next_click,
                    prev_low_res_mask=None,
                    num_clicks=1): # Added num_clicks for iterative prompting if desired
    '''
    Inference for SAM-Med3D, inputs prompt points with its labels (positive/negative for each points)
    '''
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if roi_gt is not None and (roi_gt == 0).all() and num_clicks > 0:
        # If GT is empty, and we need clicks, result is likely empty.
        # SAM might still predict something with a central click, but let's return empty.
        print("Warning: roi_gt is empty. Prediction will be empty.")
        return np.zeros_like(roi_image.cpu().numpy().squeeze()), None # Return None for low_res_mask

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

        points_coords, points_labels = torch.zeros(1, 0, 3).to(device), torch.zeros(1,
                                                                                    0).to(device)
        new_points_co, new_points_la = torch.Tensor([[[64, 64, 64]]]).to(device), torch.Tensor(
            [[1]]).to(torch.int64)
        
        current_prev_mask_for_click_generation = torch.zeros_like(roi_image, device=device)[:,0,...] # Start with empty prev_mask for click
        
        if prev_low_res_mask is None: # Initialize low_res_mask for the decoder
             prev_low_res_mask = torch.zeros(1, 1, roi_image.shape[2] // 4,
                                             roi_image.shape[3] // 4,
                                             roi_image.shape[4] // 4, device=device, dtype=torch.float)


        for _ in range(num_clicks):
            if roi_gt is not None:
                new_points_co, new_points_la = prompt_generator(
                    current_prev_mask_for_click_generation.squeeze(0).cpu(), # Expects HWD tensor
                    roi_gt[0, 0].cpu() # Expects HWD tensor
                )
                new_points_co, new_points_la = new_points_co.to(device), new_points_la.to(device)
            else: # No GT, default to a central positive click for the first click
                if points_coords.shape[1] == 0: # Only for the very first click if no GT
                    center_z = roi_image.shape[2] // 2
                    center_y = roi_image.shape[3] // 2
                    center_x = roi_image.shape[4] // 2
                    new_points_co = torch.tensor([[[center_x, center_y, center_z]]], device=device, dtype=torch.float) # X,Y,Z for SAM points
                    new_points_la = torch.tensor([[1]], device=device, dtype=torch.int64)
                else: # Subsequent clicks without GT are problematic, break or use last mask
                    print("Warning: No ground truth for subsequent click generation.")
                    break 
            
            points_coords = torch.cat([points_coords, new_points_co], dim=1)
            points_labels = torch.cat([points_labels, new_points_la], dim=1)

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=[points_coords, points_labels],
                boxes=None,
                masks=prev_low_res_mask,
            )

            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            # Update prev_low_res_mask for next iteration's prompt encoder input
            prev_low_res_mask = low_res_masks.detach() 

            # For click generation, use the upscaled version of the current prediction
            current_prev_mask_for_click_generation = F.interpolate(low_res_masks,
                                   size=roi_image.shape[-3:],
                                   mode='trilinear',
                                   align_corners=False)
            current_prev_mask_for_click_generation = torch.sigmoid(current_prev_mask_for_click_generation) > 0.5


        # Final high-resolution mask from the last low_res_masks
        final_masks_hr = F.interpolate(low_res_masks, # Use the final low_res_masks
                                       size=roi_image.shape[-3:],
                                       mode='trilinear',
                                       align_corners=False)

    medsam_seg_prob = torch.sigmoid(final_masks_hr)
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)

    return medsam_seg_mask, low_res_masks.detach() # Return last low_res_mask as well


def read_arr_from_nifti(nii_path, get_meta_info=False):
    sitk_image = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(sitk_image) # Z, Y, X

    if not get_meta_info:
        return arr

    meta_info = {
        "sitk_image_object": sitk_image, # Store the object for easy access to all props
        "sitk_origin": sitk_image.GetOrigin(),
        "sitk_direction": sitk_image.GetDirection(),
        "sitk_spacing": sitk_image.GetSpacing(),
        "original_numpy_shape": arr.shape, # Store ZYX shape
    }
    return arr, meta_info


def get_roi_from_subject(subject_canonical, meta_info, crop_transform, norm_transform):
    """
    Applies CropOrPad and ZNormalization to the canonical subject.
    `subject_canonical` is assumed to be after tio.ToCanonical().
    """
    meta_info["canonical_subject_shape"] = subject_canonical.spatial_shape # D, H, W
    meta_info["canonical_subject_affine"] = subject_canonical.image.affine.copy()

    padding_params, cropping_params = crop_transform._compute_center_crop_or_pad(subject_canonical)
    subject_cropped = crop_transform(subject_canonical)
    
    meta_info["padding_params_functional"] = padding_params # (d_pad_neg, d_pad_pos, h_pad_neg, h_pad_pos, w_pad_neg, w_pad_pos) from tio
    meta_info["cropping_params_functional"] = cropping_params # (d_crop_neg, d_crop_pos, ... )
    meta_info["roi_subject_affine"] = subject_cropped.image.affine.copy()
    
    img3D_roi = subject_cropped.image.data.clone().detach()
    img3D_roi = norm_transform(img3D_roi.squeeze(dim=1)) # (N, C, W, H, D)
    img3D_roi = img3D_roi.unsqueeze(dim=1)

    gt3D_roi = subject_cropped.label.data.clone().detach()

    # make the roi image/label 5D tensor for torch inference
    def correct_roi_dim(roi_tensor): 
        if roi_tensor.ndim == 3:
            roi_tensor = roi_tensor.unsqueeze(0).unsqueeze(0)
        if roi_tensor.ndim == 4:
            roi_tensor = roi_tensor.unsqueeze(0)
        if img3D_roi.shape[0] != 1: # Ensure channel is 1
            roi_tensor = roi_tensor[:, 0:1,...]
        return roi_tensor
    
    img3D_roi = correct_roi_dim(img3D_roi)
    gt3D_roi = correct_roi_dim(gt3D_roi)

    return img3D_roi, gt3D_roi, meta_info

def get_subject_and_meta_info(img_path, gt_path):
    _, meta_info = read_arr_from_nifti(img_path, get_meta_info=True)
    subject = tio.Subject(
        image=tio.ScalarImage(img_path),
        label=tio.LabelMap(gt_path) 
    )
    return subject, meta_info


def data_preprocess(subject, meta_info, category_index, target_spacing, crop_size=128):
    # Make the label category-specific IN THE TIO SUBJECT
    # Ensure label data is integer for category comparison
    label_data_for_cat = subject.label.data.clone()
    new_label_data = torch.zeros_like(label_data_for_cat)
    new_label_data[label_data_for_cat == category_index] = 1
    subject.label.set_data(new_label_data) # Update label map data

    meta_info["original_subject_affine"] = subject.image.affine.copy()
    meta_info["original_subject_spatial_shape"] = subject.image.spatial_shape # D, H, W

    # step-1: resample online
    resampler = tio.Resample(target=target_spacing)
    subject_resampled = resampler(subject)

    # step-2: canonicalize
    transform_canonical = tio.ToCanonical()
    subject_canonical = transform_canonical(subject_resampled)

    # step-3: try to crop or pad roi region (with normalization)
    crop_transform = tio.CropOrPad(mask_name='label', target_shape=(crop_size, crop_size, crop_size))
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    roi_image, roi_label, meta_info = get_roi_from_subject(
        subject_canonical, meta_info, crop_transform, norm_transform
    )
    # roi image/label is (1, 1, D, H, W) after normalization
    return roi_image, roi_label, meta_info


def data_postprocess(roi_pred_numpy, meta_info):
    """
    Takes the ROI prediction (numpy array, expected to be in D, H, W order),
    and uses metadata to reconstruct it onto the original image's grid.
    The output numpy array will be in the order that matches SimpleITK's GetArrayFromImage (Z, Y, X),
    assuming TorchIO's D, H, W correspond to Z, Y, X.

    Args:
        roi_pred_numpy (np.ndarray): The prediction mask for the ROI, 
                                     typically with shape (D_roi, H_roi, W_roi).
                                     Values should be 0 or 1.
        meta_info (dict): A dictionary containing metadata from the preprocessing steps,
                          including:
                          - "roi_subject_affine": Affine matrix of the ROI.
                          - "original_subject_affine": Affine matrix of the original image
                            as loaded by tio.ScalarImage.
                          - "original_subject_spatial_shape": Spatial shape (D, H, W)
                            of the original image as loaded by tio.ScalarImage.
    Returns:
        np.ndarray: The prediction mask mapped to the original image's grid,
                    with shape matching the original image's (e.g., Z, Y, X)
                    and dtype uint8.
    """
    # Convert the NumPy ROI prediction to a PyTorch tensor.
    # Add a channel dimension to make it (1, D, H, W).
    # Ensure the dtype is suitable for tio.LabelMap; float32 is safe.
    roi_pred_tensor = torch.from_numpy(roi_pred_numpy.astype(np.float32)).unsqueeze(0)

    # Create a tio.LabelMap object for the ROI prediction.
    # The affine matrix for this map is meta_info["roi_subject_affine"],
    # which describes the orientation and position of the ROI in physical space.
    pred_label_map_roi_space = tio.LabelMap(
        tensor=roi_pred_tensor,
        affine=meta_info["roi_subject_affine"]
    )

    # Define the target grid for resampling using the original image's properties.
    # These properties were stored in meta_info before resampling and canonicalization.
    # "original_subject_spatial_shape" is (D, H, W).
    # "original_subject_affine" is the corresponding affine matrix.
    
    # Create a reference image that defines the target space for resampling.
    # The content of this image doesn't matter, only its geometry (shape, affine).
    # The shape for the reference tensor needs a channel dimension: (1, D, H, W).
    reference_tensor_shape = (1, *meta_info["original_subject_spatial_shape"]) 
    
    reference_image_original_space = tio.ScalarImage(
        tensor=torch.zeros(reference_tensor_shape), # Content is irrelevant
        affine=meta_info["original_subject_affine"]
    )

    # Initialize the resampler.
    # We want to resample pred_label_map_roi_space to the grid defined by
    # reference_image_original_space.
    # For segmentation masks (labels), 'nearest' interpolation is crucial to avoid
    # introducing new label values or averaging existing ones.
    resampler_to_original_grid = tio.Resample(
        target=reference_image_original_space,
        image_interpolation='nearest' 
    )
    
    # Perform the resampling.
    pred_resampled_to_original_space = resampler_to_original_grid(pred_label_map_roi_space)

    # Extract the data as a NumPy array.
    # The tensor from TorchIO will be (C, D, H, W). Squeeze the channel dimension.
    final_pred_numpy_dhw = pred_resampled_to_original_space.data.squeeze(0).cpu().numpy()

    # Ensure the final output is np.uint8, as the input roi_pred_numpy was (0 or 1).
    # This dtype is also typically expected for label masks.
    final_pred_numpy = final_pred_numpy_dhw.astype(np.uint8)

    # The shape of final_pred_numpy (D,H,W from TorchIO) is expected to align with
    # the Z,Y,X convention used for final_pred_numpy_original_grid in the calling function,
    # which is initialized based on the GT's SITK original_numpy_shape.
    # This relies on the assumption that the image and GT are spatially aligned and
    # TorchIO's D,H,W correspond directly to SITK's Z,Y,X for the original image.

    return final_pred_numpy.transpose(2, 1, 0) # Convert to ZYX order

def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info_for_saving):
    """Saves a NumPy array to NIFTI using SimpleITK, restoring original metadata."""
    # in_arr is expected to be in (Z, Y, X) order, matching sitk.GetArrayFromImage
    out_img = sitk.GetImageFromArray(in_arr)
    
    # Use metadata from the original SITK image object stored in meta_info
    original_sitk_image = meta_info_for_saving.get("sitk_image_object")
    if original_sitk_image:
        out_img.SetOrigin(original_sitk_image.GetOrigin())
        out_img.SetDirection(original_sitk_image.GetDirection())
        out_img.SetSpacing(original_sitk_image.GetSpacing())
    else: # Fallback to individual keys if object not stored
        out_img.SetOrigin(meta_info_for_saving["sitk_origin"])
        out_img.SetDirection(meta_info_for_saving["sitk_direction"])
        out_img.SetSpacing(meta_info_for_saving["sitk_spacing"])
        
    sitk.WriteImage(out_img, out_path)


def get_category_list_and_zero_mask(gt_path):
    # Use read_arr_from_nifti to also get original numpy shape for zero_mask
    arr, meta = read_arr_from_nifti(gt_path, get_meta_info=True)
    unique_label = np.unique(arr)
    unique_fg_labels = [int(l) for l in unique_label if l != 0]
    return unique_fg_labels, np.zeros(meta["original_numpy_shape"], dtype=np.uint8)


def validate_paired_img_gt(model, img_path, gt_path, output_path, num_clicks=1, crop_size=128, target_spacing=(1.5, 1.5, 1.5), seed=233):
    # Set seed hwithin the function
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    exist_categories, final_pred_numpy_original_grid = get_category_list_and_zero_mask(gt_path)
    _, gt_meta_for_saving = read_arr_from_nifti(gt_path, get_meta_info=True)
    subject, meta_info = get_subject_and_meta_info(img_path, gt_path)

    for category_index in exist_categories:
        category_specific_subject = copy.deepcopy(subject)
        category_specific_meta_info = copy.deepcopy(meta_info)
        # roi_image is (1,1,D,H,W), roi_label is (1,1,D,H,W)
        # meta_info contains all necessary affines and shapes
        roi_image, roi_label, meta_info = data_preprocess(category_specific_subject,
                                                          category_specific_meta_info,
                                                          category_index=category_index,
                                                          target_spacing=target_spacing,
                                                          crop_size=crop_size)

        roi_pred_numpy, _ = sam_model_infer(model, roi_image, roi_gt=roi_label, 
                                            num_clicks=num_clicks,
                                            prev_low_res_mask=None)

        cls_pred_original_grid = data_postprocess(roi_pred_numpy, meta_info)
        final_pred_numpy_original_grid[cls_pred_original_grid == 1] = category_index

    # Save the combined prediction which is on the original GT's grid
    save_numpy_to_nifti(final_pred_numpy_original_grid, output_path, gt_meta_for_saving)
