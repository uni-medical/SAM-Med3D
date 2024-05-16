"""
Run inference without label masks. Based on inference.py, and requires new click methods 
from updated utils/click_method.py. Check the new click method details for more information.

Author: Karson Chrispens
Date: 5/15/2024
"""

import os
import os.path as osp

join = osp.join
import argparse
import json
import pickle
from collections import OrderedDict, defaultdict
from glob import glob
from itertools import product

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from utils.click_method import (
    get_next_click3D_torch_no_gt_naive,
    get_next_click3D_torch_no_gt,
)
from utils.data_loader import Dataset_Union_ALL_Infer

parser = argparse.ArgumentParser()
parser.add_argument("-tdp", "--test_data_path", type=str, default="./data/validation")
parser.add_argument(
    "-cp", "--checkpoint_path", type=str, default="./ckpt/sam_med3d.pth"
)
parser.add_argument("--output_dir", type=str, default="./visualization")
parser.add_argument("--task_name", type=str, default="test_amos")
parser.add_argument("--skip_existing_pred", action="store_true", default=False)
parser.add_argument("--save_image", action="store_true", default=True)
parser.add_argument("--sliding_window", action="store_true", default=False)

parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--crop_size", type=int, default=128)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("-mt", "--model_type", type=str, default="vit_b_ori")
parser.add_argument("-nc", "--num_clicks", type=int, default=5)
parser.add_argument("-pm", "--point_method", type=str, default="no_gt")
parser.add_argument("-dt", "--data_type", type=str, default="infer")

parser.add_argument("--threshold", type=int, default=0)
parser.add_argument("--dim", type=int, default=3)
parser.add_argument("--split_idx", type=int, default=0)
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--ft2d", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=2023)

args = parser.parse_args()

""" parse and output_dir and task_name """
args.output_dir = join(args.output_dir, args.task_name)
args.pred_output_dir = join(args.output_dir, "pred")
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.pred_output_dir, exist_ok=True)
args.save_name = join(args.output_dir, "dice.py")
print("output_dir set to", args.output_dir)

SEED = args.seed
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.init()

click_methods = {
    "no_gt": get_next_click3D_torch_no_gt,
    "no_gt_naive": get_next_click3D_torch_no_gt_naive,
}


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    if args.ft2d and ori_h < image_size and ori_w < image_size:
        top = (image_size - ori_h) // 2
        left = (image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        pad = None
    return masks, pad


def sam_decoder_inference(
    target_size,
    points_coords,
    points_labels,
    model,
    image_embeddings,
    mask_inputs=None,
    multimask=False,
):
    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(points_coords.to(model.device), points_labels.to(model.device)),
            boxes=None,
            masks=mask_inputs,
        )

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )

    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i : i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(
        low_res_masks,
        (target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return masks, low_res_masks, iou_predictions


def repixel_value(arr, is_seg=False):
    if not is_seg:
        min_val = arr.min()
        max_val = arr.max()
        new_arr = (arr - min_val) / (max_val - min_val + 1e-10) * 255.0
    return new_arr


def random_point_sampling(mask, get_point=1):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor(
            [label], dtype=torch.int
        )
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(
            coords[indices], dtype=torch.float
        ), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels


def finetune_model_predict2D(
    img3D,
    gt3D,
    sam_model_tune,
    target_size=256,
    click_method="no_gt",
    device="cuda",
    num_clicks=1,
    prev_masks=None,
):
    pred_list = []

    slice_mask_list = defaultdict(list)

    img3D = torch.repeat_interleave(
        img3D, repeats=3, dim=1
    )  # 1 channel -> 3 channel (align to RGB)

    click_points = []
    click_labels = []
    for slice_idx in tqdm(range(img3D.size(-1)), desc="transverse slices", leave=False):
        img2D, gt2D = repixel_value(img3D[..., slice_idx]), gt3D[..., slice_idx]

        if (gt2D == 0).all():
            empty_result = torch.zeros(list(gt3D.size()[:-1]) + [1]).to(device)
            for iter in range(num_clicks):
                slice_mask_list[iter].append(empty_result)
            continue

        img2D = F.interpolate(
            img2D, (target_size, target_size), mode="bilinear", align_corners=False
        )
        gt2D = F.interpolate(
            gt2D.float(), (target_size, target_size), mode="nearest"
        ).int()

        img2D, gt2D = img2D.to(device), gt2D.to(device)
        img2D = (img2D - img2D.mean()) / img2D.std()

        with torch.no_grad():
            image_embeddings = sam_model_tune.image_encoder(img2D.float())

        points_co, points_la = torch.zeros(1, 0, 2).to(device), torch.zeros(1, 0).to(
            device
        )
        low_res_masks = None
        gt_semantic_seg = gt2D[0, 0].to(device)
        true_masks = gt_semantic_seg > 0
        for iter in range(num_clicks):
            if low_res_masks == None:
                pred_masks = torch.zeros_like(true_masks).to(device)
            else:
                pred_masks = (prev_masks[0, 0] > 0.0).to(device)
            fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
            fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)
            mask_to_sample = torch.logical_or(fn_masks, fp_masks)
            new_points_co, _ = random_point_sampling(mask_to_sample.cpu(), get_point=1)
            new_points_la = (
                torch.Tensor([1]).to(torch.int64)
                if (true_masks[new_points_co[0, 1].int(), new_points_co[0, 0].int()])
                else torch.Tensor([0]).to(torch.int64)
            )
            new_points_co, new_points_la = new_points_co[None].to(
                device
            ), new_points_la[None].to(device)
            points_co = torch.cat([points_co, new_points_co], dim=1)
            points_la = torch.cat([points_la, new_points_la], dim=1)
            prev_masks, low_res_masks, iou_predictions = sam_decoder_inference(
                target_size,
                points_co,
                points_la,
                sam_model_tune,
                image_embeddings,
                mask_inputs=low_res_masks,
                multimask=True,
            )
            click_points.append(new_points_co)
            click_labels.append(new_points_la)

            slice_mask, _ = postprocess_masks(
                low_res_masks, target_size, (gt3D.size(2), gt3D.size(3))
            )
            slice_mask_list[iter].append(
                slice_mask[..., None]
            )  # append (B, C, H, W, 1)

    for iter in range(num_clicks):
        medsam_seg = torch.cat(slice_mask_list[iter], dim=-1).cpu().numpy().squeeze()
        medsam_seg = medsam_seg > sam_model_tune.mask_threshold
        medsam_seg = medsam_seg.astype(np.uint8)

        pred_list.append(medsam_seg)

    return pred_list, click_points, click_labels


def finetune_model_predict3D(
    img3D,
    sam_model_tune,
    device="cuda",
    click_method="no_gt",
    num_clicks=10,
    prev_masks=None,
):
    img3D = norm_transform(img3D.squeeze(dim=1))  # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)

    click_points = []
    click_labels = []

    pred_list = []

    if prev_masks is None:
        prev_masks = torch.zeros_like(img3D).to(device)
    low_res_masks = F.interpolate(
        prev_masks.float(),
        size=(args.crop_size // 4, args.crop_size // 4, args.crop_size // 4),
    )

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(
            img3D.to(device)
        )  # (1, 384, 16, 16, 16)

    for click_idx in range(num_clicks):
        with torch.no_grad():

            batch_points, batch_labels = click_methods[click_method](
                prev_masks.to(device), img3D.to(device), 170
            )  # default threshold is 170, showing that here

            points_co = torch.cat(batch_points, dim=0).to(device)
            points_la = torch.cat(batch_labels, dim=0).to(device)

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,
                masks=low_res_masks.to(device),
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),  # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings,  # (B, 384, 64, 64, 64)
                multimask_output=False,
            )
            prev_masks = F.interpolate(
                low_res_masks,
                size=img3D.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)

    return pred_list, click_points, click_labels


# TODO: check if this works?
def pad_and_crop_with_sliding_window(img3D, crop_transform, offset_mode="center"):
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=img3D.squeeze(0)),
    )
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)
    # cropping_params: (x_start, x_max-(x_start+roi_size), y_start, ...)
    # padding_params: (x_left_pad, x_right_pad, y_left_pad, ...)
    if cropping_params is None:
        cropping_params = (0, 0, 0, 0, 0, 0)
    if padding_params is None:
        padding_params = (0, 0, 0, 0, 0, 0)
    roi_shape = crop_transform.target_shape
    vol_bound = (0, img3D.shape[2], 0, img3D.shape[3], 0, img3D.shape[4])
    center_oob_ori_roi = (
        cropping_params[0] - padding_params[0],
        cropping_params[0] + roi_shape[0] - padding_params[0],
        cropping_params[2] - padding_params[2],
        cropping_params[2] + roi_shape[1] - padding_params[2],
        cropping_params[4] - padding_params[4],
        cropping_params[4] + roi_shape[2] - padding_params[4],
    )
    window_list = []
    offset_dict = {
        "rounded": list(product((-32, +32, 0), repeat=3)),
        "center": [(0, 0, 0)],
    }
    for offset in offset_dict[offset_mode]:
        # get the position in original volume~(allow out-of-bound) for current offset
        oob_ori_roi = (
            center_oob_ori_roi[0] + offset[0],
            center_oob_ori_roi[1] + offset[0],
            center_oob_ori_roi[2] + offset[1],
            center_oob_ori_roi[3] + offset[1],
            center_oob_ori_roi[4] + offset[2],
            center_oob_ori_roi[5] + offset[2],
        )
        # get corresponing padding params based on `vol_bound`
        padding_params = [0 for i in range(6)]
        for idx, (ori_pos, bound) in enumerate(zip(oob_ori_roi, vol_bound)):
            pad_val = 0
            if idx % 2 == 0 and ori_pos < bound:  # left bound
                pad_val = bound - ori_pos
            if idx % 2 == 1 and ori_pos > bound:
                pad_val = ori_pos - bound
            padding_params[idx] = pad_val
        # get corresponding crop params after padding
        cropping_params = (
            oob_ori_roi[0] + padding_params[0],
            vol_bound[1] - oob_ori_roi[1] + padding_params[1],
            oob_ori_roi[2] + padding_params[2],
            vol_bound[3] - oob_ori_roi[3] + padding_params[3],
            oob_ori_roi[4] + padding_params[4],
            vol_bound[5] - oob_ori_roi[5] + padding_params[5],
        )
        # pad and crop for the original subject
        pad_and_crop = tio.Compose(
            [
                tio.Pad(padding_params, padding_mode=crop_transform.padding_mode),
                tio.Crop(cropping_params),
            ]
        )
        subject_roi = pad_and_crop(subject)
        img3D_roi = subject_roi.image.data.clone().detach().unsqueeze(1)

        # collect all position information, and set correct roi for sliding-windows in
        # todo: get correct roi window of half because of the sliding
        windows_clip = [0 for i in range(6)]
        for i in range(3):
            if offset[i] < 0:
                windows_clip[2 * i] = 0
                windows_clip[2 * i + 1] = -(roi_shape[i] + offset[i])
            elif offset[i] > 0:
                windows_clip[2 * i] = roi_shape[i] - offset[i]
                windows_clip[2 * i + 1] = 0
        pos3D_roi = dict(
            padding_params=padding_params,
            cropping_params=cropping_params,
            ori_roi=(
                cropping_params[0] + windows_clip[0],
                cropping_params[0]
                + roi_shape[0]
                - padding_params[0]
                - padding_params[1]
                + windows_clip[1],
                cropping_params[2] + windows_clip[2],
                cropping_params[2]
                + roi_shape[1]
                - padding_params[2]
                - padding_params[3]
                + windows_clip[3],
                cropping_params[4] + windows_clip[4],
                cropping_params[4]
                + roi_shape[2]
                - padding_params[4]
                - padding_params[5]
                + windows_clip[5],
            ),
            pred_roi=(
                padding_params[0] + windows_clip[0],
                roi_shape[0] - padding_params[1] + windows_clip[1],
                padding_params[2] + windows_clip[2],
                roi_shape[1] - padding_params[3] + windows_clip[3],
                padding_params[4] + windows_clip[4],
                roi_shape[2] - padding_params[5] + windows_clip[5],
            ),
        )
        pred_roi = pos3D_roi["pred_roi"]

        # if((gt3D_roi[pred_roi[0]:pred_roi[1],pred_roi[2]:pred_roi[3],pred_roi[4]:pred_roi[5]]==0).all()):
        # print("skip empty window with offset", offset)
        #    continue

        window_list.append((img3D_roi, pos3D_roi))
    return window_list


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


if __name__ == "__main__":
    all_dataset_paths = glob(join(args.test_data_path, "*", "*"))
    all_dataset_paths = list(filter(osp.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")

    crop_transform = tio.CropOrPad(
        target_shape=(args.crop_size, args.crop_size, args.crop_size)
    )

    infer_transform = [
        tio.ToCanonical(),
    ]

    test_dataset = Dataset_Union_ALL_Infer(
        paths=all_dataset_paths,
        data_type=args.data_type,
        transform=tio.Compose(infer_transform),
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
        get_all_meta_info=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, sampler=None, batch_size=1, shuffle=True
    )

    checkpoint_path = args.checkpoint_path

    device = args.device
    print("device:", device)

    if args.dim == 3:
        sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(
            device
        )
        if checkpoint_path is not None:
            model_dict = torch.load(checkpoint_path, map_location=device)
            state_dict = model_dict["model_state_dict"]
            sam_model_tune.load_state_dict(state_dict)
    else:
        raise NotImplementedError(
            "this scipts is designed for 3D sliding-window inference, not support other dims"
        )

    sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    for batch_data in tqdm(test_dataloader):
        image3D, meta_info = batch_data
        img_name = meta_info["image_path"][0]

        modality = osp.basename(osp.dirname(osp.dirname(osp.dirname(img_name))))
        dataset = osp.basename(osp.dirname(osp.dirname(img_name)))
        vis_root = osp.join(args.pred_output_dir, modality, dataset)
        pred_path = osp.join(
            vis_root,
            osp.basename(img_name).replace(
                ".nii.gz", f"_pred{args.num_clicks-1}.nii.gz"
            ),
        )

        """ inference """
        if args.skip_existing_pred and osp.exists(pred_path):
            pass  # if the pred existed, skip the inference
        else:
            image3D_full = image3D
            pred3D_full_dict = {
                click_idx: torch.zeros_like(image3D_full).numpy()
                for click_idx in range(args.num_clicks)
            }
            offset_mode = "center" if (not args.sliding_window) else "rounded"
            sliding_window_list = pad_and_crop_with_sliding_window(
                image3D_full, crop_transform, offset_mode=offset_mode
            )
            for image3D, pos3D in sliding_window_list:
                seg_mask_list, points, labels = finetune_model_predict3D(
                    image3D,
                    sam_model_tune,
                    device=device,
                    click_method=args.point_method,
                    num_clicks=args.num_clicks,
                    prev_masks=None,
                )
                ori_roi, pred_roi = pos3D["ori_roi"], pos3D["pred_roi"]
                for idx, seg_mask in enumerate(seg_mask_list):
                    seg_mask_roi = seg_mask[
                        ...,
                        pred_roi[0] : pred_roi[1],
                        pred_roi[2] : pred_roi[3],
                        pred_roi[4] : pred_roi[5],
                    ]
                    pred3D_full_dict[idx][
                        ...,
                        ori_roi[0] : ori_roi[1],
                        ori_roi[2] : ori_roi[3],
                        ori_roi[4] : ori_roi[5],
                    ] = seg_mask_roi

            os.makedirs(vis_root, exist_ok=True)
            padding_params = sliding_window_list[-1][-1]["padding_params"]
            cropping_params = sliding_window_list[-1][-1]["cropping_params"]
            # print(padding_params, cropping_params)
            point_offset = np.array(
                [
                    cropping_params[0] - padding_params[0],
                    cropping_params[2] - padding_params[2],
                    cropping_params[4] - padding_params[4],
                ]
            )
            points = [p.cpu().numpy() + point_offset for p in points]
            labels = [l.cpu().numpy() for l in labels]
            pt_info = dict(points=points, labels=labels)
            # print("save to", osp.join(vis_root, osp.basename(img_name).replace(".nii.gz", "_pred.nii.gz")))
            pt_path = osp.join(
                vis_root, osp.basename(img_name).replace(".nii.gz", "_pt.pkl")
            )
            pickle.dump(pt_info, open(pt_path, "wb"))

            if args.save_image:
                save_numpy_to_nifti(
                    image3D_full,
                    osp.join(
                        vis_root,
                        osp.basename(img_name).replace(".nii.gz", f"_img.nii.gz"),
                    ),
                    meta_info,
                )
            for idx, pred3D_full in pred3D_full_dict.items():
                save_numpy_to_nifti(
                    pred3D_full,
                    osp.join(
                        vis_root,
                        osp.basename(img_name).replace(".nii.gz", f"_pred{idx}.nii.gz"),
                    ),
                    meta_info,
                )
                radius = 2
                for pt in points[: idx + 1]:
                    pred3D_full[
                        ...,
                        pt[0, 0, 0] - radius : pt[0, 0, 0] + radius,
                        pt[0, 0, 1] - radius : pt[0, 0, 1] + radius,
                        pt[0, 0, 2] - radius : pt[0, 0, 2] + radius,
                    ] = 10
                save_numpy_to_nifti(
                    pred3D_full,
                    osp.join(
                        vis_root,
                        osp.basename(img_name).replace(
                            ".nii.gz", f"_pred{idx}_wPt.nii.gz"
                        ),
                    ),
                    meta_info,
                )

    print("Done")
