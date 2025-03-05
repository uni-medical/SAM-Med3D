import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def get_next_click3D_torch_no_gt(prev_seg, img3D, threshold=170):
    """Selects prompt clicks from thresholded image (img3D) based on the previous segmentation (prev_seg).

    Args:
        prev_seg (torch.tensor): segmentation masks from previous iteration
        img3D (torch.tensor): input images
        threshold (int, optional): threshold value to apply to image for selecting point click. Defaults to 170.

    Returns:
        batch_points (list of torch.tensor): list of points to click
        batch_labels (list of torch.tensor): list of labels corresponding to the points
        NOTE: In this case, the labels are based on the thresholded image and not the ground truth.
    """

    mask_threshold = 0.5
    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold
    likely_masks = img3D > threshold  # NOTE: Empirical threshold
    fn_masks = torch.logical_and(likely_masks, torch.logical_not(pred_masks))
    # NOTE: Given a strict/high threshold, the false positives are not going to be very useful (at least in my case)
    # fp_masks = torch.logical_and(torch.logical_not(likely_masks), pred_masks)

    for i in range(prev_seg.shape[0]):  # , desc="generate points":

        fn_points = torch.argwhere(fn_masks[i])
        point = None
        if len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        # if no mask is given, random click a negative point
        if point is None:
            point = torch.Tensor(
                [np.random.randint(sz) for sz in fn_masks[i].size()]
            ).to(torch.int64)
            is_positive = False
        bp = point[1:].clone().detach().reshape(1, 1, -1).to(pred_masks.device)
        bl = (
            torch.tensor(
                [
                    int(is_positive),
                ]
            )
            .reshape(1, 1)
            .to(pred_masks.device)
        )

        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels


def get_next_click3D_torch_no_gt_naive(prev_seg):
    """Selects prompt clicks from the area outside predicted masks based on previous segmentation (prev_seg). 

    Args:
        prev_seg (torch.tensor): segmentation masks from previous iteration

    Returns:
        batch_points (list of torch.tensor): list of points to click
        batch_labels (list of torch.tensor): list of labels corresponding to the points
        NOTE: In this case, the labels are based on the predicted masks and not the ground truth.
    """
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold
    uncertain_masks = torch.logical_xor(
        pred_masks, pred_masks
    )  # init with all False

    for i in range(prev_seg.shape[0]):
        uncertain_region = torch.logical_or(uncertain_masks[i, 0], pred_masks[i, 0])
        points = torch.argwhere(uncertain_region) # select outside of pred mask

        if len(points) > 0:
            point = points[np.random.randint(len(points))]
            is_positive = pred_masks[i, 0, point[1], point[2], point[3]]

            bp = point[1:].clone().detach().reshape(1, 1, 3)
            bl = torch.tensor([int(is_positive)], dtype=torch.long).reshape(1, 1)
            batch_points.append(bp)
            batch_labels.append(bl)
        else:
            point = torch.Tensor(
                [np.random.randint(sz) for sz in pred_masks[i, 0].size()]
            ).to(torch.int64)
            is_positive = pred_masks[i, 0, point[1], point[2], point[3]]

            bp = point[1:].clone().detach().reshape(1, 1, 3)
            bl = torch.tensor([int(is_positive)], dtype=torch.long).reshape(1, 1)
            batch_points.append(bp)
            batch_labels.append(bl)

    return batch_points, batch_labels


def get_next_click3D_torch(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(gt_semantic_seg.shape[0]):  # , desc="generate points":

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        point = None
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # if no mask is given, random click a negative point
        if point is None:
            point = torch.Tensor(
                [np.random.randint(sz) for sz in fn_masks[i].size()]
            ).to(torch.int64)
            is_positive = False
        bp = point[1:].clone().detach().reshape(1, 1, -1).to(pred_masks.device)
        bl = (
            torch.tensor(
                [
                    int(is_positive),
                ]
            )
            .reshape(1, 1)
            .to(pred_masks.device)
        )

        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()


import edt


def get_next_click3D_torch_ritm(prev_seg, gt_semantic_seg):
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    fn_mask_single = F.pad(fn_masks, (1, 1, 1, 1, 1, 1), "constant", value=0).to(
        torch.uint8
    )[0, 0]
    fp_mask_single = F.pad(fp_masks, (1, 1, 1, 1, 1, 1), "constant", value=0).to(
        torch.uint8
    )[0, 0]
    fn_mask_dt = torch.tensor(
        edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4)
    )[1:-1, 1:-1, 1:-1]
    fp_mask_dt = torch.tensor(
        edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4)
    )[1:-1, 1:-1, 1:-1]
    fn_max_dist = torch.max(fn_mask_dt)
    fp_max_dist = torch.max(fp_mask_dt)
    is_positive = (
        fn_max_dist > fp_max_dist
    )  # the biggest area is selected to be interaction point
    dt = fn_mask_dt if is_positive else fp_mask_dt
    to_point_mask = dt > (max(fn_max_dist, fp_max_dist) / 2.0)  # use a erosion area
    to_point_mask = to_point_mask[None, None]
    # import pdb; pdb.set_trace()

    for i in range(gt_semantic_seg.shape[0]):
        points = torch.argwhere(to_point_mask[i])
        point = points[np.random.randint(len(points))]
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1, 1, 3)
        bl = torch.tensor(
            [
                int(is_positive),
            ]
        ).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()


def get_next_click3D_torch_2(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    for i in range(gt_semantic_seg.shape[0]):

        points = torch.argwhere(to_point_mask[i])
        point = points[np.random.randint(len(points))]
        # import pdb; pdb.set_trace()
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1, 1, 3)
        bl = torch.tensor(
            [
                int(is_positive),
            ]
        ).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()


def get_next_click3D_torch_with_dice(prev_seg, gt_semantic_seg):

    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = mask_pred > mask_threshold
        # mask_gt = mask_gt.astype(bool)
        mask_gt = mask_gt > 0

        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2 * volume_intersect / volume_sum

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    dice_list = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(gt_semantic_seg.shape[0]):

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # bp = torch.tensor(point[1:]).reshape(1,1,3)
        bp = point[1:].clone().detach().reshape(1, 1, 3)
        bl = torch.tensor(
            [
                int(is_positive),
            ]
        ).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))

    return batch_points, batch_labels, (sum(dice_list) / len(dice_list)).item()



def mask2D_to_bbox(gt2D, file):
    try:
        y_indices, x_indices = torch.where(gt2D > 0)
        x_min, x_max = torch.min(x_indices).item(), torch.max(x_indices).item()
        y_min, y_max = torch.min(y_indices).item(), torch.max(y_indices).item()
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        bbox_shift = torch.randint(0, 6, (1,)).item()
        scale_y, scale_x = gt2D.shape
        bbox_shift_x = int(bbox_shift * scale_x / 256)
        bbox_shift_y = int(bbox_shift * scale_y / 256)
        #print(f'{bbox_shift_x=} {bbox_shift_y=} with orig {bbox_shift=}')
        x_min = max(0, x_min - bbox_shift_x)
        x_max = min(W - 1, x_max + bbox_shift_x)
        y_min = max(0, y_min - bbox_shift_y)
        y_max = min(H - 1, y_max + bbox_shift_y)
        boxes = torch.tensor([x_min, y_min, x_max, y_max])
        return boxes
    except Exception as e:
        raise Exception(f'error {e} with file {file}')


def mask3D_to_bbox(gt3D, file):
    b_dict = {}
    z_indices, y_indices, x_indices = torch.where(gt3D > 0)
    z_min, z_max = torch.min(z_indices).item(), torch.max(z_indices).item()
    # middle of z_indices
    z_middle = z_indices[len(z_indices) // 2].item()
    D, H, W = gt3D.shape
    b_dict['z_min'] = z_min
    b_dict['z_max'] = z_max
    b_dict['z_mid'] = z_middle

    gt_mid = gt3D[z_middle]

    box_2d = mask2D_to_bbox(gt_mid, file)
    x_min, y_min, x_max, y_max = box_2d
    b_dict['z_mid_x_min'] = x_min
    b_dict['z_mid_y_min'] = y_min
    b_dict['z_mid_x_max'] = x_max
    b_dict['z_mid_y_max'] = y_max

    assert z_min == max(0, z_min)
    assert z_max == min(D - 1, z_max)
    return b_dict


def get_3D_bbox_from_gt3D(gt3D):
    batch_bboxes = []

    for i in range(gt3D.shape[0]):
        b_dict = mask3D_to_bbox(gt3D[i, 0], "fname")
        batch_bboxes.append(torch.Tensor([
                                (b_dict['z_min'], b_dict['z_mid_x_min'], b_dict['z_mid_y_min']), 
                                (b_dict['z_max'], b_dict['z_mid_x_max'], b_dict['z_mid_y_max']),
                                ])[None])
    return batch_bboxes
    

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_point(point, label, ax):
    if label == 0:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color="red"))
    else:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color="green"))
    # plt.scatter(point[0], point[1], label=label)


if __name__ == "__main__":
    gt3D = torch.randn((2, 1, 64, 256, 256)).cuda()
    prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
    prev_masks [:, 0, 10:20, 10:20, 10:20] = 1
    batch_points, batch_labels = get_next_click3D_torch(
        prev_masks.to(gt3D.device), gt3D
    )
    print(batch_points)

    print(get_3D_bbox_from_gt3D(prev_masks))
    print(get_3D_bbox_from_gt3D(prev_masks)[0].shape)
