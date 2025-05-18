import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
from surface_distance import (compute_surface_dice_at_tolerance,
                              compute_surface_distances)


def compute_dice_coefficient(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    if sum_masks == 0:
        return np.nan  # Or 1.0 depending on convention for empty masks
    return (2.0 * intersection) / sum_masks


def compute_metrics(gt_path: str,
                    pred_path: str,
                    classes: Optional[List[int]] = None,
                    metrics='all') -> Dict[str, Dict[str, float]]:
    """
    Computes evaluation metrics (DSC and NSD) between a ground truth and a prediction NIfTI file.

    Args:
        gt_path: Path to the ground truth NIfTI file.
        pred_path: Path to the prediction NIfTI file.
        classes: Optional list of class indices (integers) to compute metrics for.
                 If None, metrics are computed for all unique non-zero classes
                 found in the ground truth file.

    Returns:
        A dictionary where keys are class indices (as strings) and values are
        dictionaries containing 'dsc' and 'nsd' scores for that class.
        Returns an empty dictionary if processing fails or no classes are found/specified.

    Raises:
        FileNotFoundError: If either gt_path or pred_path does not exist.
        Exception: For errors during file loading or metric computation.
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    results: Dict[str, Dict[str, float]] = {}

    try:
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)

        gt_data = gt_nii.get_fdata().astype(np.uint8)
        pred_data = pred_nii.get_fdata().astype(np.uint8)

        case_spacing = gt_nii.header.get_zooms()[:3]  # Use only the first 3 dimensions for spacing

        # Determine classes to process
        if classes is None:
            determined_classes = sorted(np.unique(gt_data).tolist())
            if 0 in determined_classes:
                determined_classes.remove(0)  # Assume 0 is background
            if not determined_classes:
                print(f"Warning: No non-zero classes found in ground truth file: {gt_path}")
                return results
        else:
            determined_classes = sorted(classes)

        if not determined_classes:
            print(f"Warning: No classes specified or found to compute metrics for.")
            return results

        for i in determined_classes:
            organ_i_gt = (gt_data == i)
            organ_i_pred = (pred_data == i)

            if np.sum(organ_i_gt) == 0 and np.sum(organ_i_pred) == 0:
                dsc_i = np.nan
                nsd_i = np.nan
            elif np.sum(organ_i_gt) == 0 and np.sum(organ_i_pred) > 0:
                dsc_i = 0.0
                nsd_i = 0.0
            else:
                try:
                    # Calculate DSC
                    dsc_i = compute_dice_coefficient(organ_i_gt, organ_i_pred)

                    # Calculate NSD (with tolerance 1, as per original script)
                    surface_distances = compute_surface_distances(organ_i_gt, organ_i_pred,
                                                                  case_spacing)
                    nsd_i = compute_surface_dice_at_tolerance(surface_distances, 1.0)

                except Exception as metric_error:
                    print(
                        f"Warning: Error computing metrics for class {i} in {pred_path}: {metric_error}"
                    )
                    dsc_i = np.nan
                    nsd_i = np.nan

            results[str(i)] = {'dsc': float(dsc_i), 'nsd': float(nsd_i)}

    except Exception as e:
        print(f"Error processing files {gt_path} and {pred_path}: {str(e)}")
        return defaultdict(float)

    return results
