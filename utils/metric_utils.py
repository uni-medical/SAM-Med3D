import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import nibabel as nib
import numpy as np
from surface_distance import (compute_surface_dice_at_tolerance,
                              compute_surface_distances)


def compute_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.sum(mask1 * mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    if sum_masks == 0:
        return np.nan  # Or 1.0 depending on convention for empty masks
    return (2.0 * intersection) / sum_masks


def compute_metrics(gt_path: str,
                    pred_path: str,
                    classes: Optional[List[int]] = None,
                    metrics: Union[str, List[str]] = 'all') -> Dict[str, Dict[str, float]]:
    """
    Computes evaluation metrics between a ground truth and a prediction NIfTI file.

    Args:
        gt_path: Path to the ground truth NIfTI file.
        pred_path: Path to the prediction NIfTI file.
        classes: Optional list of class indices (integers) to compute metrics for.
                 If None, metrics are computed for all unique non-zero classes
                 found in the ground truth file.
        metrics: A string 'all' to compute all available metrics, or a list of
                 strings specifying which metrics to compute (e.g., ['dsc', 'nsd']).
                 Available metrics: 'dsc', 'nsd'.

    Returns:
        A dictionary where keys are class indices (as strings) and values are
        dictionaries containing the requested metric scores for that class.
        Returns an empty dictionary if processing fails or no classes are found/specified.
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    results: Dict[str, Dict[str, float]] = defaultdict(dict)
    metrics = metrics.lower() if isinstance(metrics, str) else ['dsc' if m=='dice' else m for m in metrics]
    available_metrics = {'dsc', 'nsd'}

    # Determine which metrics to compute
    metrics_to_compute: List[str]
    if isinstance(metrics, str) and metrics.lower() == 'all':
        metrics_to_compute = list(available_metrics)
    elif isinstance(metrics, list):
        metrics_to_compute = []
        for m in metrics:
            m_lower = m.lower()
            if m_lower not in available_metrics:
                raise ValueError(
                    f"Unknown metric: {m}. Available metrics are: {available_metrics}"
                )
            metrics_to_compute.append(m_lower)
        if not metrics_to_compute: # Empty list provided
            print("Warning: Empty list provided for 'metrics'. No metrics will be computed.")
            return {}
    else:
        raise ValueError(
            "Invalid 'metrics' argument. Must be 'all' or a list of metric names."
        )

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
                return dict(results)
        else:
            determined_classes = sorted(list(set(classes))) # Ensure unique and sorted

        if not determined_classes:
            print("Warning: No classes specified or found to compute metrics for.")
            return dict(results)

        # Loop through each class
        for i in determined_classes:
            class_label = str(i)
            organ_i_gt = (gt_data == i)
            organ_i_pred = (pred_data == i)

            # Handle cases where ground truth or prediction masks might be empty
            gt_empty = np.sum(organ_i_gt) == 0
            pred_empty = np.sum(organ_i_pred) == 0

            if gt_empty and pred_empty:
                # Both GT and prediction are empty for this class
                if 'dsc' in metrics_to_compute:
                    results[class_label]['dsc'] = np.nan # Or 1.0 by some conventions
                if 'nsd' in metrics_to_compute:
                    results[class_label]['nsd'] = np.nan # Or 1.0 by some conventions
                continue # Move to the next class
            elif gt_empty and not pred_empty:
                # GT is empty, but prediction is not (false positive)
                if 'dsc' in metrics_to_compute:
                    results[class_label]['dsc'] = 0.0
                if 'nsd' in metrics_to_compute:
                    results[class_label]['nsd'] = 0.0
                continue # Move to the next class
            # If pred_empty and not gt_empty (false negative), DSC and NSD will be 0, handled below.

            # --- Compute requested metrics ---
            try:
                if 'dsc' in metrics_to_compute:
                    dsc_i = compute_dice_coefficient(organ_i_gt, organ_i_pred)
                    results[class_label]['dsc'] = float(dsc_i)

                if 'nsd' in metrics_to_compute:
                    # NSD requires non-empty GT
                    if gt_empty: # Should have been caught above, but as a safeguard
                        nsd_i = 0.0 if not pred_empty else np.nan
                    else:
                        surface_distances = compute_surface_distances(
                            organ_i_gt, organ_i_pred, case_spacing)
                        nsd_i = compute_surface_dice_at_tolerance(
                            surface_distances, 1.0)
                    results[class_label]['nsd'] = float(nsd_i)

            except Exception as metric_error:
                print(
                    f"Warning: Error computing metrics for class {i} in {pred_path}: {metric_error}"
                )
                if 'dsc' in metrics_to_compute and 'dsc' not in results[class_label]:
                    results[class_label]['dsc'] = np.nan
                if 'nsd' in metrics_to_compute and 'nsd' not in results[class_label]:
                    results[class_label]['nsd'] = np.nan

    except FileNotFoundError: # Already handled by initial checks, but good practice
        raise
    except ValueError as ve: # For invalid metric names
        raise ve
    except Exception as e:
        print(f"Error processing files {gt_path} and {pred_path}: {str(e)}")
        # Return partially filled results or an empty dict if critical error
        return dict(results) if results else {}

    return dict(results)

def print_computed_metrics(results_data: Dict[Any, Any], avg_only=False, title: str = "Computed Metrics"):
    """
    Prints computed metrics in a structured format.
    Handles two types of input structures:
    1. Single result set: Dict[str(class_label), Dict[str(metric_name), float(value)]]
    2. Multi-result set: Dict[str(group_id), Dict[str(class_label), Dict[str(metric_name), float(value)]]]

    Args:
        results_data: The dictionary containing metric data.
        title: A title for the printed output.
    """
    print(f"\n--- {title} ---")
    if not results_data:
        print("No results to display.")
        return

    # --- Determine structure and process accordingly ---
    is_multi_group_type = False
    is_single_set_type = False
    processed_class_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

    # Try to determine structure by inspecting the first valid entry
    # This loop finds the first group/class that can inform the structure
    for _first_level_key, first_level_value in results_data.items():
        if isinstance(first_level_value, dict) and first_level_value: # Must be a non-empty dict
            try:
                # _second_level_key is either class_label (for multi-group) or metric_name (for single-set)
                _second_level_key = next(iter(first_level_value)) 
                # second_level_value is Dict[metric, val] (for multi-group) or val (for single-set)
                second_level_value = first_level_value[_second_level_key] 

                if isinstance(second_level_value, dict):
                    is_multi_group_type = True
                    break
                elif isinstance(second_level_value, (float, int, np.number)):
                    is_single_set_type = True
                    break
                # If neither, this entry is not informative or malformed, try next first_level_value
            except StopIteration: # first_level_value was empty, try next
                continue
        elif isinstance(first_level_value, (float, int, np.number)) and len(results_data) > 0 and isinstance(next(iter(results_data.keys())), str):
            # This handles a flat Dict[str, float] for overall averages directly
            print("Input appears to be a flat dictionary of overall metrics:")
            for metric_name, value in results_data.items():
                if isinstance(value, (float, int, np.number)):
                    print(f"  {str(metric_name).upper()}: {float(value):.4f}")
                else:
                    print(f"  {str(metric_name).upper()}: {value}")
            return


    if not is_multi_group_type and not is_single_set_type:
        # This could happen if all inner dicts are empty, or structure is unexpected
        # Check if results_data itself is Dict[str, Dict] where inner dicts are empty
        all_inner_empty = True
        if all(isinstance(v, dict) for v in results_data.values()):
            for inner_d in results_data.values():
                if inner_d: # Found a non-empty inner dict
                    all_inner_empty = False
                    break
        if all_inner_empty and any(isinstance(v,dict) for v in results_data.values()): # results_data has form Dict[str, {}]
             print("Warning: All inner dictionaries are empty. No class-specific metrics to display.")
        else:
            print("Could not determine data structure or data is malformed/empty. Cannot print detailed metrics.")
        return

    all_metric_names_overall = set() # To collect all metric names for header

    if is_multi_group_type:
        print("Processing multi-group/case results. Averaging across groups per class.")
        aggregated_by_class: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        for group_key, class_results_dict in results_data.items():
            if not isinstance(class_results_dict, dict):
                print(f"Warning: Expected dict for group '{group_key}', got {type(class_results_dict)}. Skipping.")
                continue
            for class_label, metric_val_dict in class_results_dict.items():
                if not isinstance(metric_val_dict, dict):
                    print(f"Warning: Expected dict for class '{class_label}' in group '{group_key}', got {type(metric_val_dict)}. Skipping.")
                    continue
                for metric_name, value in metric_val_dict.items():
                    all_metric_names_overall.add(metric_name)
                    if isinstance(value, (float, int, np.number)):
                        aggregated_by_class[class_label][metric_name].append(float(value))
                    else:
                        print(f"Warning: Non-numeric value '{value}' for metric '{metric_name}' in class '{class_label}', group '{group_key}'. Storing as NaN.")
                        aggregated_by_class[class_label][metric_name].append(np.nan)
        
        if not aggregated_by_class:
            print("No valid data found after aggregation from multi-group input.")
            print("--- End of Metrics ---")
            return

        for class_label, metric_lists in aggregated_by_class.items():
            for metric_name, values_list in metric_lists.items():
                if values_list:
                    processed_class_metrics[class_label][metric_name] = np.nanmean(values_list)
                else: # Should not happen if aggregation worked
                    processed_class_metrics[class_label][metric_name] = np.nan
        processed_class_metrics = dict(processed_class_metrics)

    elif is_single_set_type:
        print("Processing single set of results (per class).")
        # Ensure structure is Dict[str, Dict[str, float]]
        valid_single_set = True
        for class_label, metric_val_dict in results_data.items():
            if not isinstance(metric_val_dict, dict):
                valid_single_set = False; break
            for metric_name, value in metric_val_dict.items():
                all_metric_names_overall.add(metric_name)
                if not isinstance(value, (float, int, np.number)):
                    valid_single_set = False; break
            if not valid_single_set: break
        
        if valid_single_set:
            processed_class_metrics = results_data # type: ignore
        else:
            print("Error: Single set data is not in the expected Dict[str, Dict[str, number]] format.")
            print("--- End of Metrics ---")
            return


    # --- Print Class-wise Metrics from processed_class_metrics ---
    if not processed_class_metrics:
        print("No class-specific metrics to display after processing.")
    else:
        print("\nClass-wise Metrics:")
        sorted_class_labels = sorted(processed_class_metrics.keys(), key=lambda x: int(x) if x.isdigit() else x)
        # Ensure all_metric_names_overall is populated if it wasn't (e.g. single set path)
        if not all_metric_names_overall: 
            for class_data_dict in processed_class_metrics.values():
                if isinstance(class_data_dict, dict):
                    all_metric_names_overall.update(class_data_dict.keys())
        
        sorted_metric_names = sorted(list(all_metric_names_overall))

        if not sorted_metric_names:
            print("No metric types found to display.")
        else:
            header = f"{'Class':<12} | " + " | ".join(f"{name.upper():<7}" for name in sorted_metric_names)
            print(header)
            print("-" * len(header))

            for class_label in sorted_class_labels:
                line = f"{class_label:<12} | "
                metric_values_for_class = processed_class_metrics.get(class_label, {})
                for metric_name in sorted_metric_names:
                    value = metric_values_for_class.get(metric_name, np.nan)
                    line += f"{value:<7.4f} | "
                print(line.rstrip(" |"))
    
    # --- Calculate and Print Average Metrics (across classes from processed_class_metrics) ---
    if processed_class_metrics and all_metric_names_overall: # Check if there's anything to average
        print("\nAverage Metrics (across classes):")
        # Re-collect sorted_metric_names in case it was empty before but processed_class_metrics is not
        if not sorted_metric_names and all_metric_names_overall:
            sorted_metric_names = sorted(list(all_metric_names_overall))

        average_overall_metrics: Dict[str, float] = {}
        for metric_name in sorted_metric_names:
            all_values_for_metric = []
            for class_label in processed_class_metrics: # Use sorted_class_labels for consistency if preferred
                class_data = processed_class_metrics.get(class_label, {})
                value = class_data.get(metric_name)
                if value is not None and not np.isnan(float(value)): # Ensure it's a valid number
                    all_values_for_metric.append(float(value))
            
            if all_values_for_metric:
                average_overall_metrics[metric_name] = np.mean(all_values_for_metric) # np.mean handles lists of numbers
            else:
                average_overall_metrics[metric_name] = np.nan

        for metric_name, avg_value in average_overall_metrics.items():
            print(f"  Average {metric_name.upper()}: {avg_value:.4f}")
    
    print("--- End of Metrics ---")
