# batch_evaluation.py

import argparse
import datetime
import glob
import json
import os
from collections import defaultdict

import numpy as np
from metric_utils import compute_metrics
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Batch computation of evaluation metrics for segmentation predictions.')
    parser.add_argument('--gt_dir',
                        type=str,
                        required=True,
                        help='Directory containing the ground truth NIfTI files (.nii.gz).')
    parser.add_argument('--pred_dir',
                        type=str,
                        required=True,
                        help='Directory containing the prediction NIfTI files (.nii.gz).')
    parser.add_argument('--output_json',
                        type=str,
                        default=None,
                        help='Optional path to save the detailed results in a JSON file.')
    parser.add_argument(
        '--classes',
        type=int,
        nargs='*',
        default=None,
        help='Optional list of class indices to compute metrics for (e.g., --classes 1 2 3). '
        'If not provided, all non-zero classes from the first GT file will be used.')

    args = parser.parse_args()

    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    output_json_path = args.output_json
    specified_classes = args.classes

    if not os.path.isdir(gt_dir):
        print(f"Error: Ground truth directory not found at {gt_dir}")
        exit()
    if not os.path.isdir(pred_dir):
        print(f"Error: Prediction directory not found at {pred_dir}")
        exit()

    # Find all prediction files
    prediction_files = glob.glob(os.path.join(pred_dir, '*.nii.gz'))

    if not prediction_files:
        print(f"No .nii.gz files found in prediction directory: {pred_dir}")
        return

    # --- Data structures to store results ---
    # Stores per-file, per-class results: {filename: {class_id: {'dsc': value, 'nsd': value}}}
    all_file_results = {}
    # Stores flattened lists of all results per class: {class_id: {'dsc': [v1,
    # v2, ...], 'nsd': [v1, v2, ...]}}
    aggregated_class_results = defaultdict(lambda: defaultdict(list))
    processed_files_count = 0
    errors = []
    missing_gt_count = 0

    print(f"Found {len(prediction_files)} prediction files in {pred_dir}")

    # --- Process each prediction file ---
    for pred_file in tqdm(prediction_files, desc="Computing metrics", unit="file"):
        filename = os.path.basename(pred_file)
        gt_file = os.path.join(gt_dir, filename)

        if not os.path.exists(gt_file):
            error_msg = f"Error: Corresponding GT file not found for {filename} at {gt_file}"
            print(error_msg)
            errors.append(error_msg)
            missing_gt_count += 1
            continue  # Skip this file pair

        try:
            # Compute metrics for the current file pair
            # Pass specified_classes if provided, otherwise compute_metrics will determine
            metrics_per_class = compute_metrics(gt_file, pred_file, classes=specified_classes)

            if not metrics_per_class:
                error_msg = f"Warning: No metrics computed for {filename}. Check for errors in compute_metrics."
                print(error_msg)
                errors.append(error_msg)
                # Skip if compute_metrics returned empty (indicating failure or no classes
                # processed)
                continue

            # Store results for this file
            all_file_results[filename] = metrics_per_class

            # Aggregate results for overall calculation
            for class_id_str, metrics in metrics_per_class.items():
                class_id = int(class_id_str)  # Convert string key back to int
                aggregated_class_results[class_id]['dsc'].append(metrics.get('dsc', np.nan))
                aggregated_class_results[class_id]['nsd'].append(metrics.get('nsd', np.nan))

            processed_files_count += 1

        except FileNotFoundError as e:
            # This should ideally be caught by the os.path.exists check, but as a safeguard
            error_msg = f"Fatal Error (should not happen): File not found during compute_metrics for {filename}: {e}"
            print(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"Error processing file pair {filename}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)

    print(f"\nFinished processing. Successfully processed {processed_files_count} file pairs.")
    if missing_gt_count > 0:
        print(f"Skipped {missing_gt_count} prediction files due to missing GT.")
    if errors:
        print(f"Encountered {len(errors)} errors during processing.")

    # --- Calculate and print summary results ---

    if not all_file_results:
        print("No results were successfully computed.")
        return

    # Determine the actual classes processed (from aggregated results keys)
    processed_classes = sorted(aggregated_class_results.keys())

    print("\n====== Per-Class Evaluation Results (Mean ± Std) ======")
    per_class_summary = {}
    for class_id in processed_classes:
        dsc_values = np.array(aggregated_class_results[class_id]['dsc'])
        nsd_values = np.array(aggregated_class_results[class_id]['nsd'])

        # Use nanmean and nanstd to ignore NaN values
        mean_dsc = np.nanmean(dsc_values)
        std_dsc = np.nanstd(dsc_values)
        mean_nsd = np.nanmean(nsd_values)
        std_nsd = np.nanstd(nsd_values)

        print(f"Class {class_id}:")
        # Check if mean is NaN (happens if all values for a class were NaN)
        if np.isnan(mean_dsc):
            print("  DSC: N/A (all values were NaN)")
        else:
            print(f"  DSC: {mean_dsc:.4f} ± {std_dsc:.4f}")

        if np.isnan(mean_nsd):
            print("  NSD: N/A (all values were NaN)")
        else:
            print(f"  NSD: {mean_nsd:.4f} ± {std_nsd:.4f}")

        per_class_summary[class_id] = {
            'DSC_mean': float(mean_dsc) if not np.isnan(mean_dsc) else None,
            'DSC_std': float(std_dsc) if not np.isnan(std_dsc) else None,
            'NSD_mean': float(mean_nsd) if not np.isnan(mean_nsd) else None,
            'NSD_std': float(std_nsd) if not np.isnan(std_nsd) else None,
            'count': len(dsc_values)  # How many files contributed to this class
        }

    print("====================================================")

    # --- Calculate and print overall results ---
    print("\n======== Overall Evaluation Results (Mean ± Std) ========")
    # Flatten all non-NaN DSC and NSD values across all classes
    all_dsc_flat = [
        v for class_data in aggregated_class_results.values() for v in class_data['dsc']
        if not np.isnan(v)
    ]
    all_nsd_flat = [
        v for class_data in aggregated_class_results.values() for v in class_data['nsd']
        if not np.isnan(v)
    ]

    overall_summary = {}
    if all_dsc_flat:
        overall_mean_dsc = np.mean(all_dsc_flat)
        overall_std_dsc = np.std(
            all_dsc_flat)  # Note: using std here, nanstd not applicable to list of non-nans
        print(f"Overall DSC: {overall_mean_dsc:.4f} ± {overall_std_dsc:.4f}")
        overall_summary['DSC_overall_mean'] = float(overall_mean_dsc)
        overall_summary['DSC_overall_std'] = float(overall_std_dsc)
    else:
        print("Overall DSC: N/A (no valid values)")
        overall_summary['DSC_overall_mean'] = None
        overall_summary['DSC_overall_std'] = None

    if all_nsd_flat:
        overall_mean_nsd = np.mean(all_nsd_flat)
        overall_std_nsd = np.std(all_nsd_flat)
        print(f"Overall NSD: {overall_mean_nsd:.4f} ± {overall_std_nsd:.4f}")
        overall_summary['NSD_overall_mean'] = float(overall_mean_nsd)
        overall_summary['NSD_overall_std'] = float(overall_std_nsd)
    else:
        print("Overall NSD: N/A (no valid values)")
        overall_summary['NSD_overall_mean'] = None
        overall_summary['NSD_overall_std'] = None

    print("======================================================")

    # --- Save results to JSON if requested ---
    if output_json_path:
        results_data = {
            "metadata": {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "gt_directory": gt_dir,
                "pred_directory": pred_dir,
                "specified_classes": specified_classes,
                "processed_files_count": processed_files_count,
                "skipped_missing_gt_count": missing_gt_count,
                "errors_count": len(errors),
                "errors": errors
            },
            "per_file_results": all_file_results,
            "per_class_summary": per_class_summary,
            "overall_summary": overall_summary
        }
        try:
            with open(output_json_path, 'w') as f:
                json.dump(results_data, f, indent=4)
            print(f"\nDetailed results saved to {output_json_path}")
        except Exception as e:
            print(f"Error saving results to JSON file {output_json_path}: {str(e)}")


if __name__ == "__main__":
    main()
