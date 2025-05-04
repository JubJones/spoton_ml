import logging
import sys
import time
import traceback # Import traceback
from typing import Dict, Any, List, Tuple
import warnings
from pathlib import Path
from collections import defaultdict
import shutil

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2 # Import OpenCV

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.logging_utils import setup_logging
    from src.utils.mlflow_utils import setup_mlflow_experiment
    from src.data.eda_loader import discover_data_assets, load_all_ground_truth, load_sample_image_data
    from src.core.runner import log_git_info
except ImportError as e:
    print(f"Error importing local modules: {e}\nPlease ensure PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="eda", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning)


def calculate_statistics(df_gt: pd.DataFrame, discovered_assets) -> Dict[str, Any]:
    logger.info("Calculating EDA statistics...")
    stats = {}

    if df_gt.empty:
        logger.warning("GT DataFrame is empty. Cannot calculate statistics.")
        return {
            'total_scenes': 0, 'total_cameras': 0, 'total_annotations': 0,
            'total_unique_obj_ids': 0
        }

    # Basic Counts
    stats['total_scenes'] = df_gt['scene_id'].nunique()
    stats['total_cameras'] = df_gt[['scene_id', 'camera_id']].drop_duplicates().shape[0]
    stats['total_annotations'] = len(df_gt)
    stats['total_unique_obj_ids'] = df_gt['obj_id'].nunique()

    # Unique Object IDs per Camera
    unique_ids_per_cam = df_gt.groupby(['scene_id', 'camera_id'])['obj_id'].nunique()
    stats['obj_ids_per_cam_min'] = unique_ids_per_cam.min()
    stats['obj_ids_per_cam_max'] = unique_ids_per_cam.max()
    stats['obj_ids_per_cam_mean'] = unique_ids_per_cam.mean()
    stats['obj_ids_per_cam_median'] = unique_ids_per_cam.median()

    # Annotations per Frame (requires grouping by scene, cam, frame)
    annos_per_frame = df_gt.groupby(['scene_id', 'camera_id', 'frame_idx']).size()
    stats['annos_per_frame_min'] = annos_per_frame.min()
    stats['annos_per_frame_max'] = annos_per_frame.max()
    stats['annos_per_frame_mean'] = annos_per_frame.mean()
    stats['annos_per_frame_median'] = annos_per_frame.median()

    # BBox Dimensions (filter invalid first)
    valid_boxes = df_gt[(df_gt['w'] > 0) & (df_gt['h'] > 0)].copy()
    if not valid_boxes.empty:
        valid_boxes['area'] = valid_boxes['w'] * valid_boxes['h']
        stats['bbox_w_min'] = valid_boxes['w'].min()
        stats['bbox_w_max'] = valid_boxes['w'].max()
        stats['bbox_w_mean'] = valid_boxes['w'].mean()
        stats['bbox_w_median'] = valid_boxes['w'].median()
        stats['bbox_h_min'] = valid_boxes['h'].min()
        stats['bbox_h_max'] = valid_boxes['h'].max()
        stats['bbox_h_mean'] = valid_boxes['h'].mean()
        stats['bbox_h_median'] = valid_boxes['h'].median()
        stats['bbox_area_min'] = valid_boxes['area'].min()
        stats['bbox_area_max'] = valid_boxes['area'].max()
        stats['bbox_area_mean'] = valid_boxes['area'].mean()
        stats['bbox_area_median'] = valid_boxes['area'].median()
    else:
        stats.update({k: 0 for k in [
            'bbox_w_min', 'bbox_w_max', 'bbox_w_mean', 'bbox_w_median',
            'bbox_h_min', 'bbox_h_max', 'bbox_h_mean', 'bbox_h_median',
            'bbox_area_min', 'bbox_area_max', 'bbox_area_mean', 'bbox_area_median'
        ]})

    # Track Lengths per Camera
    track_lengths = df_gt.groupby(['scene_id', 'camera_id', 'obj_id']).size()
    stats['track_len_min'] = track_lengths.min()
    stats['track_len_max'] = track_lengths.max()
    stats['track_len_mean'] = track_lengths.mean()
    stats['track_len_median'] = track_lengths.median()

    # Add frame count info from discovery
    stats['total_frame_indices_discovered'] = sum(assets['frame_count_gt']
                                                for scene in discovered_assets.values()
                                                for assets in scene.values()
                                                if assets['frame_count_gt'] > 0) # Sum GT counts

    logger.info("Statistics calculation complete.")
    logger.debug(f"Calculated Stats: {stats}")
    return stats

def perform_quality_checks(
    df_gt: pd.DataFrame,
    discovered_assets: Dict[str, Dict[str, Dict[str, Any]]],
    sample_image_data: Dict[Tuple[str, str], List[Dict[str, Any]]],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger.info("Performing data quality checks...")
    quality_report = {}

    # 1. Cameras Missing GT (already handled by discovery filter)
    # Count discovered scenes/cameras vs those with GT data loaded
    total_discovered_cameras = sum(len(cams) for cams in discovered_assets.values())
    cameras_with_gt = df_gt[['scene_id', 'camera_id']].drop_duplicates().shape[0] if not df_gt.empty else 0
    quality_report['qc_cameras_discovered'] = total_discovered_cameras
    quality_report['qc_cameras_with_gt_loaded'] = cameras_with_gt
    quality_report['qc_cameras_missing_gt_file_or_empty'] = total_discovered_cameras - cameras_with_gt

    # 2. Invalid Annotations (Non-positive W/H)
    if not df_gt.empty:
        invalid_boxes_count = len(df_gt[(df_gt['w'] <= 0) | (df_gt['h'] <= 0)])
        quality_report['qc_annotations_invalid_wh'] = invalid_boxes_count
        logger.info(f"Found {invalid_boxes_count} annotations with W <= 0 or H <= 0.")
    else:
         quality_report['qc_annotations_invalid_wh'] = 0

    # 3. Frame Count Mismatches
    mismatches = []
    mismatch_threshold = config.get('frame_count_mismatch_threshold', 5)
    for scene_id, cameras in discovered_assets.items():
        for cam_id, assets in cameras.items():
            jpg_count = assets['frame_count_jpg']
            gt_count = assets['frame_count_gt'] # Max frame index + 1
            if jpg_count >= 0 and gt_count >= 0 and abs(jpg_count - gt_count) > mismatch_threshold:
                mismatches.append({'scene': scene_id, 'cam': cam_id, 'jpg_count': jpg_count, 'gt_count': gt_count})
                logger.warning(f"Frame count mismatch detected for {scene_id}/{cam_id}: "
                               f"JPGs={jpg_count}, GT_MaxFrame+1={gt_count} (Diff > {mismatch_threshold})")

    quality_report['qc_frame_count_mismatches_found'] = len(mismatches)
    quality_report['qc_frame_count_mismatch_details'] = mismatches # Store details

    # 4. Out-of-Bounds Annotations (based on sample image dimensions)
    oob_count = 0
    oob_details = []
    boundary_margin = config.get('bbox_boundary_margin', 5)
    if not df_gt.empty and sample_image_data:
        # Create a quick lookup for image dimensions
        img_dims = {}
        for (scene, cam), samples in sample_image_data.items():
            if samples: # Use the first sample's dimensions
                img_dims[(scene, cam)] = (samples[0]['width'], samples[0]['height'])

        if img_dims:
            for index, row in tqdm(df_gt.iterrows(), total=len(df_gt), desc="Checking OOB Annotations"):
                scene, cam = row['scene_id'], row['camera_id']
                if (scene, cam) in img_dims:
                    img_w, img_h = img_dims[(scene, cam)]
                    x1, y1, w, h = row['x'], row['y'], row['w'], row['h']
                    x2, y2 = x1 + w, y1 + h
                    # Check if box significantly exceeds boundaries (allowing some margin)
                    if (x1 < -boundary_margin or y1 < -boundary_margin or
                        x2 > img_w + boundary_margin or y2 > img_h + boundary_margin):
                        oob_count += 1
                        # Log only a few details to avoid huge reports
                        if oob_count <= 20:
                             oob_details.append({
                                 'scene': scene, 'cam': cam, 'frame': row['frame_idx'], 'obj': row['obj_id'],
                                 'box': [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                                 'img_dims': [img_w, img_h]
                             })
        else:
             logger.warning("Cannot check OOB annotations: No valid sample image dimensions found.")

    quality_report['qc_annotations_out_of_bounds'] = oob_count
    quality_report['qc_annotations_out_of_bounds_details_sample'] = oob_details
    logger.info(f"Found {oob_count} annotations potentially out of image bounds (margin={boundary_margin}px).")

    # 5. Image Dimension Consistency Check (using sample data)
    dim_consistency = defaultdict(lambda: {'count': 0, 'dims': set()})
    for (scene, cam), samples in sample_image_data.items():
        for sample in samples:
             dim_tuple = (sample['width'], sample['height'])
             dim_consistency[(scene, cam)]['count'] += 1
             dim_consistency[(scene, cam)]['dims'].add(dim_tuple)

    inconsistent_dims = []
    for (scene, cam), data in dim_consistency.items():
        if len(data['dims']) > 1:
            inconsistent_dims.append({
                'scene': scene, 'cam': cam,
                'dimensions_found': list(data['dims']),
                'samples_checked': data['count']
            })
            logger.warning(f"Inconsistent image dimensions found for {scene}/{cam}: {data['dims']}")

    quality_report['qc_image_dimension_inconsistencies_found'] = len(inconsistent_dims)
    quality_report['qc_image_dimension_inconsistency_details'] = inconsistent_dims


    logger.info("Data quality checks complete.")
    return quality_report

def generate_plots(
    df_gt: pd.DataFrame,
    stats: Dict[str, Any],
    output_dir: Path,
    config: Dict[str, Any]
) -> List[Path]:
    logger.info(f"Generating distribution plots in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    bins = config.get('plot_hist_bins', 50)
    sns.set_theme(style="whitegrid")

    if df_gt.empty:
        logger.warning("GT DataFrame is empty. Skipping plot generation.")
        return []

    # Plotting helper function
    def _save_plot(data: pd.Series, title: str, xlabel: str, filename: str):
        if data.empty:
             logger.warning(f"No data for plot: {title}. Skipping.")
             return
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=bins, kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        filepath = output_dir / f"{filename}.png"
        try:
            plt.savefig(filepath, bbox_inches='tight')
            plot_paths.append(filepath)
            logger.debug(f"Saved plot: {filepath}")
        except Exception as e:
             logger.error(f"Failed to save plot {filepath}: {e}")
        plt.close()

    # 1. Annotations per Frame
    annos_per_frame = df_gt.groupby(['scene_id', 'camera_id', 'frame_idx']).size()
    _save_plot(annos_per_frame, 'Distribution of Annotations per Frame', 'Number of Annotations', 'hist_annos_per_frame')

    # Filter for valid boxes before plotting dimensions/area
    valid_boxes = df_gt[(df_gt['w'] > 0) & (df_gt['h'] > 0)].copy()
    if not valid_boxes.empty:
         valid_boxes['area'] = valid_boxes['w'] * valid_boxes['h']
         # 2. BBox Widths
         _save_plot(valid_boxes['w'], 'Distribution of Bounding Box Widths', 'Width (pixels)', 'hist_bbox_widths')
         # 3. BBox Heights
         _save_plot(valid_boxes['h'], 'Distribution of Bounding Box Heights', 'Height (pixels)', 'hist_bbox_heights')
         # 4. BBox Areas
         _save_plot(valid_boxes['area'], 'Distribution of Bounding Box Areas', 'Area (pixels^2)', 'hist_bbox_areas')
    else:
        logger.warning("No valid bounding boxes found. Skipping dimension/area plots.")

    # 5. Track Lengths
    track_lengths = df_gt.groupby(['scene_id', 'camera_id', 'obj_id']).size()
    _save_plot(track_lengths, 'Distribution of Track Lengths (Frames per Object ID)', 'Number of Frames', 'hist_track_lengths')

    logger.info(f"Generated {len(plot_paths)} plots.")
    return plot_paths

def create_summary_report(
    stats: Dict[str, Any],
    quality_report: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    comparison_plots_generated: bool # Add flag
) -> Path:
    """Creates a text summary file of the EDA findings."""
    summary_path = output_dir / "eda_summary.txt"
    logger.info(f"Creating EDA summary report: {summary_path}")
    vis_config = config.get('preprocessing_visualization', {})
    vis_enabled = vis_config.get('enabled', False)

    with open(summary_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write(" MTMMC Dataset EDA Summary\n")
        f.write("="*40 + "\n\n")

        f.write("--- Configuration ---\n")
        f.write(f"Base Path: {config.get('base_path', 'N/A')}\n")
        f.write(f"Selection Strategy: {config.get('selection_strategy', 'N/A')}\n")
        f.write("\n")

        f.write("--- Basic Statistics ---\n")
        for key, val in stats.items():
             if 'per_cam' not in key and 'track_len' not in key and 'annos_per_frame' not in key and 'bbox' not in key:
                f.write(f"{key.replace('_', ' ').title()}: {val}\n")
        f.write("\n")

        f.write("--- Annotation Distributions ---\n")
        for metric in ['annos_per_frame', 'bbox_w', 'bbox_h', 'bbox_area', 'track_len']:
            min_v = stats.get(f'{metric}_min', 'N/A')
            max_v = stats.get(f'{metric}_max', 'N/A')
            mean_v = stats.get(f'{metric}_mean', 'N/A')
            median_v = stats.get(f'{metric}_median', 'N/A')
            if isinstance(mean_v, (float, np.number)): mean_v = f"{mean_v:.2f}"
            f.write(f"{metric.replace('_', ' ').title()}: Min={min_v}, Max={max_v}, Mean={mean_v}, Median={median_v}\n")
        f.write("\n")

        f.write("--- Data Quality Checks ---\n")
        f.write(f"Cameras Discovered: {quality_report.get('qc_cameras_discovered', 'N/A')}\n")
        f.write(f"Cameras with GT Loaded: {quality_report.get('qc_cameras_with_gt_loaded', 'N/A')}\n")
        f.write(f"Cameras Missing/Empty GT: {quality_report.get('qc_cameras_missing_gt_file_or_empty', 'N/A')}\n")
        f.write(f"Annotations with Invalid W/H: {quality_report.get('qc_annotations_invalid_wh', 'N/A')}\n")
        f.write(f"Frame Count Mismatches Found: {quality_report.get('qc_frame_count_mismatches_found', 'N/A')}\n")
        f.write(f"Annotations Out-Of-Bounds: {quality_report.get('qc_annotations_out_of_bounds', 'N/A')}\n")
        f.write(f"Image Dimension Inconsistencies: {quality_report.get('qc_image_dimension_inconsistencies_found', 'N/A')}\n")
        f.write("\n")

        f.write("--- Preprocessing Outline (Conceptual) ---\n")
        f.write("1. Load image (e.g., using OpenCV).\n")
        f.write("2. Retrieve corresponding GT annotations for the frame/camera.\n")
        f.write("3. Filter invalid annotations (e.g., w<=0, h<=0).\n")
        f.write("4. Convert bounding box format (e.g., x,y,w,h to xmin,ymin,xmax,ymax).\n")
        f.write("5. Apply image transformations (e.g., resizing, normalization) if needed for model.\n")
        f.write("6. Apply corresponding transformations to bounding boxes.\n")
        f.write("7. Format data into required structure (e.g., tensors, target dictionaries).\n")
        f.write("\n")

        # Add note about visualization
        f.write("--- Preprocessing Visualization ---\n")
        if vis_enabled and comparison_plots_generated:
             f.write("Raw vs. Preprocessed comparison plots were generated.\n")
             f.write(f"Target Width: {vis_config.get('target_input_width', 'N/A')}\n")
             f.write(f"Normalization Mean (RGB): {vis_config.get('normalization_mean', 'N/A')}\n")
             f.write(f"Normalization Std (RGB): {vis_config.get('normalization_std', 'N/A')}\n")
        elif vis_enabled:
             f.write("Preprocessing visualization was enabled but no plots were generated (check sample size or errors).\n")
        else:
             f.write("Preprocessing visualization was disabled in the configuration.\n")
        f.write("\n")


    logger.info("EDA summary report created successfully.")
    return summary_path


# --- Preprocessing and Visualization Functions ---

def preprocess_image_for_vis(
    img_bgr: np.ndarray,
    target_width: int,
    norm_mean_rgb: List[float],
    norm_std_rgb: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies resizing, color conversion, and normalization.
    Returns both the normalized image (e.g., for model input)
    and an unnormalized version for visualization.
    """
    # 1. Resizing (maintaining aspect ratio)
    h, w = img_bgr.shape[:2]
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    resized_bgr = cv2.resize(img_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # 2. Color Conversion (BGR -> RGB)
    img_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # 3. Normalization (to float, scale 0-1, then normalize)
    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array(norm_mean_rgb, dtype=np.float32)
    std = np.array(norm_std_rgb, dtype=np.float32)
    normalized_img = (img_float - mean) / std

    # 4. Un-normalize for visualization
    unnormalized_img_float = (normalized_img * std) + mean
    # Clip to [0, 1] and convert back to uint8 [0, 255]
    unnormalized_img_vis = np.clip(unnormalized_img_float * 255.0, 0, 255).astype(np.uint8)

    return normalized_img, unnormalized_img_vis

def generate_preprocessing_comparison_plots(
    sample_image_data: Dict[Tuple[str, str], List[Dict[str, Any]]],
    vis_config: Dict[str, Any],
    output_dir: Path
) -> List[Path]:
    """
    Generates side-by-side plots comparing raw and preprocessed sample images.
    """
    logger.info(f"Generating preprocessing comparison plots in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    target_width = vis_config.get('target_input_width', 640)
    norm_mean = vis_config.get('normalization_mean', [0.485, 0.456, 0.406])
    norm_std = vis_config.get('normalization_std', [0.229, 0.224, 0.225])
    num_to_plot_per_cam = vis_config.get('num_comparison_plots_per_camera', 2)

    total_samples_to_process = sum(min(num_to_plot_per_cam, len(samples))
                                    for samples in sample_image_data.values())
    pbar_vis = tqdm(total=total_samples_to_process, desc="Generating Vis Plots")

    for (scene_id, cam_id), samples in sample_image_data.items():
        count = 0
        for sample_info in samples:
            if count >= num_to_plot_per_cam:
                break # Limit plots per camera

            pbar_vis.set_postfix_str(f"{scene_id}/{cam_id} Sample {count+1}")
            img_path = Path(sample_info['path'])
            filename = img_path.name

            try:
                # Load raw BGR image
                img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
                raw_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if raw_bgr is None:
                    logger.warning(f"[{scene_id}/{cam_id}] Failed to load image {filename} for comparison plot.")
                    pbar_vis.update(1) # Update progress even on failure
                    continue

                # Preprocess
                _, processed_vis = preprocess_image_for_vis(
                    raw_bgr, target_width, norm_mean, norm_std
                )

                # Prepare raw image for display (BGR -> RGB)
                raw_rgb_display = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

                # Create plot
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f"Preprocessing Comparison: {scene_id}/{cam_id}/{filename}", fontsize=14)

                axs[0].imshow(raw_rgb_display)
                axs[0].set_title(f"Raw ({raw_rgb_display.shape[1]}x{raw_rgb_display.shape[0]})")
                axs[0].axis('off')

                axs[1].imshow(processed_vis)
                axs[1].set_title(f"Processed Vis ({processed_vis.shape[1]}x{processed_vis.shape[0]})")
                axs[1].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

                # Save plot
                plot_filename = f"compare_{scene_id}_{cam_id}_{filename}.png"
                filepath = output_dir / plot_filename
                plt.savefig(filepath)
                plot_paths.append(filepath)
                plt.close(fig) # Close figure to free memory
                count += 1
                pbar_vis.update(1)

            except Exception as e:
                logger.error(f"Failed to generate comparison plot for {img_path}: {e}", exc_info=True)
                pbar_vis.update(1) # Ensure progress bar updates even on error
                if 'fig' in locals() and plt.fignum_exists(fig.number):
                    plt.close(fig) # Attempt to close figure if error occurred mid-plot

    pbar_vis.close()
    logger.info(f"Generated {len(plot_paths)} preprocessing comparison plots.")
    return plot_paths


# --- MLflow Logging Function ---
def log_to_mlflow(
    stats: Dict[str, Any],
    quality_report: Dict[str, Any],
    dist_plot_paths: List[Path],
    comparison_plot_paths: List[Path], # Add comparison plots
    summary_path: Path,
    config_path: str
):
    """Logs EDA results to the active MLflow run."""
    logger.info("Logging EDA results to MLflow...")

    # --- Log Config
    try:
        mlflow.log_artifact(config_path, artifact_path="config")
    except Exception as e: logger.error(f"Failed to log config artifact: {e}")

    # Log Parameters (Counts)
    params_to_log = [
        'total_scenes', 'total_cameras', 'total_frame_indices_discovered',
        'total_annotations', 'total_unique_obj_ids'
    ]
    for key in params_to_log:
        if key in stats:
            try: mlflow.log_param(key, stats[key])
            except Exception as e: logger.warning(f"Failed to log param '{key}': {e}")

    # Log Metrics (Statistics & Quality Checks)
    metrics_to_log = {**stats, **quality_report}
    for key, value in metrics_to_log.items():
        if key not in params_to_log and isinstance(value, (int, float, np.number)):
            try: mlflow.log_metric(key, float(value))
            except Exception as e: logger.warning(f"Failed to log metric '{key}': {e}")

    # --- Log Artifacts ---
    # Log Distribution Plots
    if dist_plot_paths:
        dist_plot_dir = dist_plot_paths[0].parent
        try:
            mlflow.log_artifacts(str(dist_plot_dir), artifact_path="eda_plots/distributions")
            logger.info(f"Logged {len(dist_plot_paths)} distribution plots.")
        except Exception as e:
            logger.error(f"Failed to log distribution plot artifacts from {dist_plot_dir}: {e}")

    # Log Comparison Plots
    if comparison_plot_paths:
        comp_plot_dir = comparison_plot_paths[0].parent
        try:
            mlflow.log_artifacts(str(comp_plot_dir), artifact_path="eda_plots/preprocessing_comparison")
            logger.info(f"Logged {len(comparison_plot_paths)} comparison plots.")
        except Exception as e:
            logger.error(f"Failed to log comparison plot artifacts from {comp_plot_dir}: {e}")

    # Log Summary Report
    if summary_path and summary_path.exists():
        try:
            mlflow.log_artifact(str(summary_path), artifact_path="summary")
            logger.info(f"Logged summary report artifact.")
        except Exception as e:
            logger.error(f"Failed to log summary artifact: {e}")

    logger.info("MLflow logging complete.")


# --- Main Execution ---
def main():
    """Main function to orchestrate the EDA process."""
    logger.info("--- Starting MTMMC Dataset EDA Run ---")
    config_path_str = "configs/eda_config.yaml"
    final_status = "FAILED"
    run_id = None
    temp_output_dir = None

    try:
        # 1. Load Configuration
        config = load_config(config_path_str)
        if not config:
            logger.critical(f"Failed to load configuration from {config_path_str}. Exiting.")
            sys.exit(1)
        config_path_abs = (PROJECT_ROOT / config_path_str).resolve()

        temp_output_dir = PROJECT_ROOT / config.get("output_dir", "eda_artifacts_temp")
        # Clean up existing temp dir if it exists
        if temp_output_dir.exists():
            logger.warning(f"Removing existing temporary output directory: {temp_output_dir}")
            shutil.rmtree(temp_output_dir)
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using temporary output directory: {temp_output_dir}")

        # 2. Setup MLflow
        experiment_id = setup_mlflow_experiment(config, default_experiment_name="Default MTMMC EDA")
        if not experiment_id:
            logger.critical("MLflow experiment setup failed. Exiting.")
            sys.exit(1)
        run_name = f"{config.get('mlflow', {}).get('run_name_prefix', 'eda_run')}_{time.strftime('%Y%m%d_%H%M%S')}"


        # 3. Start MLflow Run
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"--- MLflow Run Started ---")
            logger.info(f"Run Name: {run_name}")
            logger.info(f"Run ID: {run_id}")
            log_git_info()

            # --- Steps 4-6: Data Discovery, Load GT, Load Sample Metadata ---
            logger.info("--- Step 1: Discovering Data Assets ---")
            discovered_assets = discover_data_assets(
                base_path=Path(config['base_path']),
                selection_strategy=config['selection_strategy'],
                scenes_to_analyze=config.get('scenes_to_analyze')
            )
            if not discovered_assets:
                 raise RuntimeError("Data discovery yielded no valid scenes/cameras. Check config and paths.")

            logger.info("--- Step 2: Loading Ground Truth Data ---")
            df_gt, max_gt_frames = load_all_ground_truth(discovered_assets)
            for (scene, cam), max_f in max_gt_frames.items():
                if scene in discovered_assets and cam in discovered_assets[scene]:
                     discovered_assets[scene][cam]['frame_count_gt_actual_max'] = max_f

            logger.info("--- Step 3: Loading Sample Image Data (Metadata) ---")
            sample_image_data = load_sample_image_data(
                discovered_assets,
                config.get('image_sample_size_per_camera', 5)
            )

            # --- Steps 7-9: Calculate Stats, Quality Checks, Generate Dist Plots ---
            logger.info("--- Step 4: Calculating Statistics ---")
            stats = calculate_statistics(df_gt, discovered_assets)

            logger.info("--- Step 5: Performing Quality Checks ---")
            quality_report = perform_quality_checks(df_gt, discovered_assets, sample_image_data, config)

            logger.info("--- Step 6: Generating Distribution Plots ---")
            dist_plot_paths = generate_plots(df_gt, stats, temp_output_dir, config)


            # --- Step 7: Generate Preprocessing Comparison Plots ---
            comparison_plot_paths = []
            vis_config = config.get('preprocessing_visualization', {})
            vis_enabled = vis_config.get('enabled', False)
            if vis_enabled:
                logger.info("--- Step 7: Generating Preprocessing Comparison Plots ---")
                comparison_plot_paths = generate_preprocessing_comparison_plots(
                    sample_image_data,
                    vis_config,
                    temp_output_dir
                )
            else:
                logger.info("--- Step 7: Skipping Preprocessing Comparison Plots (Disabled in config) ---")


            # --- Step 8: Create Summary Report (Adjusted Step Number) ---
            logger.info("--- Step 8: Creating Summary Report ---")
            # Pass flag indicating if comparison plots were actually generated
            summary_path = create_summary_report(
                stats, quality_report, config, temp_output_dir,
                comparison_plots_generated=(len(comparison_plot_paths) > 0)
            )

            # --- Step 9: Log Results to MLflow (Adjusted Step Number) ---
            logger.info("--- Step 9: Logging Results to MLflow ---")
            log_to_mlflow(
                stats,
                quality_report,
                dist_plot_paths,
                comparison_plot_paths, # Pass comparison plot paths
                summary_path,
                str(config_path_abs)
            )

            final_status = "FINISHED"
            mlflow.set_tag("run_outcome", "Success")

    except KeyboardInterrupt:
        # ... (Interrupt handling unchanged) ...
        logger.warning("EDA run interrupted by user (KeyboardInterrupt).")
        final_status = "KILLED"
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
            mlflow.set_tag("run_outcome", "Killed by user")
            mlflow.end_run(status=final_status)
        elif run_id:
            try: mlflow.tracking.MlflowClient().set_terminated(run_id, status=final_status)
            except Exception: logger.warning(f"Could not terminate run {run_id} externally.")
    except Exception as e:
        # ... (Error handling unchanged) ...
        logger.critical(f"An uncaught error occurred during the EDA run: {e}", exc_info=True)
        final_status = "FAILED"
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
             mlflow.set_tag("run_outcome", "Crashed")
             try: mlflow.log_text(f"Error: {type(e).__name__}\n{e}\n{traceback.format_exc()}", "error_log.txt")
             except Exception: pass
             mlflow.end_run(status=final_status)
        elif run_id:
             try:
                 client = mlflow.tracking.MlflowClient()
                 client.set_tag(run_id, "run_outcome", "Crashed")
                 client.set_terminated(run_id, status=final_status)
             except Exception: logger.warning(f"Could not terminate run {run_id} externally after crash.")
    finally:
        logger.info(f"--- Finalizing EDA Run (Final Status: {final_status}) ---")
        # Log the main script log file to the run if it exists
        if run_id and log_file.exists():
            try:
                # Ensure logs are flushed
                for handler in logging.getLogger().handlers: handler.flush()
                client = mlflow.tracking.MlflowClient()
                client.log_artifact(run_id, str(log_file), artifact_path="logs")
                logger.info(f"Main EDA log file '{log_file.name}' logged as artifact to run {run_id}.")
            except Exception as log_artifact_err:
                logger.warning(f"Could not log main EDA log file artifact '{log_file}': {log_artifact_err}")

        # Ensure run termination status is set correctly if run wasn't ended above
        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            logger.info(f"Ensuring MLflow run {run_id} is terminated with status {final_status}.")
            mlflow.end_run(status=final_status)
        elif run_id: # If run exists but isn't active (e.g., due to error outside context)
            try:
                logger.warning(f"Attempting to terminate run {run_id} outside active context with status {final_status}.")
                mlflow.tracking.MlflowClient().set_terminated(run_id, status=final_status)
            except Exception as term_err:
                logger.error(f"Failed to terminate run {run_id} forcefully: {term_err}")

        # Clean up temporary directory ONLY IF THE RUN WAS SUCCESSFUL
        # Keep artifacts for debugging if failed/killed
        if final_status == "FINISHED" and temp_output_dir and temp_output_dir.exists():
             try:
                 shutil.rmtree(temp_output_dir)
                 logger.info(f"Cleaned up temporary output directory: {temp_output_dir}")
             except Exception as e:
                 logger.warning(f"Failed to clean up temporary output directory {temp_output_dir}: {e}")
        elif temp_output_dir and temp_output_dir.exists():
            logger.warning(f"Run status was '{final_status}'. Preserving temporary artifacts in {temp_output_dir}")


    logger.info(f"--- EDA Run Completed (Status: {final_status}) ---")
    sys.exit(0 if final_status == "FINISHED" else 1)


if __name__ == "__main__":
    # Ensure matplotlib backend is suitable for non-interactive use if needed
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend which doesn't require a GUI
    main()