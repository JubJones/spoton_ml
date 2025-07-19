# src/pipelines/eda_pipeline.py

import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

# --- Local Imports ---
try:
    from src.components.data.eda_loader import discover_data_assets, load_all_ground_truth, load_sample_image_data
except ImportError:
    import sys
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path: sys.path.insert(0, str(_project_root))
    from data.eda_loader import discover_data_assets, load_all_ground_truth, load_sample_image_data

logger = logging.getLogger(__name__)

# Type alias for results dictionary from the pipeline run
EDAResults = Dict[str, Any] # Can contain paths, stats, etc.
PreprocessingStats = Dict[str, List[Dict[str, Any]]] # { cam_id: [{'path': path, 'original': (w,h), 'resized': (w,h)}, ...]}

class EDAPipeline:
    """Encapsulates the EDA process: discovery, loading, stats, QC, plotting, and preprocessing summary."""

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """Initializes the EDA pipeline."""
        self.config = config
        self.output_dir = output_dir
        self.vis_config = self.config.get('preprocessing_visualization', {})
        self.vis_enabled = self.vis_config.get('enabled', False)

        # --- Internal State ---
        self.discovered_assets: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
        self.df_gt: Optional[pd.DataFrame] = None
        self.sample_image_data: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]] = None
        self.stats: Dict[str, Any] = {}
        self.quality_report: Dict[str, Any] = {}
        self.dist_plot_paths: List[Path] = []
        self.comparison_plot_paths: List[Path] = []
        self.summary_path: Optional[Path] = None
        # NEW: For preprocessing stats
        self.preprocessing_stats: PreprocessingStats = defaultdict(list)
        self.aggregated_prep_stats: Dict[str, Any] = {}
        self.prep_summary_path: Optional[Path] = None


    def calculate_statistics(self) -> None:
        """Calculates various statistics from the ground truth DataFrame."""
        # --- (Implementation unchanged from previous version) ---
        logger.info("Calculating EDA statistics...")
        stats = {}
        if self.df_gt is None or self.df_gt.empty:
            logger.warning("GT DataFrame is empty or None. Cannot calculate statistics.")
            stats = {
                'total_scenes': 0, 'total_cameras': 0, 'total_annotations': 0,
                'total_unique_obj_ids': 0
            }
            self.stats = stats
            return

        # Basic Counts
        stats['total_scenes'] = self.df_gt['scene_id'].nunique()
        stats['total_cameras'] = self.df_gt[['scene_id', 'camera_id']].drop_duplicates().shape[0]
        stats['total_annotations'] = len(self.df_gt)
        stats['total_unique_obj_ids'] = self.df_gt['obj_id'].nunique()

        # Unique Object IDs per Camera
        unique_ids_per_cam = self.df_gt.groupby(['scene_id', 'camera_id'])['obj_id'].nunique()
        stats['obj_ids_per_cam_min'] = unique_ids_per_cam.min()
        stats['obj_ids_per_cam_max'] = unique_ids_per_cam.max()
        stats['obj_ids_per_cam_mean'] = unique_ids_per_cam.mean()
        stats['obj_ids_per_cam_median'] = unique_ids_per_cam.median()

        # Annotations per Frame (requires grouping by scene, cam, frame)
        annos_per_frame = self.df_gt.groupby(['scene_id', 'camera_id', 'frame_idx']).size()
        stats['annos_per_frame_min'] = annos_per_frame.min()
        stats['annos_per_frame_max'] = annos_per_frame.max()
        stats['annos_per_frame_mean'] = annos_per_frame.mean()
        stats['annos_per_frame_median'] = annos_per_frame.median()

        # BBox Dimensions (filter invalid first)
        valid_boxes = self.df_gt[(self.df_gt['w'] > 0) & (self.df_gt['h'] > 0)].copy()
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
        track_lengths = self.df_gt.groupby(['scene_id', 'camera_id', 'obj_id']).size()
        stats['track_len_min'] = track_lengths.min()
        stats['track_len_max'] = track_lengths.max()
        stats['track_len_mean'] = track_lengths.mean()
        stats['track_len_median'] = track_lengths.median()

        # Add frame count info from discovery
        if self.discovered_assets:
             stats['total_frame_indices_discovered'] = sum(assets['frame_count_gt']
                                                         for scene in self.discovered_assets.values()
                                                         for assets in scene.values()
                                                         if assets['frame_count_gt'] > 0) # Sum GT counts
        else:
             stats['total_frame_indices_discovered'] = 0

        logger.info("Statistics calculation complete.")
        logger.debug(f"Calculated Stats: {stats}")
        self.stats = stats

    def perform_quality_checks(self) -> None:
        """Performs data quality checks and returns counts/summaries."""
        # --- (Implementation unchanged from previous version) ---
        logger.info("Performing data quality checks...")
        quality_report = {}
        if self.discovered_assets is None:
            logger.warning("Discovered assets not available. Skipping quality checks.")
            self.quality_report = {}
            return

        # 1. Cameras Missing GT (already handled by discovery filter)
        total_discovered_cameras = sum(len(cams) for cams in self.discovered_assets.values())
        cameras_with_gt = 0
        if self.df_gt is not None and not self.df_gt.empty:
             cameras_with_gt = self.df_gt[['scene_id', 'camera_id']].drop_duplicates().shape[0]
        quality_report['qc_cameras_discovered'] = total_discovered_cameras
        quality_report['qc_cameras_with_gt_loaded'] = cameras_with_gt
        quality_report['qc_cameras_missing_gt_file_or_empty'] = total_discovered_cameras - cameras_with_gt

        # 2. Invalid Annotations (Non-positive W/H)
        if self.df_gt is not None and not self.df_gt.empty:
            invalid_boxes_count = len(self.df_gt[(self.df_gt['w'] <= 0) | (self.df_gt['h'] <= 0)])
            quality_report['qc_annotations_invalid_wh'] = invalid_boxes_count
            logger.info(f"Found {invalid_boxes_count} annotations with W <= 0 or H <= 0.")
        else:
             quality_report['qc_annotations_invalid_wh'] = 0

        # 3. Frame Count Mismatches
        mismatches = []
        mismatch_threshold = self.config.get('frame_count_mismatch_threshold', 5)
        for scene_id, cameras in self.discovered_assets.items():
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
        boundary_margin = self.config.get('bbox_boundary_margin', 5)
        if self.df_gt is not None and not self.df_gt.empty and self.sample_image_data:
            # Create a quick lookup for image dimensions
            img_dims = {}
            for (scene, cam), samples in self.sample_image_data.items():
                if samples: # Use the first sample's dimensions
                    img_dims[(scene, cam)] = (samples[0]['width'], samples[0]['height'])

            if img_dims:
                for index, row in tqdm(self.df_gt.iterrows(), total=len(self.df_gt), desc="Checking OOB Annotations"):
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
        if self.sample_image_data:
             for (scene, cam), samples in self.sample_image_data.items():
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
        self.quality_report = quality_report

    def generate_distribution_plots(self) -> None:
        """Generates histogram plots for distributions."""
        # --- (Implementation unchanged from previous version) ---
        logger.info(f"Generating distribution plots in {self.output_dir}...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = []
        bins = self.config.get('plot_hist_bins', 50)
        sns.set_theme(style="whitegrid")

        if self.df_gt is None or self.df_gt.empty:
            logger.warning("GT DataFrame is empty or None. Skipping distribution plot generation.")
            self.dist_plot_paths = []
            return

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
            filepath = self.output_dir / f"{filename}.png"
            try:
                plt.savefig(filepath, bbox_inches='tight')
                plot_paths.append(filepath)
                logger.debug(f"Saved plot: {filepath}")
            except Exception as e:
                 logger.error(f"Failed to save plot {filepath}: {e}")
            plt.close()

        # 1. Annotations per Frame
        annos_per_frame = self.df_gt.groupby(['scene_id', 'camera_id', 'frame_idx']).size()
        _save_plot(annos_per_frame, 'Distribution of Annotations per Frame', 'Number of Annotations', 'hist_annos_per_frame')

        # Filter for valid boxes before plotting dimensions/area
        valid_boxes = self.df_gt[(self.df_gt['w'] > 0) & (self.df_gt['h'] > 0)].copy()
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
        track_lengths = self.df_gt.groupby(['scene_id', 'camera_id', 'obj_id']).size()
        _save_plot(track_lengths, 'Distribution of Track Lengths (Frames per Object ID)', 'Number of Frames', 'hist_track_lengths')

        logger.info(f"Generated {len(plot_paths)} distribution plots.")
        self.dist_plot_paths = plot_paths

    def _preprocess_image_for_vis(
        self,
        img_bgr: np.ndarray,
        target_width: int,
        norm_mean_rgb: List[float],
        norm_std_rgb: List[float]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Tuple[int, int], Optional[Tuple[int, int]]]:
        """
        Applies resizing, color conversion, and normalization.
        Returns normalized image, visualizable image, original dims, and resized dims.
        Handles potential errors during preprocessing.
        """
        original_dims = img_bgr.shape[1::-1] # (width, height)
        resized_dims = None
        normalized_img = None
        unnormalized_img_vis = None

        try:
            # 1. Resizing (maintaining aspect ratio)
            h, w = img_bgr.shape[:2]
            aspect_ratio = h / w
            target_height = int(target_width * aspect_ratio)
            resized_bgr = cv2.resize(img_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_dims = (target_width, target_height)

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

        except Exception as e:
            logger.error(f"Error during image preprocessing for visualization: {e}", exc_info=True)
            # Return None for images if error occurs
            return None, None, original_dims, None

        return normalized_img, unnormalized_img_vis, original_dims, resized_dims

    def _process_sample_images(self) -> None:
        """
        Processes sample images to collect preprocessing stats and generate comparison plots.
        Separated logic for clarity.
        """
        if not self.vis_enabled or not self.sample_image_data:
            logger.info("Preprocessing visualization disabled or no sample data. Skipping image processing step.")
            self.comparison_plot_paths = []
            self.preprocessing_stats = {}
            return

        logger.info(f"Processing sample images for stats and comparison plots...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = []
        prep_stats: PreprocessingStats = defaultdict(list)

        target_width = self.vis_config.get('target_input_width', 640)
        norm_mean = self.vis_config.get('normalization_mean', [0.485, 0.456, 0.406])
        norm_std = self.vis_config.get('normalization_std', [0.229, 0.224, 0.225])
        num_to_plot_per_cam = self.vis_config.get('num_comparison_plots_per_camera', 2)

        # Determine total samples to iterate for stats (all samples)
        total_samples_for_stats = sum(len(samples) for samples in self.sample_image_data.values())
        pbar_samples = tqdm(total=total_samples_for_stats, desc="Processing Samples")

        plot_count_per_cam = defaultdict(int)

        for (scene_id, cam_id), samples in self.sample_image_data.items():
            cam_key = f"{scene_id}_{cam_id}" # Use combined key for stats dict
            for sample_info in samples:
                pbar_samples.update(1)
                img_path = Path(sample_info['path'])
                filename = img_path.name

                try:
                    # Load raw BGR image
                    img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
                    raw_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                    if raw_bgr is None:
                        logger.warning(f"[{scene_id}/{cam_id}] Failed to load image {filename}. Skipping.")
                        continue

                    # Preprocess (get stats and visualizable image)
                    _, processed_vis, orig_dims, resized_dims = self._preprocess_image_for_vis(
                        raw_bgr, target_width, norm_mean, norm_std
                    )

                    # Store stats regardless of plotting
                    if resized_dims: # Only store if preprocessing was successful
                         prep_stats[cam_key].append({
                             'path': str(img_path),
                             'original': orig_dims, # (w, h)
                             'resized': resized_dims # (w, h)
                         })

                    # Generate plot only if enabled, needed, and preprocessing succeeded
                    should_plot = (plot_count_per_cam[cam_key] < num_to_plot_per_cam) and (processed_vis is not None)
                    if should_plot:
                        pbar_samples.set_postfix_str(f"{scene_id}/{cam_id} Plotting {plot_count_per_cam[cam_key]+1}")
                        raw_rgb_display = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                        fig.suptitle(f"Preprocessing Comparison: {scene_id}/{cam_id}/{filename}", fontsize=14)
                        axs[0].imshow(raw_rgb_display)
                        axs[0].set_title(f"Raw ({orig_dims[0]}x{orig_dims[1]})")
                        axs[0].axis('off')
                        axs[1].imshow(processed_vis)
                        axs[1].set_title(f"Processed Vis ({resized_dims[0]}x{resized_dims[1]})")
                        axs[1].axis('off')
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        plot_filename = f"compare_{scene_id}_{cam_id}_{filename}.png"
                        filepath = self.output_dir / plot_filename
                        plt.savefig(filepath)
                        plot_paths.append(filepath)
                        plt.close(fig)
                        plot_count_per_cam[cam_key] += 1

                except Exception as e:
                    logger.error(f"Failed during sample processing for {img_path}: {e}", exc_info=True)
                    if 'fig' in locals() and plt.fignum_exists(fig.number):
                        plt.close(fig) # Attempt to close figure if error occurred mid-plot

        pbar_samples.close()
        self.comparison_plot_paths = plot_paths
        self.preprocessing_stats = dict(prep_stats) # Convert back to regular dict
        logger.info(f"Processed {total_samples_for_stats} sample images for stats.")
        logger.info(f"Generated {len(plot_paths)} preprocessing comparison plots.")


    def _aggregate_preprocessing_stats(self) -> None:
        """Aggregates the collected preprocessing statistics."""
        if not self.preprocessing_stats:
             logger.info("No preprocessing stats collected to aggregate.")
             self.aggregated_prep_stats = {}
             return

        logger.info("Aggregating preprocessing statistics...")
        all_orig_widths = []
        all_orig_heights = []
        all_resized_widths = []
        all_resized_heights = []
        total_images_processed = 0

        for cam_stats in self.preprocessing_stats.values():
            total_images_processed += len(cam_stats)
            for stats in cam_stats:
                all_orig_widths.append(stats['original'][0])
                all_orig_heights.append(stats['original'][1])
                all_resized_widths.append(stats['resized'][0])
                all_resized_heights.append(stats['resized'][1])

        if not total_images_processed:
             self.aggregated_prep_stats = {'total_images_analyzed': 0}
             return

        self.aggregated_prep_stats = {
            'total_images_analyzed': total_images_processed,
            'target_resize_width': self.vis_config.get('target_input_width', 'N/A'),
            'avg_original_width': np.mean(all_orig_widths) if all_orig_widths else 0,
            'avg_original_height': np.mean(all_orig_heights) if all_orig_heights else 0,
            'avg_resized_width': np.mean(all_resized_widths) if all_resized_widths else 0,
            'avg_resized_height': np.mean(all_resized_heights) if all_resized_heights else 0,
            'min_original_width': np.min(all_orig_widths) if all_orig_widths else 0,
            'min_original_height': np.min(all_orig_heights) if all_orig_heights else 0,
            'max_original_width': np.max(all_orig_widths) if all_orig_widths else 0,
            'max_original_height': np.max(all_orig_heights) if all_orig_heights else 0,
            'min_resized_width': np.min(all_resized_widths) if all_resized_widths else 0,
            'min_resized_height': np.min(all_resized_heights) if all_resized_heights else 0,
            'max_resized_width': np.max(all_resized_widths) if all_resized_widths else 0,
            'max_resized_height': np.max(all_resized_heights) if all_resized_heights else 0,
        }
        logger.info("Preprocessing statistics aggregated.")


    def generate_preprocessing_summary_artifact(self) -> None:
        """Generates a text artifact summarizing preprocessing statistics."""
        if not self.aggregated_prep_stats:
            logger.info("No aggregated preprocessing stats available to generate summary artifact.")
            self.prep_summary_path = None
            return

        summary_file_path = self.output_dir / "preprocessing_summary.txt"
        logger.info(f"Generating preprocessing summary artifact: {summary_file_path}")

        try:
            with open(summary_file_path, 'w') as f:
                f.write("="*40 + "\n")
                f.write(" Preprocessing Statistics Summary\n")
                f.write("="*40 + "\n\n")

                stats = self.aggregated_prep_stats
                f.write(f"Total Sample Images Analyzed: {stats.get('total_images_analyzed', 'N/A')}\n")
                f.write(f"Target Resize Width Config: {stats.get('target_resize_width', 'N/A')}\n\n")

                f.write("--- Original Dimensions ---\n")
                f.write(f"Average Width:  {stats.get('avg_original_width', 0):.1f}\n")
                f.write(f"Average Height: {stats.get('avg_original_height', 0):.1f}\n")
                f.write(f"Min Dimensions: {stats.get('min_original_width', 0)} x {stats.get('min_original_height', 0)}\n")
                f.write(f"Max Dimensions: {stats.get('max_original_width', 0)} x {stats.get('max_original_height', 0)}\n\n")

                f.write("--- Resized Dimensions (Aspect Ratio Preserved) ---\n")
                f.write(f"Average Width:  {stats.get('avg_resized_width', 0):.1f}\n")
                f.write(f"Average Height: {stats.get('avg_resized_height', 0):.1f}\n")
                f.write(f"Min Dimensions: {stats.get('min_resized_width', 0)} x {stats.get('min_resized_height', 0)}\n")
                f.write(f"Max Dimensions: {stats.get('max_resized_width', 0)} x {stats.get('max_resized_height', 0)}\n\n")

                f.write("--- Normalization Applied (RGB Order) ---\n")
                f.write(f"Mean: {self.vis_config.get('normalization_mean', 'N/A')}\n")
                f.write(f"Std Dev: {self.vis_config.get('normalization_std', 'N/A')}\n")

            self.prep_summary_path = summary_file_path
            logger.info("Preprocessing summary artifact generated successfully.")

        except Exception as e:
            logger.error(f"Failed to write preprocessing summary artifact: {e}", exc_info=True)
            self.prep_summary_path = None


    def create_summary_report(self) -> None:
        """Creates the main text summary file of the EDA findings."""
        # --- (Implementation mostly unchanged, but reads internal state) ---
        summary_path = self.output_dir / "eda_summary.txt"
        logger.info(f"Creating EDA summary report: {summary_path}")
        comparison_plots_generated = len(self.comparison_plot_paths) > 0

        with open(summary_path, 'w') as f:
            f.write("="*40 + "\n")
            f.write(" MTMMC Dataset EDA Summary\n")
            f.write("="*40 + "\n\n")

            f.write("--- Configuration ---\n")
            f.write(f"Base Path: {self.config.get('base_path', 'N/A')}\n")
            f.write(f"Selection Strategy: {self.config.get('selection_strategy', 'N/A')}\n")
            f.write("\n")

            f.write("--- Basic Statistics ---\n")
            for key, val in self.stats.items():
                 if 'per_cam' not in key and 'track_len' not in key and 'annos_per_frame' not in key and 'bbox' not in key:
                    f.write(f"{key.replace('_', ' ').title()}: {val}\n")
            f.write("\n")

            f.write("--- Annotation Distributions ---\n")
            for metric in ['annos_per_frame', 'bbox_w', 'bbox_h', 'bbox_area', 'track_len']:
                min_v = self.stats.get(f'{metric}_min', 'N/A')
                max_v = self.stats.get(f'{metric}_max', 'N/A')
                mean_v = self.stats.get(f'{metric}_mean', 'N/A')
                median_v = self.stats.get(f'{metric}_median', 'N/A')
                if isinstance(mean_v, (float, np.number)): mean_v = f"{mean_v:.2f}"
                f.write(f"{metric.replace('_', ' ').title()}: Min={min_v}, Max={max_v}, Mean={mean_v}, Median={median_v}\n")
            f.write("\n")

            f.write("--- Data Quality Checks ---\n")
            f.write(f"Cameras Discovered: {self.quality_report.get('qc_cameras_discovered', 'N/A')}\n")
            f.write(f"Cameras with GT Loaded: {self.quality_report.get('qc_cameras_with_gt_loaded', 'N/A')}\n")
            f.write(f"Cameras Missing/Empty GT: {self.quality_report.get('qc_cameras_missing_gt_file_or_empty', 'N/A')}\n")
            f.write(f"Annotations with Invalid W/H: {self.quality_report.get('qc_annotations_invalid_wh', 'N/A')}\n")
            f.write(f"Frame Count Mismatches Found: {self.quality_report.get('qc_frame_count_mismatches_found', 'N/A')}\n")
            f.write(f"Annotations Out-Of-Bounds: {self.quality_report.get('qc_annotations_out_of_bounds', 'N/A')}\n")
            f.write(f"Image Dimension Inconsistencies: {self.quality_report.get('qc_image_dimension_inconsistencies_found', 'N/A')}\n")
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

            f.write("--- Preprocessing Visualization & Stats ---\n") # Modified section title
            if self.vis_enabled:
                 f.write(f"Preprocessing analysis performed on {self.aggregated_prep_stats.get('total_images_analyzed', 0)} sample images.\n")
                 f.write(f"Visual comparison plots generated: {'Yes' if comparison_plots_generated else 'No'}\n")
                 f.write(f"Statistics summary artifact generated: {'Yes' if self.prep_summary_path else 'No'}\n")
                 f.write(f"Target Width Config: {self.vis_config.get('target_input_width', 'N/A')}\n")
                 f.write(f"Normalization Mean (RGB): {self.vis_config.get('normalization_mean', 'N/A')}\n")
                 f.write(f"Normalization Std (RGB): {self.vis_config.get('normalization_std', 'N/A')}\n")
            else:
                 f.write("Preprocessing visualization/stats were disabled in the configuration.\n")
            f.write("\n")

        logger.info("EDA summary report created successfully.")
        self.summary_path = summary_path


    def run(self) -> Tuple[bool, EDAResults]:
        """Executes the full EDA pipeline."""
        logger.info("--- Starting EDA Pipeline Execution ---")
        try:
            # Step 1: Discover Data Assets
            logger.info("--- Step 1: Discovering Data Assets ---")
            self.discovered_assets = discover_data_assets(
                base_path=Path(self.config['base_path']),
                selection_strategy=self.config['selection_strategy'],
                scenes_to_analyze=self.config.get('scenes_to_analyze')
            )
            if not self.discovered_assets:
                raise RuntimeError("Data discovery yielded no valid scenes/cameras. Check config and paths.")

            # Step 2: Load Ground Truth Data
            logger.info("--- Step 2: Loading Ground Truth Data ---")
            self.df_gt, max_gt_frames = load_all_ground_truth(self.discovered_assets)
            for (scene, cam), max_f in max_gt_frames.items():
                if scene in self.discovered_assets and cam in self.discovered_assets[scene]:
                    self.discovered_assets[scene][cam]['frame_count_gt_actual_max'] = max_f

            # Step 3: Load Sample Image Data (Metadata)
            logger.info("--- Step 3: Loading Sample Image Data (Metadata) ---")
            self.sample_image_data = load_sample_image_data(
                self.discovered_assets,
                self.config.get('image_sample_size_per_camera', 5) # Sample size still relevant here
            )

            # Step 4: Calculate General Statistics
            logger.info("--- Step 4: Calculating General Statistics ---")
            self.calculate_statistics()

            # Step 5: Perform Quality Checks
            logger.info("--- Step 5: Performing Quality Checks ---")
            self.perform_quality_checks()

            # Step 6: Generate Distribution Plots
            logger.info("--- Step 6: Generating Distribution Plots ---")
            self.generate_distribution_plots()

            # Step 7: Process Sample Images (Stats Collection & Comparison Plots)
            logger.info("--- Step 7: Processing Sample Images (Stats & Visualization) ---")
            self._process_sample_images() # Collects stats and generates plots if enabled

            # Step 8: Aggregate Preprocessing Stats
            logger.info("--- Step 8: Aggregating Preprocessing Statistics ---")
            self._aggregate_preprocessing_stats()

            # Step 9: Generate Preprocessing Summary Artifact
            logger.info("--- Step 9: Generating Preprocessing Summary Artifact ---")
            self.generate_preprocessing_summary_artifact()

            # Step 10: Create Main Summary Report
            logger.info("--- Step 10: Creating Main Summary Report ---")
            self.create_summary_report()

            logger.info("--- EDA Pipeline Execution Finished Successfully ---")

            # Prepare results for the caller
            results = {
                "status": True,
                "stats": self.stats,
                "quality_report": self.quality_report,
                "dist_plot_paths": self.dist_plot_paths,
                "comparison_plot_paths": self.comparison_plot_paths,
                "summary_path": self.summary_path,
                "prep_summary_path": self.prep_summary_path # Add path to new artifact
            }
            return True, results

        except Exception as e:
            logger.critical(f"EDA pipeline failed during execution: {e}", exc_info=True)
            results = {
                "status": False,
                "stats": self.stats, # Return whatever was calculated
                "quality_report": self.quality_report,
                "dist_plot_paths": self.dist_plot_paths,
                "comparison_plot_paths": self.comparison_plot_paths,
                "summary_path": self.summary_path,
                "prep_summary_path": self.prep_summary_path
            }
            return False, results