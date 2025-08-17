"""
MTMMC Dataset Validation and Health Check for RF-DETR Training
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class MTMCDatasetValidator:
    """Validates MTMMC dataset structure and content for RF-DETR training."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.validation_results = {
            "structure_valid": False,
            "annotations_valid": False,
            "images_accessible": False,
            "statistics": {},
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
    
    def validate_dataset_structure(self) -> bool:
        """Validate basic MTMMC dataset structure."""
        logger.info("🔍 Validating MTMMC dataset structure...")
        
        required_files = [
            "kaist_mtmdc_train.json",
            "kaist_mtmdc_val.json"
        ]
        
        required_dirs = [
            "train/train",
            "val"
        ]
        
        # Check required files
        missing_files = []
        for file_name in required_files:
            file_path = self.base_path / file_name
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            self.validation_results["issues"].append(f"❌ Missing annotation files: {missing_files}")
            return False
        
        # Check required directories
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.base_path / dir_name
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            self.validation_results["issues"].append(f"❌ Missing directories: {missing_dirs}")
            return False
        
        # Check scene structure
        train_dir = self.base_path / "train" / "train"
        scenes = [d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith('s')]
        
        if not scenes:
            self.validation_results["issues"].append("❌ No scene directories found in train/train/")
            return False
        
        # Validate scene structure
        valid_scenes = []
        for scene_dir in scenes:
            cameras = [d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith('c')]
            if cameras:
                # Check camera structure
                for camera_dir in cameras[:2]:  # Check first 2 cameras for efficiency
                    rgb_dir = camera_dir / "rgb"
                    gt_file = camera_dir / "gt" / "gt.txt"
                    
                    if not rgb_dir.exists():
                        self.validation_results["warnings"].append(f"⚠️  Missing rgb directory: {rgb_dir}")
                    if not gt_file.exists():
                        self.validation_results["warnings"].append(f"⚠️  Missing gt.txt file: {gt_file}")
                
                valid_scenes.append(scene_dir.name)
        
        self.validation_results["statistics"]["scenes_found"] = valid_scenes
        self.validation_results["structure_valid"] = len(valid_scenes) > 0
        
        if self.validation_results["structure_valid"]:
            logger.info(f"✅ Dataset structure validated. Found scenes: {valid_scenes}")
        
        return self.validation_results["structure_valid"]
    
    def validate_coco_annotations(self) -> bool:
        """Validate COCO format annotation files."""
        logger.info("📋 Validating COCO annotation files...")
        
        annotation_files = [
            ("train", self.base_path / "kaist_mtmdc_train.json"),
            ("val", self.base_path / "kaist_mtmdc_val.json")
        ]
        
        annotation_stats = {}
        
        for split_name, ann_file in annotation_files:
            if not ann_file.exists():
                self.validation_results["issues"].append(f"❌ Missing annotation file: {ann_file}")
                continue
            
            try:
                with open(ann_file, 'r') as f:
                    annotations = json.load(f)
                
                # Validate COCO structure
                required_keys = ['images', 'annotations', 'categories']
                missing_keys = [key for key in required_keys if key not in annotations]
                
                if missing_keys:
                    self.validation_results["issues"].append(
                        f"❌ Missing COCO keys in {split_name}: {missing_keys}"
                    )
                    continue
                
                # Analyze annotations
                stats = {
                    "num_images": len(annotations['images']),
                    "num_annotations": len(annotations['annotations']),
                    "num_categories": len(annotations['categories']),
                    "categories": [cat['name'] for cat in annotations['categories']]
                }
                
                # Analyze annotation distribution
                category_counts = Counter(ann['category_id'] for ann in annotations['annotations'])
                stats["annotations_per_category"] = dict(category_counts)
                
                # Analyze image dimensions
                image_sizes = [(img['width'], img['height']) for img in annotations['images']]
                stats["unique_image_sizes"] = list(set(image_sizes))
                
                # Analyze bounding box sizes
                bbox_areas = []
                bbox_aspect_ratios = []
                for ann in annotations['annotations']:
                    bbox = ann['bbox']  # [x, y, width, height]
                    area = bbox[2] * bbox[3]
                    aspect_ratio = bbox[3] / bbox[2] if bbox[2] > 0 else 0  # height/width for person detection
                    bbox_areas.append(area)
                    bbox_aspect_ratios.append(aspect_ratio)
                
                if bbox_areas:
                    stats["bbox_stats"] = {
                        "mean_area": np.mean(bbox_areas),
                        "median_area": np.median(bbox_areas),
                        "min_area": np.min(bbox_areas),
                        "max_area": np.max(bbox_areas),
                        "mean_aspect_ratio": np.mean(bbox_aspect_ratios),
                        "median_aspect_ratio": np.median(bbox_aspect_ratios)
                    }
                
                annotation_stats[split_name] = stats
                logger.info(f"✅ {split_name} annotations validated: {stats['num_images']} images, {stats['num_annotations']} annotations")
                
            except json.JSONDecodeError as e:
                self.validation_results["issues"].append(f"❌ Invalid JSON in {split_name}: {e}")
            except Exception as e:
                self.validation_results["issues"].append(f"❌ Error processing {split_name} annotations: {e}")
        
        self.validation_results["statistics"]["annotations"] = annotation_stats
        
        # Validate person detection setup
        if annotation_stats:
            sample_categories = next(iter(annotation_stats.values()))["categories"]
            if "person" in sample_categories:
                logger.info("✅ Person category detected - ready for person detection training")
            else:
                self.validation_results["warnings"].append(
                    f"⚠️  'person' category not found. Available categories: {sample_categories}"
                )
        
        self.validation_results["annotations_valid"] = len(self.validation_results["issues"]) == 0
        return self.validation_results["annotations_valid"]
    
    def validate_image_accessibility(self, sample_size: int = 10) -> bool:
        """Validate that images are accessible and readable."""
        logger.info(f"🖼️  Validating image accessibility (sampling {sample_size} images)...")
        
        # Load train annotations to get image paths
        train_ann_file = self.base_path / "kaist_mtmdc_train.json"
        if not train_ann_file.exists():
            self.validation_results["issues"].append("❌ Cannot validate images - missing train annotations")
            return False
        
        try:
            with open(train_ann_file, 'r') as f:
                annotations = json.load(f)
            
            images = annotations['images']
            sample_images = np.random.choice(images, min(sample_size, len(images)), replace=False)
            
            accessible_count = 0
            image_stats = {
                "channels": [],
                "dimensions": [],
                "file_sizes_mb": []
            }
            
            for img_info in sample_images:
                # Construct image path (MTMMC specific path structure)
                img_path = self.base_path / "train" / img_info['file_name']
                
                if not img_path.exists():
                    # Try alternative path structure
                    img_path = self.base_path / img_info['file_name']
                
                if img_path.exists():
                    try:
                        # Read image using OpenCV
                        image = cv2.imread(str(img_path))
                        if image is not None:
                            accessible_count += 1
                            
                            # Collect image statistics
                            h, w, c = image.shape
                            image_stats["dimensions"].append((w, h))
                            image_stats["channels"].append(c)
                            image_stats["file_sizes_mb"].append(img_path.stat().st_size / (1024**2))
                        else:
                            self.validation_results["warnings"].append(f"⚠️  Could not read image: {img_path}")
                    except Exception as e:
                        self.validation_results["warnings"].append(f"⚠️  Error reading {img_path}: {e}")
                else:
                    self.validation_results["warnings"].append(f"⚠️  Image not found: {img_path}")
            
            # Calculate statistics
            if image_stats["dimensions"]:
                unique_dims = list(set(image_stats["dimensions"]))
                avg_file_size = np.mean(image_stats["file_sizes_mb"])
                
                self.validation_results["statistics"]["image_stats"] = {
                    "accessible_ratio": accessible_count / len(sample_images),
                    "unique_dimensions": unique_dims,
                    "average_file_size_mb": avg_file_size,
                    "channels": list(set(image_stats["channels"]))
                }
                
                logger.info(f"✅ Image accessibility: {accessible_count}/{len(sample_images)} images readable")
                logger.info(f"📐 Image dimensions: {unique_dims}")
                logger.info(f"📁 Average file size: {avg_file_size:.2f}MB")
            
            self.validation_results["images_accessible"] = accessible_count > 0
            return self.validation_results["images_accessible"]
            
        except Exception as e:
            self.validation_results["issues"].append(f"❌ Error validating images: {e}")
            return False
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not self.validation_results["structure_valid"]:
            recommendations.append("🔧 Fix dataset structure before proceeding with training")
        
        if not self.validation_results["annotations_valid"]:
            recommendations.append("🔧 Fix annotation format issues before training")
        
        # Performance recommendations based on statistics
        stats = self.validation_results["statistics"]
        
        if "annotations" in stats:
            train_stats = stats["annotations"].get("train", {})
            if train_stats.get("num_annotations", 0) < 1000:
                recommendations.append("📊 Consider data augmentation - limited training annotations detected")
            
            # Bounding box analysis
            bbox_stats = train_stats.get("bbox_stats", {})
            if bbox_stats:
                mean_aspect_ratio = bbox_stats.get("mean_aspect_ratio", 0)
                if mean_aspect_ratio > 3.0:
                    recommendations.append("📏 High aspect ratios detected - ensure person detection is optimized for tall/narrow boxes")
                elif mean_aspect_ratio < 1.5:
                    recommendations.append("📏 Low aspect ratios detected - verify person annotations are correct")
        
        if "image_stats" in stats:
            img_stats = stats["image_stats"]
            if img_stats.get("accessible_ratio", 0) < 0.9:
                recommendations.append("🖼️  Some images are inaccessible - verify dataset integrity")
            
            unique_dims = img_stats.get("unique_dimensions", [])
            if len(unique_dims) > 5:
                recommendations.append("📐 Multiple image dimensions detected - ensure proper resizing in data pipeline")
        
        self.validation_results["recommendations"].extend(recommendations)
        return recommendations
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete dataset validation."""
        logger.info("🚀 Starting comprehensive MTMMC dataset validation...")
        
        # Run all validations
        self.validate_dataset_structure()
        self.validate_coco_annotations()
        self.validate_image_accessibility()
        self.generate_recommendations()
        
        # Overall status
        overall_valid = (
            self.validation_results["structure_valid"] and 
            self.validation_results["annotations_valid"] and 
            self.validation_results["images_accessible"]
        )
        
        # Log summary
        logger.info("\n" + "="*50)
        logger.info("📋 DATASET VALIDATION SUMMARY")
        logger.info("="*50)
        
        status_emoji = "✅" if overall_valid else "❌"
        logger.info(f"{status_emoji} Overall Status: {'PASSED' if overall_valid else 'FAILED'}")
        
        logger.info(f"📁 Structure: {'✅ Valid' if self.validation_results['structure_valid'] else '❌ Invalid'}")
        logger.info(f"📋 Annotations: {'✅ Valid' if self.validation_results['annotations_valid'] else '❌ Invalid'}")
        logger.info(f"🖼️  Images: {'✅ Accessible' if self.validation_results['images_accessible'] else '❌ Issues'}")
        
        # Log issues and recommendations
        if self.validation_results["issues"]:
            logger.error("\n🚨 CRITICAL ISSUES:")
            for issue in self.validation_results["issues"]:
                logger.error(f"  {issue}")
        
        if self.validation_results["warnings"]:
            logger.warning("\n⚠️  WARNINGS:")
            for warning in self.validation_results["warnings"]:
                logger.warning(f"  {warning}")
        
        if self.validation_results["recommendations"]:
            logger.info("\n💡 RECOMMENDATIONS:")
            for rec in self.validation_results["recommendations"]:
                logger.info(f"  {rec}")
        
        logger.info("="*50)
        
        return self.validation_results


def validate_mtmmc_dataset(base_path: str) -> Dict[str, Any]:
    """
    Convenience function to validate MTMMC dataset.
    
    Args:
        base_path: Path to MTMMC dataset root
        
    Returns:
        Validation results dictionary
    """
    validator = MTMCDatasetValidator(base_path)
    return validator.run_full_validation()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python dataset_validator.py <path_to_mtmmc_dataset>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    dataset_path = sys.argv[1]
    result = validate_mtmmc_dataset(dataset_path)
    
    exit_code = 0 if (result["structure_valid"] and result["annotations_valid"]) else 1
    sys.exit(exit_code)