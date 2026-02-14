"""
ReID Training Runner using Torchreid with Local Results Storage and MLflow Logging

This script fines-tunes an OSNet model for Person Re-Identification using a custom dataset
generated from Ground Truth annotations.

Usage:
    python src/run_training_reid.py
"""

import logging
import sys
import time
import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import cv2
import numpy as np
import mlflow
from tqdm import tqdm

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.reproducibility import set_seed
    from src.utils.logging_utils import setup_logging
    # Check if torchreid is available
    import torchreid
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure torchreid and project modules are installed.")
    sys.exit(1)

# --- Logging Setup ---
log_file = setup_logging(log_prefix="train_reid", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)


class LocalReIDDataset(torchreid.data.datasets.ImageDataset):
    dataset_dir = ''  # Use root directly
    
    def __init__(self, root='', **kwargs):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        
        train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        query_dir = os.path.join(self.dataset_dir, 'query')
        gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')
        
        # Check dirs
        for d in [train_dir, query_dir, gallery_dir]:
            if not os.path.exists(d):
                print(f"Creating missing directory: {d}")
                os.makedirs(d, exist_ok=True)

        train = self.process_dir(train_dir, relabel=True)
        query = self.process_dir(query_dir, relabel=False)
        gallery = self.process_dir(gallery_dir, relabel=False)
        
        super(LocalReIDDataset, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        if not os.path.exists(dir_path):
                print(f"Error: Directory does not exist: {dir_path}")
                return []
                
        img_paths = [os.path.join(dir_path, p) for p in os.listdir(dir_path) if p.endswith(('.jpg', '.jpeg', '.png'))]
        
        data = []
        pid_container = set()
        
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            try:
                parts = img_name.split('_')
                if len(parts) < 3:
                    continue

                pid = int(parts[0])
                cid_str = parts[1]
                if not cid_str.startswith('c'):
                    continue
                    
                cid = int(cid_str.replace('c', ''))
            except Exception as e:
                continue 
            
            if pid == -1: continue
            pid_container.add(pid)
            data.append((img_path, pid, cid))
        
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            data = [(img_path, pid2label[pid], cid) for img_path, pid, cid in data]
            
        return data

# Register immediately at module level
torchreid.data.register_image_dataset('custom_reid', LocalReIDDataset)


class CustomReIDDatamanager:
    """
    Manages the creation and loading of a custom ReID dataset from GT files.
    """
    def __init__(self, root: str, sources: List[str], targets: List[str], 
                 height: int = 256, width: int = 128, batch_size_train: int = 32,
                 batch_size_test: int = 100, workers: int = 4, transforms: List[str] = ['random_flip', 'random_crop']):
        
        self.root = Path(root)
        # Use fixed dataset name 'custom_reid' which maps to LocalReIDDataset
        self.dataset_name = 'custom_reid'
        
        # Initialize torchreid ImageDataManager
        # logic: ImageDataManager will init LocalReIDDataset(root=self.root)
        self.datamanager = torchreid.data.ImageDataManager(
            root=str(self.root),
            sources=[self.dataset_name],
            targets=[self.dataset_name],
            height=height,
            width=width,
            batch_size_train=batch_size_train,
            batch_size_test=batch_size_test,
            transforms=transforms,
            num_instances=4,  # For triplet loss: number of instances per identity in a batch
            workers=workers,
            use_gpu=False # We handle device placement manually in CustomEngine
        )
    
    def _register_dataset(self):
        # Deprecated: registration happens at module level
        pass


class CustomEngine(torchreid.engine.ImageSoftmaxEngine):
    def __init__(self, datamanager, model, optimizer=None, scheduler=None, use_gpu=False, label_smooth=True, device='cpu'):
        super(CustomEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu, label_smooth)
        self.device = device

    def parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        imgs = imgs.to(self.device)
        pids = pids.to(self.device)
        return imgs, pids

    def parse_data_for_test(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        imgs = imgs.to(self.device)
        return imgs, pids, camids

    def extract_features(self, input):
        return self.model(input.to(self.device))


def generate_reid_dataset(config: Dict[str, Any], output_root: Path, dry_run: bool = False) -> Path:
    """
    Generates a ReID dataset from the project's videos and GT files.
    """
    logger.info("Generating ReID dataset from Ground Truth...")
    
    data_config = config.get("data", {})
    base_path = Path(data_config.get("base_path"))
    scenes = data_config.get("scenes_to_include", [])
    min_h = data_config.get("min_height", 128)
    min_w = data_config.get("min_width", 64)
    min_vis = data_config.get("min_visability", 0.3)
    
    # Setup directories
    train_dir = output_root / "bounding_box_train"
    query_dir = output_root / "query"
    gallery_dir = output_root / "bounding_box_test"
    
    if output_root.exists():
        # If dry run and data exists, we can use it. But if it's empty, we might need dummy data.
        pass
    else:
        train_dir.mkdir(parents=True, exist_ok=True)
        query_dir.mkdir(parents=True, exist_ok=True)
        gallery_dir.mkdir(parents=True, exist_ok=True)
    
    total_crops = 0
    pids_collected = set()

    # Dry run with missing data: generate dummy data
    if dry_run and (not base_path.exists() or len(scenes) == 0):
        logger.warning("Dry run with missing/inaccessible data. Generating dummy data for verification.")
        
        # Determine number of dummy identities
        num_identities = 5
        num_cams = 2
        
        for pid in range(num_identities):
            for cam_id in range(num_cams):
                # Create a dummy image
                img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
                
                # Save as train
                filename = f"{pid:04d}_c{cam_id:02d}_000001.jpg"
                cv2.imwrite(str(train_dir / filename), img)
                total_crops += 1
                
                # Save as query/gallery (for even PIDs)
                if pid % 2 == 0:
                    filename_q = f"{pid:04d}_c{cam_id:02d}_000002.jpg"
                    if cam_id == 0:
                        cv2.imwrite(str(query_dir / filename_q), img)
                    else:
                        cv2.imwrite(str(gallery_dir / filename_q), img)
                    total_crops += 1
                    
        return output_root
    
    for scene in scenes:
        scene_id = scene.get("scene_id")
        camera_ids = scene.get("camera_ids", [])
        
        logger.info(f"Processing scene {scene_id}...")
        
        for cam_id in camera_ids:
            # Construct paths
            video_dir = base_path / "train" / scene_id / cam_id / "rgb"
            gt_path = base_path / "train" / scene_id / cam_id / "gt" / "gt.txt"
            
            if not video_dir.exists() or not gt_path.exists():
                logger.warning(f"Skipping {scene_id}/{cam_id}: Missing video or GT.")
                continue
                
            # Read GT
            gt_data = {} 
            with open(gt_path, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split(',')
                        frame = int(parts[0])
                        pid = int(parts[1])
                        x, y, w, h = map(float, parts[2:6])
                        
                        if w < min_w or h < min_h: 
                            continue
                            
                        if frame not in gt_data: gt_data[frame] = []
                        gt_data[frame].append((pid, x, y, w, h))
                    except ValueError:
                        continue
            
            image_files = sorted(list(video_dir.glob("*.jpg")))
            
            # Dry run: process only first 10 frames per cam
            if dry_run:
                image_files = image_files[:10]
            
            for img_file in tqdm(image_files, desc=f"{scene_id}/{cam_id}", disable=dry_run):
                frame_id = int(img_file.stem)
                
                if frame_id in gt_data:
                    img = cv2.imread(str(img_file))
                    if img is None: continue
                    img_h, img_w = img.shape[:2]
                    
                    for (pid, x, y, w, h) in gt_data[frame_id]:
                        x1 = max(0, int(x))
                        y1 = max(0, int(y))
                        x2 = min(img_w, int(x+w))
                        y2 = min(img_h, int(y+h))
                        
                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0: continue
                        
                        # Resize if needed? No, let datamanager handle resize.
                        
                        cid_int = int(cam_id.replace('c', ''))
                        filename = f"{pid:04d}_c{cid_int:02d}_{frame_id:06d}.jpg"
                        
                        if pid % 10 == 0:
                            # Test/Query set
                            if hash(filename) % 2 == 0:
                                target_dir = query_dir
                            else:
                                target_dir = gallery_dir
                        else:
                            target_dir = train_dir
                        
                        cv2.imwrite(str(target_dir / filename), crop)
                        total_crops += 1
                        pids_collected.add(pid)
    
    if total_crops == 0 and dry_run:
        logger.warning("No images extracted during dry run. Generating dummy data for verification.")
        # Determine number of dummy identities
        num_identities = 5
        num_cams = 2
        
        for pid in range(num_identities):
            for cam_id in range(num_cams):
                # Create a dummy image
                img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
                
                # Save as train
                filename = f"{pid:04d}_c{cam_id:02d}_000001.jpg"
                cv2.imwrite(str(train_dir / filename), img)
                total_crops += 1
                
                # Save as query/gallery (for even PIDs)
                if pid % 2 == 0:
                    filename_q = f"{pid:04d}_c{cam_id:02d}_000002.jpg"
                    if cam_id == 0:
                        cv2.imwrite(str(query_dir / filename_q), img)
                    else:
                        cv2.imwrite(str(gallery_dir / filename_q), img)
                    total_crops += 1
                        
    logger.info(f"Dataset generation complete. {total_crops} images extracted.")
    return output_root


def main():
    parser = argparse.ArgumentParser(description="Train ReID Model")
    parser.add_argument("--config", default="configs/reid_training_config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run a dry run/test")
    args = parser.parse_args()

    # Load Configuration
    config = load_config(args.config)
    if not config:
        sys.exit(1)
    
    # Setup Paths
    output_dir = Path(config["data"]["output_dir"])
    project_root = PROJECT_ROOT
    
    # Check if dataset needs generation
    # if not output_dir.exists() or len(list(output_dir.glob("*"))) == 0:
    # Always generate/update? Or check flag? Assuming generate for now as it's a dynamic user request.
    dataset_path = generate_reid_dataset(config, output_dir, dry_run=args.dry_run)
    
    # Initialize Datamanager
    train_config = config.get("training", {})
    datamanager = CustomReIDDatamanager(
        root=str(dataset_path),
        sources=None, # Deprecated
        targets=None,
        height=256,
        width=128,
        batch_size_train=train_config.get("train_batch_size", 32),
        batch_size_test=train_config.get("test_batch_size", 100),
        workers=0 if args.dry_run else 4
    )
    
    # Build Model
    model_config = config.get("model", {})
    model_name = model_config.get("name", "osnet_ain_x1_0")
    pretrained_path = model_config.get("pretrained_weights")
    
    logger.info(f"Building model: {model_name}")
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=datamanager.datamanager.num_train_pids,
        loss=model_config.get("loss", "softmax"),
        pretrained=True # Load imagenet weights first
    )
    
    # Load custom pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading custom pretrained weights from {pretrained_path}")
        torchreid.utils.load_pretrained_weights(model, pretrained_path)
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Optimizer & Scheduler
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=float(train_config.get("learning_rate", 0.0003)),
        weight_decay=float(train_config.get("weight_decay", 5e-4))
    )
    
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=train_config.get("step_size", 20),
        gamma=train_config.get("gamma", 0.1)
    )
    
    # MLflow Setup
    mlflow.set_experiment("reid_training")
    with mlflow.start_run() as run:
        mlflow.log_params(train_config)
        mlflow.log_param("model", model_name)
        
        # Engine
        engine = CustomEngine(
            datamanager.datamanager,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            label_smooth=True,
            device=device
        )
        
        # Training Loop
        max_epoch = 1 if args.dry_run else train_config.get("max_epoch", 60)
        
        engine.run(
            save_dir=str(project_root / "training_output" / "reid" / run.info.run_id),
            max_epoch=max_epoch,
            eval_freq=train_config.get("eval_freq", 5),
            print_freq=train_config.get("print_freq", 10),
            test_only=False,
            fixbase_epoch=train_config.get("fixbase_epoch", 5),
            open_layers=train_config.get("open_layers", ["classifier"])
        )
        
        # Log artifacts
        mlflow.log_artifacts(str(project_root / "training_output" / "reid" / run.info.run_id))

if __name__ == "__main__":
    main()
