"""
ReID Fine-tuning Runner

This script fine-tunes a specific ReID model (e.g., OSNet) using Ground Truth IDs from the dataset.
It follows the project's standard structure for configuration, logging, and MLflow integration.

Usage:
    python src/run_training_reid_finetune.py --config configs/reid_finetune_config.yaml
"""

import logging
import sys
import time
import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import cv2
import numpy as np
import mlflow
from tqdm import tqdm
import torch

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.reproducibility import set_seed
    from src.utils.logging_utils import setup_logging
    # MLflow and Runner Utilities
    from src.utils.mlflow_utils import setup_mlflow_experiment, log_artifacts
    from src.utils.runner import log_params_recursive, log_git_info
    
    # Try importing torchreid
    try:
        import torchreid
    except ImportError:
        print("Error: torchreid not installed. Please install it to use this script.")
        print("pip install torchreid")
        sys.exit(1)
        
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

# --- Logging Setup ---
log_file = setup_logging(log_prefix="train_reid_finetune", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)


# --- Custom Dataset Class ---
class LocalReIDDataset(torchreid.data.datasets.ImageDataset):
    """
    Custom Dataset class for Torchreid that loads data from our generated directory structure.
    Expected structure:
        dataset_dir/
            bounding_box_train/
            bounding_box_test/
            query/
    """
    dataset_dir = ''  # Will be set dynamically
    
    def __init__(self, root='', **kwargs):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = self.root # Root IS the dataset dir in our case
        
        train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        query_dir = os.path.join(self.dataset_dir, 'query')
        gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')
        
        required_dirs = [train_dir, query_dir, gallery_dir]
        for d in required_dirs:
            if not os.path.exists(d):
                logger.warning(f"Directory not found: {d}. Creating it (might be empty).")
                os.makedirs(d, exist_ok=True)

        train = self.process_dir(train_dir, relabel=True)
        query = self.process_dir(query_dir, relabel=False)
        gallery = self.process_dir(gallery_dir, relabel=False)
        
        super(LocalReIDDataset, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        if not os.path.exists(dir_path):
            return []
            
        img_paths = [os.path.join(dir_path, p) for p in os.listdir(dir_path) if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        data = []
        pid_container = set()
        
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            # Expected format: {pid:04d}_c{cam_id:02d}_{frame_id:06d}.jpg
            try:
                parts = img_name.split('_')
                if len(parts) < 3:
                    continue

                pid = int(parts[0])
                cid_str = parts[1]
                
                if cid_str.startswith('c'):
                    cid = int(cid_str.replace('c', ''))
                else:
                    cid = 0 # Default if no camera ID found
                
                if pid == -1: continue # Junk images
                
                pid_container.add(pid)
                data.append((img_path, pid, cid))
            except Exception as e:
                logger.debug(f"Skipping malformed filename {img_name}: {e}")
                continue 
        
        if relabel:
            # Map PIDs to 0-N range for training
            pid2label = {pid: label for label, pid in enumerate(sorted(list(pid_container)))}
            data = [(img_path, pid2label[pid], cid) for img_path, pid, cid in data]
            
        return data

# Register the dataset immediately
try:
    torchreid.data.register_image_dataset('custom_reid_finetune', LocalReIDDataset)
except ValueError:
    # Already registered
    pass


def generate_dataset_from_gt(config: Dict[str, Any], output_root: Path, dry_run: bool = False):
    """
    Parses GT files and generates ID-injected crops for ReID training.
    """
    logger.info("--- Generating ReID Dataset from Ground Truth ---")
    
    data_config = config.get("data", {})
    base_path = Path(data_config.get("base_path", "D:/MTMMC"))
    scenes = data_config.get("scenes_to_include", [])
    min_h = data_config.get("min_height", 128)
    min_w = data_config.get("min_width", 64)
    
    # Setup directories
    train_dir = output_root / "bounding_box_train"
    query_dir = output_root / "query"
    gallery_dir = output_root / "bounding_box_test"
    
    # Clean up previous generation if strict? For now, we'll overwrite/add.
    # Actually, it's safer to clean to avoid stale IDs
    if output_root.exists() and not dry_run:
         logger.info(f"Cleaning existing output directory: {output_root}")
         shutil.rmtree(output_root)
    
    train_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    total_crops = 0
    pids_collected: Set[int] = set()
    
    # Handle Dry Run with dummy data if actual data is missing
    if dry_run and not base_path.exists():
        logger.warning("Dry run detected and base path missing. Generating dummy data.")
        _generate_dummy_data(train_dir, query_dir, gallery_dir)
        return

    for scene in scenes:
        scene_id = scene.get("scene_id")
        camera_ids = scene.get("camera_ids", [])
        
        logger.info(f"Processing Scene: {scene_id} | Cameras: {camera_ids}")
        
        for cam_id in camera_ids:
            # Construct paths
            # Structure: base_path/train/sXX/cYY/rgb/*.jpg AND .../gt/gt.txt
            scene_dir = base_path / "train" / scene_id
            camera_dir = scene_dir / cam_id
            rgb_dir = camera_dir / "rgb"
            gt_file = camera_dir / "gt" / "gt.txt"
            
            if not rgb_dir.exists() or not gt_file.exists():
                logger.warning(f"  Missing data for {scene_id}/{cam_id}. Skipping.")
                continue
                
            # 1. Parse GT
            gt_data = {} # frame_id -> list of (pid, x, y, w, h)
            with open(gt_file, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split(',')
                        # Format: frame, id, x, y, w, h
                        frame_idx = int(parts[0])
                        pid = int(parts[1])
                        x, y, w, h = map(float, parts[2:6])
                        
                        # Filter small boxes
                        if w < min_w or h < min_h: 
                            continue
                            
                        if frame_idx not in gt_data:
                            gt_data[frame_idx] = []
                        gt_data[frame_idx].append((pid, x, y, w, h))
                    except ValueError:
                        continue
            
            # 2. Process Images
            image_files = sorted(list(rgb_dir.glob("*.jpg")))
            
            if dry_run:
                image_files = image_files[:20] # Limit frames for dry run
            
            logger.info(f"  Processing {len(image_files)} frames for {scene_id}/{cam_id}...")
            
            for img_path in tqdm(image_files, desc=f"  {scene_id}/{cam_id}", disable=dry_run):
                frame_idx = int(img_path.stem)
                
                if frame_idx not in gt_data:
                    continue
                    
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                height, width = img.shape[:2]
                
                for (pid, x, y, w, h) in gt_data[frame_idx]:
                    # Crop
                    x1, y1 = max(0, int(x)), max(0, int(y))
                    x2, y2 = min(width, int(x + w)), min(height, int(y + h))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    crop = img[y1:y2, x1:x2]
                    
                    # Save
                    cid_int = int(cam_id.replace('c', ''))
                    filename = f"{pid:04d}_c{cid_int:02d}_{frame_idx:06d}.jpg"
                    
                    # Split logic: simplified. 
                    # If PID % 10 == 0 -> Test (Query/Gallery). Else -> Train.
                    # This ensures disjoint identities between train and test.
                    if pid % 10 == 0:
                        # For test set, we need query and gallery.
                        # Simple split: even frames gallery, odd frames query? 
                        # Or random? Let's do hash based.
                        if hash(filename) % 2 == 0:
                            target = gallery_dir
                        else:
                            target = query_dir
                    else:
                        target = train_dir
                    
                    cv2.imwrite(str(target / filename), crop)
                    total_crops += 1
                    pids_collected.add(pid)

    logger.info(f"Dataset Generation Complete. Total Crops: {total_crops}. Total Identities: {len(pids_collected)}")
    
    if total_crops == 0 and dry_run:
        logger.warning("No crops generated in dry run. Generating dummy data.")
        _generate_dummy_data(train_dir, query_dir, gallery_dir)

def _generate_dummy_data(train_dir, query_dir, gallery_dir):
    """Helper to generate dummy data for testing pipeline"""
    for pid in range(50): # Increased to 50 to satisfy Rank-20 evaluation
        for cam in range(2):
            img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            fname = f"{pid:04d}_c{cam:02d}_000001.jpg"
            cv2.imwrite(str(train_dir / fname), img)
            
            if pid % 2 == 0:
                fname_q = f"{pid:04d}_c{cam:02d}_000002.jpg"
                if cam == 0:
                    cv2.imwrite(str(query_dir / fname_q), img)
                else:
                    cv2.imwrite(str(gallery_dir / fname_q), img)

# --- Custom Engine for MPS Support (Optional) ---
class CustomEngine(torchreid.engine.ImageSoftmaxEngine):
    def __init__(self, datamanager, model, optimizer=None, scheduler=None, use_gpu=False, label_smooth=True, device='cpu'):
        super(CustomEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu, label_smooth)
        self.device = device
        self.max_epoch = 0 # Default to 0, will be updated or used for manual loop
        self.epoch = 0
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader

    def parse_data_for_train(self, data):
        if isinstance(data, dict):
            imgs = data['img']
            pids = data['pid']
        else:
            imgs = data[0]
            pids = data[1]
        
        imgs = imgs.to(self.device)
        pids = pids.to(self.device)
        return imgs, pids

    def parse_data_for_test(self, data):
        if isinstance(data, dict):
            imgs = data['img']
            pids = data['pid']
            camids = data['camid']
        else:
            imgs = data[0]
            pids = data[1]
            camids = data[2]
        
        imgs = imgs.to(self.device)
        return imgs, pids, camids

# --- MLflow Writer for Torchreid ---
class MLflowWriter:
    def __init__(self, log_dir):
        pass # No setup needed for MLflow
        
    def add_scalar(self, name, value, global_step):
        mlflow.log_metric(name, value, step=global_step)
        
    def close(self):
        pass

def main():
    parser = argparse.ArgumentParser(description="ReID Fine-tuning")
    parser.add_argument("--config", default="configs/reid_finetune_config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run a dry run/test")
    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args.config)
    if not config:
        sys.exit(1)
        
    set_seed(config.get("training", {}).get("seed", 42))
    
    # 2. Generate Dataset
    output_dir = Path(config["data"]["output_dir"])
    if args.dry_run:
        output_dir = output_dir.with_name(output_dir.name + "_dryrun")
        logger.info(f"Dry run: Using separate output directory {output_dir}")
        if output_dir.exists():
            shutil.rmtree(output_dir) # Clean start for dry run
    
    generate_dataset_from_gt(config, output_dir, dry_run=args.dry_run)
    generate_dataset_from_gt(config, output_dir, dry_run=args.dry_run)
    
    # 3. Initialize Data Manager
    train_config = config.get("training", {})
    
    logger.info("Initializing Data Manager...")
    datamanager = torchreid.data.ImageDataManager(
        root=str(output_dir),
        sources=['custom_reid_finetune'],
        targets=['custom_reid_finetune'],
        height=256,
        width=128,
        batch_size_train=train_config.get("train_batch_size", 32),
        batch_size_test=train_config.get("test_batch_size", 100),
        transforms=['random_flip', 'random_crop'],
        num_instances=4, 
        workers=0 if args.dry_run else 4 
    )
    
    # 4. Build Model
    model_config = config.get("model", {})
    model_name = model_config.get("name", "osnet_ain_x1_0")
    pretrained_weights = model_config.get("pretrained_weights")
    
    logger.info(f"Building Model: {model_name}")
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=datamanager.num_train_pids,
        loss=model_config.get("loss", "softmax"),
        pretrained=True # Start with ImageNet weights
    )
    
    # Load custom weights if available (Fine-tuning start point)
    if pretrained_weights and os.path.exists(pretrained_weights):
        logger.info(f"Loading custom pretrained weights: {pretrained_weights}")
        torchreid.utils.load_pretrained_weights(model, pretrained_weights)
    else:
        logger.warning(f"Custom weights not found at {pretrained_weights}. Using ImageNet weights only.")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # if torch.backends.mps.is_available():
    #     device = 'mps' # MPS disabled due to compatibility issues with torchreid
    logger.info(f"Using device: {device}")
    model = model.to(device)

    # 5. Optimizer & Engine
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
    
    # Use CustomEngine for better device handling (GPU/MPS/CPU)
    engine = CustomEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        use_gpu=(device == 'cuda'), # Only use torchreid's internal gpu handling for CUDA
        device=device
    )
    
    # 6. Training
    experiment_id = setup_mlflow_experiment(config, "reid_finetune")
    if not experiment_id:
        logger.error("Failed to setup MLflow experiment.")
        return

    with mlflow.start_run(experiment_id=experiment_id) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Consistent logging with YOLO runner
        logger.info("Logging parameters and git info...")
        log_params_recursive(config)
        log_git_info()
        
        # Additional specific tags
        mlflow.set_tag("model", model_name)
        if pretrained_weights:
            mlflow.set_tag("pretrained_weights", pretrained_weights)
            
        logger.info("Starting Training...")
        max_epoch = 1 if args.dry_run else train_config.get("max_epoch", 60)
        
        # Manual setup for engine (normally handled by engine.run)
        engine.max_epoch = max_epoch
        engine.train_loader = datamanager.train_loader
        engine.test_loader = datamanager.test_loader
        engine.pbar = True
        
        # 1. Setup Save Dir
        save_dir = str(PROJECT_ROOT / "training_output" / "reid_finetune" / run.info.run_id)
        os.makedirs(save_dir, exist_ok=True)
        
        engine.writer = MLflowWriter(save_dir) # Inject MLflow writer!
        
        start_epoch = 0 # Can be loaded from resume
        eval_freq = train_config.get("eval_freq", 5)
        fixbase_epoch = 0 # Disabled for stability
        open_layers = train_config.get("open_layers", ["classifier"])
        
        best_rank1 = 0.0
        logger.info(f"Start epoch: {start_epoch}, Max epoch: {max_epoch}")
        
        for epoch in range(start_epoch, max_epoch):
            # --- Training ---
            engine.epoch = epoch # Manually set epoch for engine
            engine.train(
                print_freq=train_config.get("print_freq", 10),
                fixbase_epoch=fixbase_epoch, 
                open_layers=open_layers
            )
            
            # --- Validation ---
            if (epoch + 1) % eval_freq == 0 or (epoch + 1) == max_epoch:
                logger.info(f"Evaluating at epoch {epoch + 1}")
                rank1 = engine.test(
                    dist_metric='euclidean',
                    normalize_feature=False,
                    visrank=False,
                    save_dir=save_dir
                )
                
                # Check for best model
                is_best = rank1 > best_rank1
                if is_best:
                    best_rank1 = rank1
                    logger.info(f"New best model! Rank-1: {rank1:.1%}")
                
                # Save checkpoint
                engine.save_model(epoch, rank1, save_dir, is_best=is_best)
                
                # Log checkpoints to MLflow as artifacts (consistent with YOLO runner)
                log_artifacts(save_dir, artifact_directory="checkpoints")
                
        
        logger.info("Training Complete.")
                
        
        logger.info("Training Complete.")

if __name__ == "__main__":
    main()
