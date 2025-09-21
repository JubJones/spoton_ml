"""
Minimal ReID training script (simple, self-contained) for MTMMC-derived crops.

Goal
- Train a lightweight ReID model on a Market1501-like folder (bounding_box_train)
- Save a .pt checkpoint with BNNeck embeddings suitable for downstream use

Expected dataset layout (see REID_TRAINING.md):
- <reid_root>/bounding_box_train/*.jpg  filenames like: 0001_c01_sXX_cYY_f000123.jpg
  - PID is the first integer group (e.g., 1)
  - CAMID parsed from "_cXX_" segment (e.g., 1)

Usage examples
- python src/run_reid_training.py --reid-root /path/to/reid_training_output \
    --epochs 20 --batch-size 128 --img-size 256 128 --output-dir runs/reid

Outputs
- Saves: runs/reid/weights/reid_resnet18_bnn.pt (state_dict)
- Also writes a small metadata JSON with class mapping
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
import cv2
import random
import numpy as np

try:
    from torchvision import models, transforms
except Exception as e:
    raise ImportError("torchvision is required for this script. Please install it.")


def set_seed(seed: int = 42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_pid_cam_from_name(name: str) -> Tuple[int, int]:
    """
    Parse PID and CAMID from a filename stem.
    Expected patterns include: '0001_c01_...' or '0012_c16_...'
    - PID = leading integer until first underscore
    - CAMID = integer captured from '_cXX_'
    """
    stem = Path(name).stem
    # PID
    pid_str = stem.split('_')[0]
    try:
        pid = int(pid_str)
    except ValueError:
        raise ValueError(f"Cannot parse PID from filename: {name}")
    # CAMID
    camid = 1
    parts = stem.split('_')
    for i, p in enumerate(parts):
        if p.startswith('c') and len(p) >= 2 and p[1:].isdigit():
            camid = int(p[1:])
            break
    return pid, camid


class MarketLikeDataset(Dataset):
    def __init__(self, root: Path, img_size: Tuple[int, int], augment: bool = True):
        super().__init__()
        self.root = Path(root)
        self.img_dir = self.root / 'bounding_box_train'
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Expected folder not found: {self.img_dir}")

        # List images
        exts = {'.jpg', '.jpeg', '.png'}
        self.paths: List[Path] = [p for p in sorted(self.img_dir.iterdir()) if p.suffix.lower() in exts]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.img_dir}")

        # Build label mapping
        pids = []
        camids = []
        for p in self.paths:
            pid, camid = parse_pid_cam_from_name(p.name)
            pids.append(pid)
            camids.append(camid)
        unique_pids = sorted(set(pids))
        self.pid2label = {pid: i for i, pid in enumerate(unique_pids)}

        self.labels = [self.pid2label[pid] for pid in pids]
        self.camids = camids

        # Transforms
        H, W = img_size
        if augment:
            self.tf = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        x = self.tf(img)
        y = self.labels[idx]
        camid = self.camids[idx]
        return x, y, camid, str(p)


class MTMMCOnTheFlyDataset(Dataset):
    """
    On-the-fly MTMMC ReID dataset using global instance_id from kaist_mtmdc_train.json.
    Filters items by scenes/cameras. Crops bbox per sample at access time.
    """

    def __init__(
        self,
        base_path: Path,
        scenes_to_include: list,
        img_size: Tuple[int, int],
        pid2label: Optional[dict],
        drop_unseen_pids: bool = False,
        split: str = "train",
    ):
        super().__init__()
        assert split in {"train", "val"}
        self.base_path = Path(base_path)
        self.img_size = img_size
        self.split = split

        json_path = self.base_path / "kaist_mtmdc_train.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing MTMMC JSON: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        images = {im["id"]: im for im in data.get("images", [])}
        anns = data.get("annotations", [])

        # Build included scenes/cams map
        include = {}
        for sc in scenes_to_include:
            scene_id = sc.get("scene_id")
            cams = set(sc.get("camera_ids", []))
            if scene_id and cams:
                include[scene_id] = cams

        items = []  # (img_path, pid, camid, (x,y,w,h))
        for a in anns:
            if a.get("category_id") != 1:
                continue
            im = images.get(a.get("image_id"))
            if not im:
                continue
            fn = im.get("file_name", "")  # e.g. 'train/s14/c01/rgb/000000.jpg'
            parts = fn.split('/')
            if len(parts) < 5:
                continue
            split_name, scene, cam = parts[0], parts[1], parts[2]
            if split_name != "train":
                continue
            cams = include.get(scene)
            if cams is None or cam not in cams:
                continue
            pid = a.get("instance_id")
            if pid is None:
                continue
            bbox = a.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            img_path = self.base_path / fn
            if not img_path.exists():
                continue
            camid = int(cam[1:]) if cam.startswith('c') and cam[1:].isdigit() else 1
            items.append((img_path, int(pid), camid, tuple(map(float, bbox))))

        self.items_all = items
        self.pid2label = pid2label

        H, W = img_size
        self.tf = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if drop_unseen_pids and self.pid2label is not None:
            self.items_all = [it for it in self.items_all if it[1] in self.pid2label]

    def __len__(self) -> int:
        return len(self.items_all)

    def __getitem__(self, idx: int):
        p, pid, camid, (x, y, w, h) = self.items_all[idx]
        img = cv2.imread(str(p))
        if img is None:
            H, W = self.img_size
            img = np.zeros((H, W, 3), dtype=np.uint8)
            x1 = y1 = 0
            x2, y2 = W, H
        else:
            H0, W0 = img.shape[:2]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W0, x2), min(H0, y2)
            if x2 <= x1 or y2 <= y1:
                cx, cy = W0 // 2, H0 // 2
                s = min(W0, H0) // 4
                x1, y1, x2, y2 = cx - s, cy - s, cx + s, cy + s
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[y1:y2, x1:x2]

        crop = Image.fromarray(img)
        x_tensor = self.tf(crop)
        label = self.pid2label[pid] if (self.pid2label is not None and pid in self.pid2label) else pid
        return x_tensor, label, camid, str(p)

class BNNeckReID(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512, backbone: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        if backbone == 'resnet18':
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == 'resnet50':
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Project to embedding_dim if needed
        self.proj = nn.Identity() if feat_dim == embedding_dim else nn.Linear(feat_dim, embedding_dim, bias=False)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=True)

    def forward(self, x, return_feat: bool = False):
        f = self.backbone(x)           # (N, feat_dim)
        f = self.proj(f)               # (N, embedding_dim)
        bn = self.bnneck(f)            # (N, embedding_dim)
        logits = self.classifier(bn)   # (N, num_classes)
        if return_feat:
            return bn, logits
        return logits

    @torch.no_grad()
    def extract_features(self, x):
        f = self.backbone(x)
        f = self.proj(f)
        f = self.bnneck(f)
        f = nn.functional.normalize(f, dim=1)
        return f


@dataclass
class TrainConfig:
    reid_root: Optional[Path]
    output_dir: Path
    img_size: Tuple[int, int] = (256, 128)
    backbone: str = 'resnet18'
    embedding_dim: int = 512
    batch_size: int = 4
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 5e-4
    workers: int = 4
    label_smoothing: float = 0.1
    seed: int = 42
    # MTMMC on-the-fly mode
    mtmmc_root: Optional[Path] = None
    reid_config_path: Optional[Path] = None
    val_split_ratio: float = 0.2


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Decide mode
    if cfg.mtmmc_root is not None:
        if cfg.reid_config_path is None:
            raise ValueError("--reid-config must be provided when using --mtmmc-root")
        with open(cfg.reid_config_path, 'r') as f:
            cfg_yaml = yaml.safe_load(f)
        data_cfg = cfg_yaml.get('data', {})
        scenes_to_include = data_cfg.get('scenes_to_include', [])

        # Build aggregate dataset and split
        ds_all = MTMMCOnTheFlyDataset(
            base_path=cfg.mtmmc_root,
            scenes_to_include=scenes_to_include,
            img_size=cfg.img_size,
            pid2label=None,
            split="train",
        )
        n = len(ds_all.items_all)
        idxs = list(range(n))
        random.Random(cfg.seed).shuffle(idxs)
        n_val = int(n * cfg.val_split_ratio)
        val_idxs = set(idxs[:n_val])
        train_items = [ds_all.items_all[i] for i in range(n) if i not in val_idxs]
        val_items = [ds_all.items_all[i] for i in range(n) if i in val_idxs]

        train_pids = sorted({pid for _, pid, _, _ in train_items})
        pid2label = {pid: i for i, pid in enumerate(train_pids)}

        ds_train = MTMMCOnTheFlyDataset(
            base_path=cfg.mtmmc_root,
            scenes_to_include=scenes_to_include,
            img_size=cfg.img_size,
            pid2label=pid2label,
            drop_unseen_pids=False,
            split="train",
        )
        ds_train.items_all = train_items
        ds_train.pid2label = pid2label

        ds_val = MTMMCOnTheFlyDataset(
            base_path=cfg.mtmmc_root,
            scenes_to_include=scenes_to_include,
            img_size=cfg.img_size,
            pid2label=pid2label,
            drop_unseen_pids=True,
            split="val",
        )
        ds_val.items_all = [it for it in val_items if it[1] in pid2label]
        num_classes = len(pid2label)
        dl = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    else:
        if cfg.reid_root is None:
            raise ValueError("--reid-root must be provided in pre-cropped mode")
        ds = MarketLikeDataset(cfg.reid_root, cfg.img_size, augment=True)
        num_classes = len(set(ds.labels))
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)
        dl_val = None

    model = BNNeckReID(num_classes=num_classes, embedding_dim=cfg.embedding_dim, backbone=cfg.backbone, pretrained=True).to(device)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = cfg.output_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    for epoch in range(cfg.epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for batch in dl:
            imgs, labels, camids, paths = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs, return_feat=False)
            loss = ce(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            running_correct += (preds == labels).sum().item()
            running_total += imgs.size(0)
            global_step += 1

        epoch_loss = running_loss / max(1, running_total)
        epoch_acc = running_correct / max(1, running_total)
        msg = f"Epoch {epoch+1}/{cfg.epochs} - loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}"
        if 'dl_val' in locals() and dl_val is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in dl_val:
                    imgs, labels, camids, paths = batch
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    logits = model(imgs, return_feat=False)
                    loss = ce(logits, labels)
                    val_loss += loss.item() * imgs.size(0)
                    preds = logits.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += imgs.size(0)
            if val_total > 0:
                msg += f" | val_loss: {val_loss/val_total:.4f} val_acc: {val_correct/val_total:.4f}"
            model.train()
        print(msg)

    # Save checkpoint and metadata
    ckpt_path = weights_dir / f"reid_{cfg.backbone}_bnn.pt"
    torch.save(model.state_dict(), ckpt_path)
    # Determine pid2label mapping for metadata
    meta_pid2label = None
    if 'pid2label' in locals():
        meta_pid2label = pid2label
    else:
        try:
            meta_pid2label = ds.pid2label  # type: ignore[name-defined]
        except Exception:
            meta_pid2label = None

    meta = {
        'num_classes': num_classes,
        'embedding_dim': cfg.embedding_dim,
        'backbone': cfg.backbone,
        'img_size': cfg.img_size,
        'label_smoothing': cfg.label_smoothing,
        'pid2label': meta_pid2label,  # useful for later mapping
    }
    with open(weights_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved weights: {ckpt_path}")
    print(f"Saved metadata: {weights_dir / 'metadata.json'}")


def parse_args():
    ap = argparse.ArgumentParser(description='Minimal ReID training script')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--reid-root', type=str, help='Path with bounding_box_train under it (pre-cropped mode)')
    group.add_argument('--mtmmc-root', type=str, help='Path to MTMMC root (on-the-fly mode)')
    ap.add_argument('--reid-config', type=str, default='configs/reid_training_config.yaml', help='ReID YAML with data.scenes_to_include (on-the-fly)')
    ap.add_argument('--output-dir', type=str, default='runs/reid', help='Output dir for weights/logs')
    ap.add_argument('--img-size', type=int, nargs=2, default=[256, 128], help='H W image size')
    ap.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    ap.add_argument('--embedding-dim', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--label-smoothing', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-split-ratio', type=float, default=0.2)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        reid_root=Path(args.reid_root) if args.reid_root else None,
        output_dir=Path(args.output_dir),
        img_size=(args.img_size[0], args.img_size[1]),
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        mtmmc_root=Path(args.mtmmc_root) if args.mtmmc_root else None,
        reid_config_path=Path(args.reid_config) if args.mtmmc_root else None,
        val_split_ratio=float(args.val_split_ratio),
    )
    train(cfg)


if __name__ == '__main__':
    main()
