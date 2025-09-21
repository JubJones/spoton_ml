"""
Write offline embeddings for BoxMOT from trained ReID weights.

Inputs
- --sequences-root: path with MOT-like sequences (<root>/<SEQ>/img1/000001.jpg)
- --dets-root: runs/dets_n_embs/<det_stem>/dets (produced by `boxmot generate`)
- --reid-weights: e.g., runs/reid/weights/reid_resnet18_bnn.pt
- --reid-stem: folder name under embs/, must match `--reid-model` stem passed to BoxMOT later

Outputs
- runs/dets_n_embs/<det_stem>/embs/<reid_stem>/<SEQ>.txt with one L2-normalized embedding per detection row

Example
python src/write_offline_embeddings.py \
  --sequences-root ./data/MOT17/train \
  --dets-root runs/dets_n_embs/yolov8n/dets \
  --reid-weights runs/reid/weights/reid_resnet18_bnn.pt \
  --reid-stem reid_resnet18_bnn \
  --img-size 256 128
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import numpy as np
import torch
from PIL import Image
import cv2

try:
    from torchvision import transforms
except Exception as e:
    raise ImportError("torchvision is required. Please install it.")


def add_project_root_to_path():
    this = Path(__file__).resolve()
    root = this.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


add_project_root_to_path()

try:
    from src.run_reid_training import BNNeckReID
except Exception:
    # Fallback local definition if import fails (kept minimal)
    import torch.nn as nn
    from torchvision import models

    class BNNeckReID(nn.Module):
        def __init__(self, num_classes: int, embedding_dim: int = 512, backbone: str = 'resnet18', pretrained: bool = False):
            super().__init__()
            if backbone == 'resnet18':
                base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
                feat_dim = base.fc.in_features
                base.fc = nn.Identity()
                self.backbone = base
            else:
                base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
                feat_dim = base.fc.in_features
                base.fc = nn.Identity()
                self.backbone = base
            self.proj = nn.Identity() if feat_dim == embedding_dim else nn.Linear(feat_dim, embedding_dim, bias=False)
            self.bnneck = nn.BatchNorm1d(embedding_dim)
            self.bnneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(embedding_dim, num_classes, bias=True)

        @torch.no_grad()
        def extract_features(self, x):
            f = self.backbone(x)
            f = self.proj(f)
            f = self.bnneck(f)
            f = torch.nn.functional.normalize(f, dim=1)
            return f


def load_model_and_meta(weights_path: Path, device: torch.device, embedding_dim: int | None, backbone: str | None):
    # Try to load metadata.json from the same folder
    meta_path = weights_path.parent / 'metadata.json'
    meta = {}
    if meta_path.exists():
        try:
            meta = json.load(open(meta_path, 'r'))
        except Exception:
            meta = {}
    emb_dim = embedding_dim or int(meta.get('embedding_dim', 512))
    bb = backbone or meta.get('backbone', 'resnet18')

    model = BNNeckReID(num_classes=int(meta.get('num_classes', 1)), embedding_dim=emb_dim, backbone=bb, pretrained=False).to(device)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, emb_dim, meta


def preprocess_batch(rgb_crops: list[Image.Image], img_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    H, W = img_size
    tf = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    xs = [tf(im) for im in rgb_crops]
    batch = torch.stack(xs, dim=0).to(device)
    return batch


def write_embeddings_for_sequence(seq_name: str, seq_dir: Path, dets_path: Path, out_path: Path, model, img_size: tuple[int, int], device: torch.device):
    # Ensure parent exists and truncate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    # Load dets, ignoring header
    dets = np.loadtxt(dets_path, comments='#')
    if dets.ndim == 1 and dets.size > 0:
        dets = dets.reshape(1, -1)
    if dets.size == 0:
        # Nothing to do
        out_path.touch()
        return

    frame_ids = dets[:, 0].astype(int)
    N = dets.shape[0]
    order = np.arange(N)
    # Sort by (frame, original order) to batch per frame but preserve order later
    sort_idx = np.lexsort((order, frame_ids))

    feats_out = np.zeros((N, 512), dtype=np.float32)  # will resize later once we see first batch

    i = 0
    while i < N:
        j = i
        fid = frame_ids[sort_idx[i]]
        idxs = []
        while j < N and frame_ids[sort_idx[j]] == fid:
            idxs.append(sort_idx[j])
            j += 1

        # Load frame image
        img_path = seq_dir / 'img1' / f"{fid:06d}.jpg"
        im_bgr = cv2.imread(str(img_path))
        if im_bgr is None:
            # Create a black image placeholder if missing
            im_bgr = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        crops = []
        for k in idxs:
            _, x1, y1, x2, y2, *_ = dets[k]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(im_rgb.shape[1], x2), min(im_rgb.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            else:
                crop = im_rgb[y1:y2, x1:x2]
            crops.append(Image.fromarray(crop))

        batch = preprocess_batch(crops, img_size, device)
        with torch.no_grad():
            f = model.extract_features(batch)  # (M, D)
            f = f.detach().cpu().numpy().astype(np.float32)
        if feats_out.shape[1] != f.shape[1]:
            feats_out = np.zeros((N, f.shape[1]), dtype=np.float32)
        feats_out[idxs, :] = f
        i = j

    # Write in original order
    with open(out_path, 'wb') as f:
        np.savetxt(f, feats_out, fmt='%f')


def infer_det_stem(dets_root: Path) -> str:
    # runs/dets_n_embs/<det_stem>/dets
    if dets_root.name != 'dets':
        raise ValueError("--dets-root must end with .../<det_stem>/dets")
    return dets_root.parent.name


def main():
    ap = argparse.ArgumentParser(description='Write offline embeddings for BoxMOT')
    ap.add_argument('--sequences-root', type=str, required=True, help='Root with <SEQ>/img1/*.jpg')
    ap.add_argument('--dets-root', type=str, required=True, help='Path to runs/dets_n_embs/<det_stem>/dets')
    ap.add_argument('--reid-weights', type=str, required=True, help='Path to trained ReID weights .pt')
    ap.add_argument('--reid-stem', type=str, required=True, help='Embeddings folder name (must match --reid-model stem in BoxMOT)')
    ap.add_argument('--img-size', type=int, nargs=2, default=None, help='H W image size for ReID model (defaults to training metadata if present)')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--embedding-dim', type=int, default=None)
    ap.add_argument('--backbone', type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    weights_path = Path(args.reid_weights)
    model, emb_dim, meta = load_model_and_meta(weights_path, device, args.embedding_dim, args.backbone)

    dets_root = Path(args.dets_root)
    det_stem = infer_det_stem(dets_root)
    sequences_root = Path(args.sequences_root)
    embs_root = dets_root.parent / 'embs' / args.reid_stem
    embs_root.mkdir(parents=True, exist_ok=True)

    # Iterate sequences by det files present
    for det_file in sorted(dets_root.glob('*.txt')):
        seq_name = det_file.stem
        seq_dir = sequences_root / seq_name
        if not (seq_dir / 'img1').exists():
            print(f"[WARN] Missing img1 for sequence {seq_name} in {seq_dir}, skipping")
            continue
        out_path = embs_root / f"{seq_name}.txt"
        # decide image size: CLI > metadata > default
        if args.img_size is not None:
            img_size = (args.img_size[0], args.img_size[1])
        else:
            ms = meta.get('img_size', [256, 128]) if isinstance(meta, dict) else [256, 128]
            img_size = (int(ms[0]), int(ms[1]))

        print(f"Writing embeddings for {seq_name} → {out_path} (img_size={img_size})")
        write_embeddings_for_sequence(
            seq_name=seq_name,
            seq_dir=seq_dir,
            dets_path=det_file,
            out_path=out_path,
            model=model,
            img_size=img_size,
            device=device,
        )

    print(f"Done. Embeddings under: {embs_root}")


if __name__ == '__main__':
    main()
