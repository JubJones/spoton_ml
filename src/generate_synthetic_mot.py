"""
Generate a tiny synthetic MOT-style sequence and a matching dets file for dry-run tests.

Creates:
- <sequences_root>/<SEQ>/img1/000001.jpg (black image, size HxW)
- runs/dets_n_embs/<det_stem>/dets/<SEQ>.txt with one detection row

Usage example:
python src/generate_synthetic_mot.py \
  --sequences-root ./synthetic_mot \
  --det-stem yolov8n \
  --seq-name SYN \
  --img-size 480 640 \
  --box 10 10 50 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def main():
    ap = argparse.ArgumentParser(description='Generate synthetic MOT-style sequence and dets')
    ap.add_argument('--sequences-root', type=str, default='./synthetic_mot', help='Root for sequences')
    ap.add_argument('--det-stem', type=str, default='yolov8n', help='Detector stem name')
    ap.add_argument('--seq-name', type=str, default='SYN', help='Sequence name')
    ap.add_argument('--img-size', type=int, nargs=2, default=[480, 640], help='H W for the synthetic frame')
    ap.add_argument('--box', type=int, nargs=4, default=[10, 10, 50, 100], help='x1 y1 x2 y2 detection bbox')
    args = ap.parse_args()

    sequences_root = Path(args.sequences_root)
    seq_dir = sequences_root / args.seq_name
    img_dir = seq_dir / 'img1'
    img_dir.mkdir(parents=True, exist_ok=True)

    H, W = args.img_size
    # black image
    im = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    (img_dir / '000001.jpg').parent.mkdir(parents=True, exist_ok=True)
    im.save(img_dir / '000001.jpg', format='JPEG')

    # dets path
    dets_dir = Path('runs') / 'dets_n_embs' / args.det_stem / 'dets'
    dets_dir.mkdir(parents=True, exist_ok=True)
    det_file = dets_dir / f"{args.seq_name}.txt"

    # header path # BoxMOT writes header as the 'img1' path; include absolute img1 path for consistency
    header = str((img_dir).resolve())
    x1, y1, x2, y2 = args.box
    conf, cls_id = 0.9, 0
    # frame x1 y1 x2 y2 conf cls
    row = f"1 {x1} {y1} {x2} {y2} {conf} {cls_id}\n"
    with open(det_file, 'w') as f:
        f.write(f"# {header}\n")
        f.write(row)

    print(f"Synthetic sequence written to: {seq_dir}")
    print(f"Detections file written to: {det_file}")


if __name__ == '__main__':
    main()

