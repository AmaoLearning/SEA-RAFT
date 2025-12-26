import sys
import argparse
import os
import re
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config.parser import parse_args
from core.raft import RAFT
from core.utils.utils import load_ckpt


def _numeric_key(filename: str) -> Tuple[int, str]:
    """
    Create a sorting key based on numeric filename (without extension).
    Falls back to lexicographic for non-numeric names.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"\d+", stem)
    if m:
        return (int(m.group()), stem)
    return (0, stem)


def list_cam_dirs(parent: str) -> List[str]:
    """List all subdirectories in parent whose names start with 'cam'."""
    try:
        entries = os.listdir(parent)
    except Exception as e:
        raise RuntimeError(f"Failed to list parent directory '{parent}': {e}")
    cam_dirs: List[str] = []
    for name in entries:
        path = os.path.join(parent, name)
        if os.path.isdir(path) and name.startswith('cam'):
            cam_dirs.append(path)
    cam_dirs.sort(key=lambda p: p)
    return cam_dirs


def list_images(cam_dir: str) -> List[str]:
    """List all PNG images in a cam* directory, sorted by numeric filename."""
    try:
        files = [f for f in os.listdir(cam_dir) if f.lower().endswith('.png')]
    except Exception as e:
        raise RuntimeError(f"Failed to list image directory '{cam_dir}': {e}")
    files.sort(key=lambda f: _numeric_key(f))
    return [os.path.join(cam_dir, f) for f in files]


def read_image_as_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Read an image (PNG) into a (1,3,H,W) float32 RGB tensor on device."""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)  # (3,H,W)
    return img.unsqueeze(0).to(device)


@torch.no_grad()
def forward_flow(args, model: RAFT, image1: torch.Tensor, image2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute forward optical flow using RAFT returning (flow, info)."""
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]  # (N,2,H,W)
    info_final = output['info'][-1]
    return flow_final, info_final


@torch.no_grad()
def calc_flow(args, model: RAFT, image1: torch.Tensor, image2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute flow with internal up/down sampling consistent with custom.py."""
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down


@torch.no_grad()
def warp_backward_by_forward(flow_bwd: torch.Tensor, flow_fwd: torch.Tensor) -> torch.Tensor:
    """
    Warp backward flow field into the coordinate frame of image1 using forward flow.
    - flow_bwd: (N,2,H,W) flow from image2->image1 in image2 coordinates
    - flow_fwd: (N,2,H,W) flow from image1->image2 in image1 coordinates
    Returns: (N,2,H,W) backward flow sampled at positions x+fwd(x), aligned to image1.
    """
    assert flow_bwd.shape == flow_fwd.shape, "Forward and backward flow tensor shapes do not match"
    N, C, H, W = flow_bwd.shape
    # Build base grid of pixel coordinates (x,y)
    ys, xs = torch.meshgrid(torch.arange(H, device=flow_fwd.device), torch.arange(W, device=flow_fwd.device), indexing='ij')
    xs = xs.float()
    ys = ys.float()
    # Forward-warped coordinates in image2
    x2 = xs[None, None, :, :] + flow_fwd[:, 0:1, :, :]
    y2 = ys[None, None, :, :] + flow_fwd[:, 1:2, :, :]
    # Normalize to [-1,1] for grid_sample
    grid_x = 2.0 * (x2 / max(W - 1, 1)) - 1.0
    grid_y = 2.0 * (y2 / max(H - 1, 1)) - 1.0
    grid = torch.stack([grid_x.squeeze(1), grid_y.squeeze(1)], dim=-1)  # (N,H,W,2)
    # Sample backward flow at (x+fwd(x)) to align with image1
    warped_bwd = F.grid_sample(flow_bwd, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped_bwd


def save_flow_numpy(flow: torch.Tensor, out_path: str) -> None:
    """Save flow tensor (N,2,H,W) as HxWx2 numpy array for the first batch item."""
    arr = flow[0].permute(1, 2, 0).detach().cpu().numpy()  # (H,W,2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)


@torch.no_grad()
def process_cam_dir(cam_dir: str, out_root: str, args, model: RAFT, device: torch.device) -> None:
    images = list_images(cam_dir)
    cam_name = os.path.basename(cam_dir)
    out_dir = os.path.join(out_root, cam_name)
    if len(images) < 2:
        print(f"Skipping {cam_dir}: less than 2 images")
        return
    for idx in range(len(images) - 1):
        img1_path = images[idx]
        img2_path = images[idx + 1]
        id_stem = os.path.splitext(os.path.basename(img1_path))[0]
        try:
            img1 = read_image_as_tensor(img1_path, device)
            img2 = read_image_as_tensor(img2_path, device)
        except Exception as e:
            print(f"Failed to read images ({img1_path}, {img2_path}): {e}")
            continue
        # Forward flow: image1 -> image2
        flow_fwd, _ = calc_flow(args, model, img1, img2)
        # Backward flow: image2 -> image1
        flow_bwd_raw, _ = calc_flow(args, model, img2, img1)
        # Warp backward flow using forward flow
        flow_bwd_warped = warp_backward_by_forward(flow_bwd_raw, flow_fwd)
        # Save
        fwd_out = os.path.join(out_dir, f"of_fwd_{id_stem}.npy")
        bwd_out = os.path.join(out_dir, f"of_bwd_{id_stem}.npy")
        try:
            save_flow_numpy(flow_fwd, fwd_out)
            save_flow_numpy(flow_bwd_warped, bwd_out)
        except Exception as e:
            print(f"Failed to save optical flow ({fwd_out}, {bwd_out}): {e}")
            continue
        print(f"Saved: {fwd_out} and {bwd_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--path', help='checkpoint path', type=str, default=None)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    parser.add_argument('--parent', help='Parent directory containing cam* subdirectories', type=str, required=True)
    parser.add_argument('--out_root', help='Output root directory (default: optical_flow)', type=str, default='optical_flow')
    args = parse_args(parser)
    if args.path is None and args.url is None:
        raise ValueError("Either --path or --url must be provided")
    # Build model
    if args.path is not None:
        model = RAFT(args)
        load_ckpt(model, args.path)
    else:
        model = RAFT.from_pretrained(args.url, args=args)
    device = torch.device('cuda' if args.device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()
    # Discover cam* dirs and process
    cam_dirs = list_cam_dirs(args.parent)
    if not cam_dirs:
        raise RuntimeError(f"No cam* subdirectories found in parent directory '{args.parent}'")
    for cam_dir in cam_dirs:
        try:
            process_cam_dir(cam_dir, args.out_root, args, model, device)
        except Exception as e:
            print(f"Failed to process camera directory '{cam_dir}': {e}")
            continue


if __name__ == '__main__':
    main()
