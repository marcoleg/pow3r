import argparse
import numpy as np
import torch
import open3d as o3d

import pow3r.tools.path_to_dust3r  # noqa: F401
from dust3r.utils.image import load_images
from dust3r.utils.device import todevice, to_numpy
from pow3r.model import inference as supr


def load_matrix(path, shape):
    """Load a matrix from a text or numpy file."""
    if path.endswith('.npy'):
        data = np.load(path)
    else:
        data = np.loadtxt(path)
    data = np.asarray(data, dtype=np.float32)
    return data.reshape(*shape)


def main(args):
    # load intrinsics and extrinsics
    K = load_matrix(args.intrinsics, (3, 3))
    T1 = load_matrix(args.pose1, (4, 4))
    T2 = load_matrix(args.pose2, (4, 4))

    # load images
    imgs = load_images([args.img1, args.img2], size=args.resolution)
    view1, view2 = todevice(imgs, args.device)

    # attach camera information
    for v, T in zip((view1, view2), (T1, T2)):
        v['camera_intrinsics'] = torch.from_numpy(K)
        v['camera_pose'] = torch.from_numpy(T)
        H, W = v['img'].shape[-2:]
        v['true_shape'] = (H, W)

    # load model
    ckpt = torch.load(args.ckpt, map_location='cpu')
    crop_res = (384, 512)
    slider = supr.AsymmetricSliding(
        crop_res, bootstrap_depth='c2f_both', fix_rays='full', sparsify_depth=1.1
    )
    slider.load_from_checkpoint(ckpt)
    slider = slider.to(args.device)

    with torch.no_grad():
        pred1, pred2 = to_numpy(slider(view1, view2))

    colors1 = view1['img'].reshape(-1, 3).detach().cpu().numpy()
    colors2 = view2['img'].reshape(-1, 3).detach().cpu().numpy()

    pts1 = pred1['pts3d'][0]
    pts2 = pred2['pts3d2'][0]

    pts1_w = (T1[:3, :3] @ pts1.reshape(-1, 3).T + T1[:3, 3:4]).T
    pts2_w = (T2[:3, :3] @ pts2.reshape(-1, 3).T + T2[:3, 3:4]).T

    merged_pts = np.vstack([pts1_w, pts2_w])
    merged_clrs = np.vstack([colors1, colors2])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(merged_clrs, 0, 1))
    o3d.io.write_point_cloud(args.output, pcd)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a point cloud from two images and poses")
    parser.add_argument('--img1', required=True, help='first image path')
    parser.add_argument('--img2', required=True, help='second image path')
    parser.add_argument('--pose1', required=True, help='4x4 pose matrix for img1')
    parser.add_argument('--pose2', required=True, help='4x4 pose matrix for img2')
    parser.add_argument('--intrinsics', required=True, help='3x3 camera intrinsics matrix')
    parser.add_argument('--ckpt', default='checkpoints/Pow3R_ViTLarge_BaseDecoder_512_linear.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution', type=int, default=1280, help='resize images to this size before processing')
    parser.add_argument('--output', default='pow3r_pair.ply', help='output ply filename')
    main(parser.parse_args())
