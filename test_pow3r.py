import os
import json
import numpy as np
import torch
import pow3r.tools.path_to_dust3r
from dust3r.utils.image import load_images
from dust3r.utils.device import todevice, to_numpy
from pow3r.model import inference as supr
from tqdm import tqdm
import open3d as o3d  # per salvare point cloud .ply

# === PARAMETRI ===
IMG_DIR = '/mnt/Datasets/DL3DV-140-Benchmark/8324b3ca22085040c2a0ecb7284e0cdf776b1f846b73a7c0df893587cb4a45f8/gaussian_splat/images_4/'
POSE_JSON = '/mnt/Datasets/DL3DV-140-Benchmark/8324b3ca22085040c2a0ecb7284e0cdf776b1f846b73a7c0df893587cb4a45f8/gaussian_splat/transforms.json'  # formato: {"0001.jpg": [[4x4]], "0002.jpg": [[4x4]], ...}
CKPT_PATH = 'checkpoints/Pow3R_ViTLarge_BaseDecoder_512_linear.pth'
DEVICE = 'cuda'
RESOLUTION = 1280
K = None

# === LOAD POSES ===
with open(POSE_JSON) as f:
    loaded = json.load(f)
    if K is None:
        w = loaded['w']
        h = loaded['h']
        fx = loaded['fl_x']
        fy = loaded['fl_y']
        cx = loaded['cx']
        cy = loaded['cy']
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    poses = {}
    for k in range(len(loaded['frames'])):
        poses[loaded['frames'][k]['file_path'].split('/')[1]] = loaded['frames'][k]['transform_matrix']

# === IMAGES ===
img_list = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg') or f.endswith('.png')])

# === MODEL ===
crop_res = (384, 512)
ckpt = torch.load(CKPT_PATH, map_location='cpu')
slider = supr.AsymmetricSliding(crop_res, bootstrap_depth='c2f_both', fix_rays='full', sparsify_depth=1.1)
slider.load_from_checkpoint(ckpt)
slider = slider.to(DEVICE)

all_pts, all_clrs = [], []

for i in tqdm(range(len(img_list) - 1)):
    if i > 21: break
    name1, name2 = img_list[i], img_list[i + 1]
    img1_path, img2_path = os.path.join(IMG_DIR, name1), os.path.join(IMG_DIR, name2)

    # carica immagini
    imgs = load_images([img1_path, img2_path], size=RESOLUTION)
    view1, view2 = todevice(imgs, DEVICE)

    colors_valid1 = view1['img'].reshape(-1, 3)
    colors_valid2 = view2['img'].reshape(-1, 3)

    # registra nel mondo usando le pose note
    T1 = np.array(poses[name1]).reshape(4, 4)  # 4x4
    T2 = np.array(poses[name2]).reshape(4, 4)  # 4x4

    view1['camera_pose'] = torch.from_numpy(T1)
    view2['camera_pose'] = torch.from_numpy(T2)

    view1['camera_intrinsics'] = torch.from_numpy(K)
    view2['camera_intrinsics'] = torch.from_numpy(K)

    view1['true_shape'] = (h, w)
    view2['true_shape'] = (h, w)

    # inferenza
    with torch.no_grad():
        pred1, pred2 = to_numpy(slider(view1, view2))

    # estrai pointmap X^{1,1} e X^{2,2}
    pts1 = pred1['pts3d'][0]  # shape (H, W, 3)
    pts2 = pred2['pts3d2'][0]  # X^{2,2}, nel frame 2

    pts1_world = (T1[:3,:3] @ pts1.reshape(-1,3).T + T1[:3,3:4]).T
    pts2_world = (T2[:3,:3] @ pts2.reshape(-1,3).T + T2[:3,3:4]).T

    all_pts.append(pts1_world)
    all_pts.append(pts2_world)
    all_clrs.append(colors_valid1.detach().cpu())
    all_clrs.append(colors_valid2.detach().cpu())

# === UNISCI E SALVA
merged_pts = np.vstack(all_pts)
merged_clrs = np.vstack(all_clrs)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(merged_pts)
pcd.colors = o3d.utility.Vector3dVector(np.clip(merged_clrs, 0, 1))  # (N, 3) in [0, 1]
o3d.io.write_point_cloud("/home/isarlab/PycharmProjects/pow3r/pow3r_global.ply", pcd)
o3d.visualization.draw_geometries([pcd])
print("Saved pow3r_global.ply")
