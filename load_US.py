#check files exist in folder 
import os
from os import path
from os.path import isfile, isdir, join

import torch as tr
import numpy as np

# https://zhuanlan.zhihu.com/p/593204605/ 去看

# def load_ultra_data(basedir, testskip=1):
#     # find a folder called "images"
#     img_fld = [f for f in os.listdir(basedir) if isdir(join(basedir, f))]
    
#     #for ultra-nerf dataset
    
#     trnda_arr = ''
    
#     skip = 0
    
#     for i in img_fld:
#         # if "train" in i :
#         #    skip = 8
#         #    basedir = join(basedir, i)
#         #    break
#         # else:
#         trnda_arr = [join(basedir, i) for i in img_fld if "train" not in i]
#         print(trnda_arr)
        
#     # if traindata:
#     #     return trnda_arr, skip
#     # else:
#     #     return trnda_arr, skip
#     return trnda_arr, skip
device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

def load_image_by_path(basedir, dirname):
    # if type(basedir) is list:
    #     print("Yes")
    # elif type(basedir) is str:
    imgdir = join(basedir, 'images')
        
    npy_file = [f for f in os.listdir(basedir) if path.splitext(f)[1] == ".npy"]
    # print(len(npy_file)) # ['images.npy', 'poses.npy']
        
    img_meta = np.load(join(basedir, npy_file[0])) # (800, 512, 256)
    pose_meta = np.load(join(basedir, npy_file[1]))
        
    H, W = img_meta.shape[1], img_meta.shape[2]
        
    c2w = poses_avg(img_meta, pose_meta)
        
    # dists = np.sum(np.square(c2w[:3, 3] - pose_meta[:, :3, 3]), -1)
        
    images = img_meta.astype(np.float32)
    poses = pose_meta.astype(np.float32)
    
    image_datas = []
    for file in os.listdir(imgdir):
        if file.endswith(".png"):
            image_datas.append(os.path.join(imgdir, file))
    
    temp_dict = {dirname+'data_dir': imgdir,
                 dirname+'data_image_file': image_datas,
                 dirname+'H': H,
                 dirname+'W': W,
                 dirname+'c2w': c2w,
                 dirname+'images': images,
                 dirname+'poses': poses}
        
    return temp_dict

def poses_avg(img_meta, pose_meta):
    hwf = pose_meta[0, :3, -1:]
    
    center = pose_meta[:, :3, 3].mean(0) 
    vec2 = normalize(pose_meta[:, :3, 2].sum(0)) # normalize, vec2 for Z-axis
    up = pose_meta[:, :3, 1].sum(0) # up for Y-axis
    
    # view matrix
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    
    m = np.stack([vec0, vec1, vec2, center], 1)
    
    c2w = np.concatenate([m, hwf], 1)
    
    return c2w

def normalize(x):
    return x / np.linalg.norm(x)

# ray helpers
def get_rays_us_linear(H, W, sw, sh, c2w):
    t = tr.Tensor(c2w[:3, -1]).to(device)
    R = tr.Tensor(c2w[:3, :3]).to(device)
    
    x = tr.arange(-H/2, W/2, dtype=tr.float32) * sw
    y = tr.zeros_like(x)
    z = tr.zeros_like(x)
    
    origin_base = tr.stack([x, y, z], dim=1)
    # print(origin_base.shape) # torch.Size([384, 3])
    o_base_prim = origin_base[..., None, :].to(device) # 新增倒數第2維
    # R_torch = tr.Tensor(R) #"if R's type is ndarray"
    # print(R_torch.shape, o_base_prim.shape) # torch.Size([3, 3]) torch.Size([384, 1, 3])
    
    R, o_base_prim = R.to(device), o_base_prim.to(device)
    
    o_R = tr.mul(R, o_base_prim)
    # print(o_R.shape) # torch.Size([384, 3, 3])
    
    rays_o_r = tr.sum(o_R, dim=-1)
    rays_o = rays_o_r + t
    
    dirs_base = tr.tensor([0., 1., 0.]).to(device)
    dirs_r = tr.matmul(R, dirs_base)
    rays_d = tr.broadcast_to(dirs_r, rays_o.shape)
    # print("R = ", R_torch, "\ndirs_base = ", dirs_base, "\ndirs_r = ", dirs_r)
    
    return rays_o, rays_d
