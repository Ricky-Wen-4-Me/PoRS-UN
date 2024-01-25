import os
import configargparse
import numpy as np
import time

from tensorboardX import SummaryWriter
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM

import torch as tr
from torch.autograd import Variable

import load_US as US
import render_method as rm
import run_nerf_helper as rnh
import positional_encoding as PE

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

probe_depth = 100
probe_width = 37

def config_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument('--config', is_config_file=True,
                        help='config file path', default='config_test.txt')
    parser.add_argument("--expname", type=str, help='experiment name')
    
    # data preprocess
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument('--datadir', type=str, default='./data/Ultra-NeRF/generated_liver_0501/train_views/',
                          help='input train data directory')
    '''validation and test data'''
    parser.add_argument('--valdatadir', type=str, default='./data/Ultra-NeRF/generated_liver_0501/test_views/',
                          help='input val data directory')
    parser.add_argument('--testdatadir', type=str, default='./data/Ultra-NeRF/generated_liver_0501/test_views/',
                          help='input test data directory')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--dataset_type", type=str, default='us', help='options: us')
    
    # training options
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument("--loss", type=str, default='l2')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128,
                        help='channels per layer')
    parser.add_argument("--raychunk", type=int, default=4096 * 16,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=2048 * 16,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument('--probe_depth', type=int, default=140)
    parser.add_argument('--probe_width', type=int, default=80)
    parser.add_argument("--output_ch", type=int, default=5)
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--lrate", type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument("--ssim_filter_size", type=int, default=7)
    parser.add_argument("--ssim_lambda", type=float, default=0.75, 
                        help='SSIM coefficient of loss function defined at the Ultra-NeRF paper')
    
    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')# 要不要PE
    parser.add_argument("--i_embed_gauss", type=int, default=0,
                        help='mapping size for Gaussian positional encoding, 0 for none')# 要不要用高斯PE
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')# 座標的PE維度
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')# 方向的PE維度
    
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=50,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=100,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5000000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=5000000,
                        help='frequency of render_poses video saving')
    
    return parser

def run_network(inputs, fn, embed_fn, netchunk=512*32):
    def batchify(fn, chunk):
        if chunk is None:
            return fn
        
        def ret(inputs):
            return tr.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        
        return ret
    
    inputs_flat = tr.reshape(inputs, [-1, inputs.shape[-1]])
    
    embedded = embed_fn(inputs_flat)
    outputs_flat = batchify(fn, netchunk)(embedded)
    
    print("inputs.shape[:-1]: ", list(inputs.shape[:-1]), "\n",
          "outputs_flat.shape[-1]: ", [outputs_flat.shape[-1]], "\n")
    outputs = tr.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    
    return outputs

def render_rays_us(ray_batch, network_fn, network_query_fn, 
                   N_samples, retraw=False, lindisp=False, 
                   args=None):
    # Volumetric rendering
    def raw2outputs(raw, z_vals, rays_d):
        ret = rm.render_method_convolutional_ultrasound(raw, z_vals, args)
        
        return ret
    
    # batch size
    N_rays = ray_batch.shape[0]
    
    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    
    # Extract lower, upper bound for ray distance.
    bounds = tr.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]
    
    # Decide where to sample along each ray. Under the logic, all rays will be sampled at the same times.
    t_vals = tr.linspace(0., 1., N_samples).to(device)
    
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same integration points will be used for all rays.
        z_vals = near * (1. - t_vals) + far * (t_vals).to(device)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals)).to(device)
    
    z_vals = tr.broadcast_to(z_vals, [N_rays, N_samples])
    
    # Points in space to evaluate model at.
    origin = rays_o[..., None, :]
    step = rays_d[..., None, :] * z_vals[..., :, None]
    
    pts = step + origin
    
    # Evaluate model at each point.
    # network_query_fn doesn't def yet.
    raw = network_query_fn(pts, network_fn)
    
    ret = raw2outputs(raw, z_vals, rays_d)
    
    if retraw:
        ret['raw'] = raw
    
    for k in ret:
        tr.testing.assert_allclose(ret[k], 'output {}'.format(k))
        
    return ret
        
def make_ray_batches(rays, c2w=None, chunk=32 * 256, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    # print(rays.shape[0], chunk)

    for i in range(0, rays.shape[0], chunk):
        ret = render_rays_us(rays[i:i+chunk], **kwargs)
        
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            
            all_ret[k].append(ret[k])
    
    all_ret = {k: tr.cat(all_ret[k], 0) for k in all_ret}
    
    return all_ret
        

def render_us(H, W, sw, sh, chunk=1024 * 32, rays=None, c2w=None, near=0., far=55. * 0.001, **kwargs):
    # Render rays
    rays_o, rays_d = [], []
    
    if c2w is not None:
        rays_o, rays_d = US.get_rays_us_linear(H, W, sw, sh, c2w)
    
    sh = rays_d.shape
    
    rays_o = rays_o.view(-1, 3).to(dtype=tr.float32)
    rays_d = rays_d.view(-1, 3).to(dtype=tr.float32)
    
    near, far = near * tr.ones_like(rays_d[..., :1]), \
        far * tr.ones_like(rays_d[..., :1])
    # print("o = ", rays_o, "\nd = ", rays_d)
    # print("near = ", near.shape, "\nfar = ", far.shape)
    
    rays = tr.cat([rays_o, rays_d, near, far], dim=-1)
    
    # print(rays, rays.shape)
    
    all_ret = make_ray_batches(rays, c2w=c2w, chunk=chunk, **kwargs)
    
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        
        all_ret[k] = tr.view(all_ret[k], k_sh)
    print(all_ret)
    
    return all_ret

def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = PE.get_embedder(args.multires, args.i_embed, args.i_embed_gauss)
    
    input_ch_views = 0
    embeddirs_fn = None
    
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = PE.get_embedder(args.multires_views, args.i_embed)
    
    output_ch = 3
    skips = [4]
    
    model = rnh.init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips)
    
    grad_vars = list(model.parameters())
    
    paras = model.parameters()
    
    mds = {'model':model}
    
    def network_query_fn(inputs, network_fn):
        return run_network(
            inputs, network_fn, 
            embed_fn=embed_fn,
            netchunk=args.netchunk)
    
    # kwargs = keyword argument
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'N_samples': args.N_samples,
        'network_fn': model
    }
    
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    
    start = 0
    basedir = args.basedir
    expname = 'mokey'
    
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
        
    print('Found checkpoints!!!', ckpts)
    
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        
        print('reload from', ft_weights)
        
        model.weight.data = np.load(ft_weights, allow_pickle=True)
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to ', start)
        
    return render_kwargs_train, render_kwargs_test, start, grad_vars, mds, paras

def _train_yet():
    parser = config_parser()
    args = parser.parse_args()
    
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tr.cuda.manual_seed_all(args.random_seed)
    
    traindir = args.datadir
    valdir, testdir = args.valdatadir, args.testdatadir
    # print(dir, valdir, testdir)
    
    datadir_array = [traindir, valdir, testdir]
    dirname_array = ["train_", "val_", "test_"]
    
    data_dict = {}
    
    for i in range(len(datadir_array)):
        t_dict = US.load_image_by_path(datadir_array[i], dirname_array[i])
        
        data_dict.update(t_dict)
    
    # if args.dataset_type == 'us':
    #     datadir, skip = US.load_ultra_data(dir, args.testskip) 
    # print(datadir)
    
    # print(data_dict.keys())
    
    images = tr.from_numpy(data_dict['train_images'])
    poses = tr.from_numpy(data_dict['train_poses'])
    images = images.to(device)
    poses = poses.to(device)
    
    scaling = 0.001
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    
    near = 0.
    far = probe_depth
    
    H, W = data_dict['train_H'], data_dict['train_W']
    
    sy = probe_depth / float(H)
    sx = probe_width / float(W)
    sh = sy
    sw = sx
    
    basedir = args.basedir
    
    pose = poses[:, :3, :3]
    # print("poses = ", poses[0], "\n")
    # print("pose = ", pose[0], "\n")
    # print(pose.shape) # torch.Size([800, 3, 3])
    
    basedir = args.basedir
    expname = 'mokey'
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
        
        if args.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            
            with open(f, 'w') as file:
                cfg = open(args.config, 'r').read()
                
                file.write(cfg)
    
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models_dict, parameters = create_nerf(args)
    
    rays_o, rays_d = US.get_rays_us_linear(H, W, sw, sh, data_dict['train_c2w'])
    batch_rays = tr.stack([rays_o, rays_d], 0)
    
    bds_dict = {
        'near': near,
        'far': far,
    }
    
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    
    render_kwargs_train["args"] = args
    # print(render_kwargs_train)
    
    #Create optimizer
    
    lrate = args.lrate
    
    grad_vars = [Variable(var, requires_grad=True) for var in parameters]
    optimizer = tr.optim.Adam(grad_vars, lrate)
    models_dict['optimizer'] = optimizer
    
    print(optimizer)
    
    global_step = tr.tensor(start, dtype=tr.long, requires_grad=False)
    
    N_iters = args.n_iters
    
    i_train, i_test, i_val = data_dict['train_images'].shape[0], data_dict['test_images'].shape[0], data_dict['val_images'].shape[0]
    
    print('Begin:\n')
    print('TRAIN views are ', i_train)
    print('TEST views are ', i_test)
    print('VAL views are ', i_val)
    
    log_dir = os.path.join(basedir, 'summaries', expname)
    writer = SummaryWriter(log_dir)
    
    for i in range(start, N_iters):
        time0 = time.time()
        # Sample random ray batch
        # Random from one image
        img_i = np.random.choice(i_train)
        
        try:
            target = tr.transpose(images[img_i])
        except:
            print(img_i)
            
        pose = poses[img_i, :3, :4]
        
        ssim_weight = args.ssim_lambda
        l2_weight = 1. - ssim_weight
        
        rays_o, rays_d = US.get_rays_us_linear(H, W, sw, sh, pose)
        
        batch_rays = tr.stack([rays_o, rays_d], 0)
        
        loss = {}
    
        with tr.autograd.set_detect_anomaly(True):
            rendering_output = render_us(
                    H, W, sw, sh, c2w=pose, chunk=args.raychunk, rays=batch_rays, **render_kwargs_train)
            
            output_image = rendering_output['intensity_map']
            
            if args.loss == 'l2':
                l2_intensity_loss = rnh.img2mse(output_image, target)
                
                loss["l2"] = (1., l2_intensity_loss)
            elif args.loss == 'ssim':
                ms_ssim = MS_SSIM(kernel_size=args.ssim_filter_size, data_range=1.0, k1=0.01, k2=0.1)
                ssim_intensity_loss = 1. - ms_ssim(output_image, target)
                loss["ssim"] = (ssim_weight, ssim_intensity_loss)
                
                l2_intensity_loss = rnh.img2mse(output_image, target)
                loss["l2"] = (l2_weight, l2_intensity_loss)
            
            total_loss = sum(w * l for w, l in loss.values())
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        dt = time.time() - time0
        #####           end            #####
        
        # Rest is logging
        def save_weights(net, prefix, i):
            path = os.path.join(basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            
            np.save(path, net.save_weights())
            print('saved weights at : ', path)
        
        if i % args.i_weights == 0:
            for k in models_dict:
                save_weights(models_dict[k], k, i)
                
        if i % args.i_print == 0 or i < 10:
            print(expname, i, total_loss.numpy(), global_step.numpy())
            print('iter time {".05f"}'.format(dt))
            
            g_i = 0
            for g_i, t in enumerate(grad_vars):
                writer.add_histogram(f'{g_i}', t, global_step=g_i)
            
            writer.add_scalar('misc/learning_rate', lrate, global_step=optimizer.state_dict()['state'][0]['step'])
            
            loss_string = "total loss = "
            for l_key, l_value in loss.items():
            
            writer.close()
            
if __name__=='__main__':
    
    _train_yet()