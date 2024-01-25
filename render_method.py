import numpy as np
import torch as tr

from torch.distributions import Bernoulli
from torch.distributions.normal import Normal
import torch.nn.functional as F

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                    ):
    delta_t = 1  # 9.197324e-01
    
    x_cos = np.array(list(range(-size, size + 1)), dtype=np.float32)
    x_cos *= delta_t
    
    d1 = Normal(mean, std*2.)
    d2 = Normal(mean, std)
    
    vals_x = d1.log_prob(tr.arange(start=-size, end=size+1, dtype=tr.float32)*delta_t)
    vals_y = d2.log_prob(tr.arange(start=-size, end=size+1, dtype=tr.float32)*delta_t)
    
    gauss_kernel = tr.einsum('i,j->ij', vals_x, vals_y)
    
    return gauss_kernel / tr.sum(gauss_kernel)

g_size = 3
g_mean = 0.
g_variance = 1.
g_kernel = gaussian_kernel(g_size, g_mean, g_variance)
g_kernel = tr.as_tensor(g_kernel[None, None, :, :], dtype=tr.float32).to(device)

def render_method_convolutional_ultrasound(raw, z_vals, args):
    def raw2attenualtion(raw, dists):
        return tr.exp(-raw * dists)
    
    # Compute distance between points
    # In paper the points are sampled equidistantly
    dists = tr.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])
    dists = tr.squeeze(dists)
    dists = tr.cat([dists, dists[:, -1, None]], dim=-1)
    
    # ATTENUATION
    # Predict attenuation coefficient for each sampled point. This value is positive.
    atnua_coeff = tr.abs(raw[..., 0])
    atnua = raw2attenualtion(atnua_coeff, dists)
    
    # Compute total attenuation at each pixel location as exp{-sum[a_n*d_n]}
    atnua_trnms = tr.cumprod(atnua, dim=1) / atnua
    
    # REFLECTION
    prob_border = tr.sigmoid(raw[..., 2])
    
    # reflection coefficient for the geometry estimation
    border_dstb = Bernoulli(probs=prob_border)
    bd_samples = border_dstb.sample()
    border_indctr = bd_samples.detach()
    
    # Predict reflection coefficient. This value is between (0, 1).
    reflc_coeff = tr.sigmoid(raw[..., 1])
    reflc_trnms = 1. - reflc_coeff * border_indctr
    reflc_trnms = tr.cumprod(reflc_trnms, dim=1) / reflc_trnms 
    reflc_trnms = tr.squeeze(reflc_trnms)
    
    #padding = 3, make the output size = input size
    border_convlt = F.conv2d(input=border_indctr[None, None, :, :], weight=g_kernel, stride=1, padding=3)
    border_convlt = tr.squeeze(border_convlt)
    
    # BACKSCATTERING
    # Scattering density coefficient can be either learned or constant for fully developed speckle
    density_coeff_value = tr.sigmoid(raw[..., 3])
    density_coeff = tr.ones_like(reflc_coeff) * density_coeff_value
    
    sctrer_dens_dstb = Bernoulli(probs=density_coeff)
    sctrer_dens = sctrer_dens_dstb.sample()
    
    # Predict scattering amplitude
    ampltd = tr.sigmoid(raw[..., 4])
    
    # Compute scattering template
    sctrer_map = tr.mul(sctrer_dens, ampltd)
    
    print(sctrer_map.shape)
    
    psf_sctr = F.conv2d(input=sctrer_map[None, None, :, :], weight=g_kernel, stride=1, padding=3)
    psf_sctr = tr.squeeze(psf_sctr)
    
    print(psf_sctr.shape)
    
    # Compute remaining intensity at a point n
    transmission = tr.mul(atnua_trnms, reflc_trnms)
    # Compute backscattering part of the final echo
    b = tr.mul(transmission, psf_sctr)
    # Compute reflection part of the final echo
    r = tr.mul(tr.mul(transmission, reflc_coeff), border_convlt)
    
    # Compute the final echo
    intensity_map = b + r
    
    ret = {'intensity_map': intensity_map,
           'attenuation_coeff': atnua_coeff,
           # 'reflection_coeff': reflc_coeff,
           'attenuation_transmission': atnua_trnms,
           # 'reflection_transmission': reflc_trnms,
           'scatterers_density': sctrer_dens,
           'scatterers_density_coeff': density_coeff,
           'scatter_amplitude': ampltd,
           'b': b,
           'r': r,
           "transmission": transmission}
    
    return ret