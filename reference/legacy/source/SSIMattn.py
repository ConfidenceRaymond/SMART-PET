#inspired by https://github.com/sergivalverde/pytorch-ssim-3D/blob/master/pytorch_ssim/__init__.py


import torch  
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
import math

#import tensorflow_probability as tfp

 

def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.
    Length of list = window_size

    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

    
def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def ssim3d(img1, img2, window_size=11, sigma = 1.5, window=True, size_average=True):
    img1, img2 = img1.float32(), img2.float32()
    L = int(torch.max([torch.max(img1), torch.max(img2)]) - torch.min([torch.min(img1), torch.min(img2)])) # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2
    
    try:
        _, channels, depth, height, width = img1.size()
    except:
        channels, depth, height, width = img1.size()

    gauss_dis = gaussian(window_size, sigma)
    window = create_window_3D(window_size, channels)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, depth, height, width) # window should be atleast 11x11 
        window = create_window_3D(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv3d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv3d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv3d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv3d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 * L) ** 2  #Note: Removed L from here (ref PT implementation)
    C2 = (0.03 * L) ** 2 


    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2
    

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        value = ssim_score.mean()  #Return mean value scalar
    else: 
        value = ssim_score #Return map
    

    return value


def self_ssim3d(img1, window_size=11, sigma=3, window=None, size_average=False):
    img1 = img1.float()
    EPSILON = 1e-10
    
    if True in torch.isnan(img1):
        L = 1
    else:
        L = torch.max(img1).item() - torch.min(img1).item() # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2
    
    try:
        _, channels, depth, height, width = img1.size()
    except:
        channels, depth, height, width = img1.size()

    gauss_dis = gaussian(window_size, sigma)
    window = create_window_3D(window_size, channels)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, depth, height, width) # window should be atleast 11x11 
        window = create_window_3D(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv3d(img1, window, padding=pad, groups=channels)
    mu1_sq = mu1 ** 2
    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv3d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    # Some constants for stability 
    C2 = (0.03 * L) ** 2 


    self_ssim = (2.0 * sigma1_sq + C2) / (2.0 * (sigma1_sq**2) + C2)

    if size_average:
        value = self_ssim.mean()  #Return mean value scalar
    else: 
        value = self_ssim         #Return map
    

    return value


 