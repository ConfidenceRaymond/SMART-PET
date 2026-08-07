from turtle import forward
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn as nn
import random


def smooth_positive_labels(y):
    output = y - 0.3 + (np.random.random(y) * 0.5)

    if output >= 1:

        return 1
    else:
        return output

def smooth_negative_labels(y):
    return 0 + np.random.random(y) * 0.3


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=float(smooth_positive_labels(1)), target_fake_label=float(smooth_negative_labels(1))):
        super(GANLoss, self).__init__()
        if random.randint(0, 100) >= 7:  # noisy labels
            self.register_buffer('real_label', torch.tensor(target_real_label))
            self.register_buffer('fake_label', torch.tensor(target_fake_label))
        else:
            self.register_buffer('real_label', torch.tensor(target_fake_label))
            self.register_buffer('fake_label', torch.tensor(target_real_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # print(target_tensor),
        # print(target_tensor.shape)
        return self.loss(input, target_tensor)





class VDPenalised_l1(nn.Module):
    def __init__(self):
        super(VDPenalised_l1,self).__init__()
        self.criterion = nn.L1Loss()
        self.ep = 1e-08

    def VD(self, inputs, targets):
        VD_out = torch.mean(torch.mean(inputs)) / torch.mean(torch.mean(targets))
        return VD_out
    
    def base_exp(self, epoch, VD_ratio):
    
        decayfactor = (1000*(torch.abs(VD_ratio)))
        epoch_inv = epoch #self.watxh_epch(epoch)
        decayfactor_div = epoch_inv / decayfactor             #(epoch_inv * 0.1)*(decayfactor**(1 + (1/epoch_inv)))

        return decayfactor_div, decayfactor, epoch_inv
    
    def tot_loss(self, ll, voxel_pen):
        tot_out = (ll * voxel_pen) + ll
        return tot_out
          
        
    def __call__(self, inputs, targets, epoch):
        #inputs = torch.Tensor(inpt)
        #targets = torch.Tensor(targ)
        Voxel_ratio = self.VD(inputs, targets)
        decayfactor_div, decayfactor, epoch_inv = self.base_exp(epoch, Voxel_ratio)
        l1_loss = self.criterion(inputs, targets)

        l1_out = self.tot_loss(l1_loss, decayfactor_div)
        
        return l1_out, l1_loss, Voxel_ratio, decayfactor_div, decayfactor, epoch_inv
            
    


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)





class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.window_size = 11
        self.size_average = True
        self.ssim = SSIM3D(self.window_size, self.size_average)
        
    def forward(self, x, y):
        return 1. - (self.ssim(x,y))
    
'''  
x= torch.ones([1,1,192, 192, 192])
y= torch.zeros([1,1,192, 192, 192])

loss = SSIM3D()

loss1 = loss(x,y)

print(loss1)
'''