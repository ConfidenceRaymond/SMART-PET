#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from typing import Any
import torch
import os
import numpy as np
import torchvision
import torchio as tio
from statistics import mean

from torch.autograd import Variable
import torch.nn.functional as F 
import torch.nn as nn

import warnings
import functools
import math
import nibabel as nib

from skimage.metrics import structural_similarity as ssim_y
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import mean_squared_error as mse

from matplotlib import pyplot
from torchvision.utils import save_image

from torchvision.models import vgg19
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warn = functools.partial(warnings.warn, stacklevel=2)

#check https://torchmetrics.readthedocs.io/en/v0.7.2/references/functional.html for DSC, AUC, SNR, mssim, cosine_similarity, pairwise_cosine_similarity

def check_shape_equality(sd_imgs, synth_imgs):
    """Raise an error if the shape do not match."""
    if not sd_imgs.shape == synth_imgs.shape:
        raise ValueError('Input images must have the same dimensions.')
    return

_integer_types = (       # 8 bits
                  torch.half,       # 16 bits
                  torch.bfloat16,          # 16 or 32 or 64 bits
                  torch.float,      # 32 or 64 bits
                  torch.double)  # 64 bits


_integer_ranges = {t: ((torch.finfo(t)).min, (torch.finfo(t).max))
                   for t in _integer_types}

dtype_range = {bool: (False, True),
               float: (-1, 1),
               torch.float: (-1, 1),
               torch.float16: (-1, 1),
               torch.float32: (-1, 1),
               torch.float64: (-1, 1)}
dtype_range.update(_integer_ranges)

def _as_floats(sd_imgs, synth_imgs):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    sd_imgs = sd_imgs.float().to(device)
    synth_imgs = synth_imgs.float().to(device)
    return sd_imgs, synth_imgs



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
    L = torch.max(torch.max(img1), torch.max(img2)) - torch.min(torch.min(img1), torch.min(img2)) # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2
    
    #img1 = torch.squeeze(img1, 0)
    #img2 = torch.squeeze(img2 , 0)
    #
    
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


class SSIM:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(sd_imgs, synth_imgs):
        check_shape_equality(sd_imgs, synth_imgs)
        sd_imgs, synth_imgs = _as_floats(sd_imgs, synth_imgs)
        
        return ssim3d(sd_imgs, synth_imgs)


class MSE:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "MSE"

    @staticmethod
    def __call__(sd_imgs, synth_imgs):
        check_shape_equality(sd_imgs, synth_imgs)
        sd_imgs, synth_imgs = _as_floats(sd_imgs, synth_imgs)
        
        return torch.mean((sd_imgs - synth_imgs) ** 2, dtype=torch.float64)


class NRMSE:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "NRMSE"

    @staticmethod
    def __call__(sd_imgs, synth_imgs, normalization='euclidean'):
        check_shape_equality(sd_imgs, synth_imgs)
        sd_imgs, synth_imgs = _as_floats(sd_imgs, synth_imgs)
        
        # Ensure that both 'Euclidean' and 'euclidean' match
        normalization = normalization.lower()
        if normalization == 'euclidean':
            denom = torch.sqrt(torch.mean((sd_imgs * sd_imgs), dtype=torch.float32))
        elif normalization == 'min-max':
            denom = torch.max(sd_imgs) - torch.min(sd_imgs)
        elif normalization == 'mean':
            denom = sd_imgs.mean()
        else:
            raise ValueError("Unsupported norm_type")
        
        MSE_ = MSE()
        
        return torch.sqrt(MSE_(sd_imgs, synth_imgs)) / denom


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(sd_imgs, synth_imgs, data_range=None):
        
        if data_range is None:
            if sd_imgs.dtype != sd_imgs.dtype:
                raise ValueError("Inputs have mismatched dtype.  Setting data_range based on "
					"image_true.")
            # dmin, dmax = dtype_range[sd_imgs.dtype]
            # true_min, true_max = torch.min(sd_imgs), torch.max(sd_imgs)
            # if true_max > dmax or true_min < dmin:
            #     print(
			# 		"image_true has intensity values outside the range expected "
			# 		"for its data type. Manually calculating the data_range.")
        data_range = torch.max(torch.max(sd_imgs), torch.max(synth_imgs)) - torch.min(torch.min(sd_imgs), torch.min(synth_imgs))
            # elif true_min >= 0:
            #     data_range = dmax # most common case (255 for uint8, 1 for float)
            # elif true_min < 0:
            #     data_range = dmax - dmin
			
        MSE_ = MSE()
   
        return 20 * torch.log10(data_range / torch.sqrt(MSE_(sd_imgs, synth_imgs)))
    

    
    
class RE:
    """Relative Error
    img1 and img2 have range [0, 255]
    
    img1 =sd
    img2 = synth
    """

    def __init__(self):
        self.name = "RE"

    @staticmethod
    def __call__(img1, img2):
        return torch.mean(torch.sum((img1 - img2)) / (torch.sum(img1) + EPSILON))
    
class FID_vgg19:
    def __init__(self):
        self.name = "FID_vgg19"
        

    @staticmethod
    def __call__(sd, ld):
        # Load pre-trained VGG19 model
        vgg19_ = vgg19(pretrained=True)
        vgg19_54 = nn.Sequential(*list(vgg19_.features.children())[:35]).eval()
        #self.vgg19_54  # Set to evaluation mode (no gradient calculation)
        # Calculate the minimum and maximum values in the tensor
        min_value_sd, max_value_sd = torch.min(sd), torch.max(sd)
        min_value_ld, max_value_ld = torch.min(ld), torch.max(ld)

        # Normalize the tensor to the range [-1, 1]
        sd = (2 * (sd - min_value_sd) / (max_value_sd - min_value_sd)) - 1
        ld = (2 * (ld - min_value_ld) / (max_value_ld - min_value_ld)) - 1

        # Define a common transformation for both high-resolution and low-resolution images
        common_transform = transforms.Compose([
            # transforms.Resize((224, 224)),  # Resize to VGG input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        
        sd = sd.squeeze(0) #(1,1,128,128,128)
        ld = ld.squeeze(0)
        # Create a list to store L1 losses for each slice
        l1_losses = []

        # Iterate through slices
        for slice_index in range(sd.shape[2]):
            # Slice the 3rd dimension
            slice_image_sd = sd[:, :, :, slice_index]
            slice_image_ld = ld[:, :, :, slice_index]

            # Apply the common transformations to both high-res and low-res slices
            standard_res_input = common_transform(slice_image_sd.expand(3, -1, -1)) 
            low_res_input = common_transform(slice_image_ld.expand(3, -1, -1))

            # Forward pass through VGG19 to extract features for the slices
            with torch.no_grad():
                vgg19_54 = vgg19_54.to(standard_res_input.device)
                high_res_features = vgg19_54(standard_res_input)
                low_res_features = vgg19_54(low_res_input)

            # Calculate L1 loss between high-res and low-res features for the slice
            criterion = nn.L1Loss()
            slice_l1_loss = criterion(high_res_features, low_res_features)
            l1_losses.append(slice_l1_loss.item())

        # Compute the mean L1 loss across all slices
        mean_l1_loss = np.mean(l1_losses)
        return mean_l1_loss
   
class SNR:
    def __init__(self):
        self.name = "SNR"
    
    @staticmethod    
    def __call__(sd, synth):
        print("Got here")
        # Calculate the mean signal within the region of interest (ROI)
        mean_signal_sd = torch.mean(sd)
        mean_signal_synth = torch.mean(synth)
        print("mean_signal_sd:", mean_signal_sd.item(), "mean_signal_synth:", mean_signal_synth.item())
        

        # Calculate the standard deviation of background noise
        background_noise_sd = torch.std(sd)
        background_noise_synth = torch.std(synth)
        print("background_noise_sd:", background_noise_synth.item(), "background_noise_synth:", background_noise_sd.item())

        # Calculate SNR
        snr_sd = torch.abs(mean_signal_sd / background_noise_sd)
        snr_synth = torch.abs(mean_signal_synth / background_noise_synth)
        print("snr_sd:", snr_sd.item(), "snr_synth:", snr_synth.item())
        #snr = (snr_sd/snr_synth).item()
        snr = abs(snr_sd - snr_synth)
        return snr


class CNR:
    def __init__(self):
        self.name = "CNR"

    @staticmethod
    def __call__(sd, synth):
        # Calculate the mean signal within the region of interest (ROI)
        mean_signal_sd = torch.mean(sd)
        mean_signal_synth = torch.mean(synth)
        #print("mean_signal_sd:", mean_signal_sd, "mean_signal_synth:", mean_signal_synth)

        # Calculate the standard deviation of background noise
        background_noise_sd = torch.std(sd)
        background_noise_synth = torch.std(synth)
        

        # Calculate CNR
        cnr = torch.abs((mean_signal_sd - mean_signal_synth) / background_noise_synth)
        cnr = cnr.item()
        return cnr



def ErrorMetrics(sd, synth):
    
    
    MSE_=MSE()
    NRMSE_ =NRMSE()
    SSIM_ =SSIM()
    PSNR_ =PSNR()
    FID = FID_vgg19()
    SNR_ = SNR()
    CNR_ = CNR()
    assert len(sd.shape) == len(synth.shape) == 5
    assert sd.shape[0] == synth.shape[0]
    assert sd.shape[1] == synth.shape[1]
    assert sd.shape[2] == synth.shape[2]
    assert sd.shape[3] == synth.shape[3]
    assert sd.shape[4] == synth.shape[4]
    
    #sd[sd == 0] = 0
    #synth[synth == 0] = 0
    
    errors = {}
    
    EPSILON = 1e-10
    errors['PSNR'] = "{:.4f}".format(PSNR_(sd, synth))
    errors['SSIM'] = "{:.4f}".format(SSIM_(sd, synth))
    errors['NRMSE'] = "{:.4f}".format(NRMSE_(sd, synth))
    errors['MSE'] = "{:.4f}".format(MSE_(sd, synth))
    errors['FID'] = "{:.4f}".format(FID(sd, synth))
    errors['SNR'] = "{:.4f}".format(SNR_(sd, synth))
    errors['CNR'] = "{:.4f}".format(CNR_(sd, synth))
    
    return errors


# generate samples and save as a plot and save the model
def summarize_performance(paths, opt, ld_imgs, sd_imgs, synth_imgs, ld_affines, sd_affines, fname, epochs):
    #affine=torch.squeeze(affine,0)
    #affine_nib=affine.detach().cpu().numpy()   
    #([4, 1, 128, 128, 128])

    filename_ld = os.path.join(paths, 'batch_patches_all_' + str(epochs) + '.png')
    save_to = os.path.join(paths, fname + '_epoch_' + str(epochs) + '.nii')
    
    #print(synth_imgs.shape)
    k = int(opt.patch_size // 4)
    for ii in range(synth_imgs.shape[0]):
        ld_img = ld_imgs[ii,:,:,:,:]
        sd_img = sd_imgs[ii,:,:,:,:]
        synth_img = synth_imgs[ii,:,:,:,:]
        ld_affine, sd_affine = ld_affines[ii,:,:], sd_affines[ii,:,:]
        batch_ld = tio.ScalarImage(tensor=ld_img, affine = ld_affine).data[..., 80]
        #print(batch_ld.shape)
        #batch_lds = batch_ld.data[..., 57]
        batch_sd = tio.ScalarImage(tensor=sd_img, affine = sd_affine).data[..., 80]
        batch_syn = tio.ScalarImage(tensor=synth_img, affine = ld_affine).data[..., 80]
    
        slices = torch.cat((torch.unsqueeze(batch_ld, dim = 0),torch.unsqueeze(batch_sd, dim = 0),torch.unsqueeze(batch_syn, dim = 0)), dim=0)
        torchvision.utils.save_image(slices, filename_ld, nrow=1, normalize=True, scale_each=True)
    

def means(qual_perf, ss = 'train'):
    
    if ss == 'train':
        lst_arr = np.array(qual_perf)
        mns = np.mean(lst_arr, axis=0)
        
        ql_m = mns[0], mns[1]
        
    if ss == 'val':
        psnr_, ssim_, nrmse_, mse_ = [], [], [], []
        for [psnr, ssim, nrmse, mse] in qual_perf:
            #try:
            psnr_.append(eval(psnr))
            ssim_.append(eval(ssim))
            nrmse_.append(eval(nrmse))
            mse_.append(eval(mse))
            #except NameError:
                #continue
        ql_m = round(mean(psnr_), 4), round(mean(ssim_), 4), round(mean(nrmse_), 4), round(mean(mse_), 4)
        
    return ql_m

def csvwriter(path, ind_epoch, perf_results):
	header = ['idx','PSNR', 'SSIM', 'NRMSE', 'MSE']
	errors = perf_results.insert(0, ind_epoch)
			
	with open(os.path.join(path, 'performance.csv'), 'w') as add:
		error = csv.writer(add)
		error.writerow(header)
		error.writerow(errors)



# in_ =tio.ScalarImage('/home/uanazodo/my_envs/SMART3D/Dataset/data/remi_ld_003.nii')
# out_ =tio.ScalarImage('/home/uanazodo/my_envs/SMART3D/Dataset/data/remi_sd_003.nii')

# sub = tio.Subject(out_ =tio.ScalarImage('/home/uanazodo/my_envs/SMART3D/Dataset/data/remi_sd_003.nii'))

# out_max, out_min = torch.max(out_.data).item(), torch.min(out_.data).item()
# rescale = tio.RescaleIntensity(
#     out_min_max=(-1, 1), in_min_max=(out_min, out_max))

# norm = rescale(sub)
# print('norm', norm.out_.data.min())
# norm.out_.save('/home/uanazodo/my_envs/SMART3D/norm.nii')
# print(torch.max(norm))
# print(torch.min(norm))
 

# in_ = in_.data
# out_ = out_.data
# print(out_.shape)

# in_np = in_.numpy()
# out_np = out_.numpy()



# EPSILON = 1e-10
#dr = int(np.max([out_np_ss.max(), in_np_ss.max()]) - np.min([out_np_ss.min(), in_np_ss.min()]))
#print('psnr_sk', psnr(out_np_ss, in_np_ss, data_range=dr))
#print('ssim_sk', "{:.4f}".format(ssim_y(out_np_ss, in_np_ss, win_size=11, gaussian_weights=True, sigma=1.5,data_range=dr, multichannel = True)))
#print('nrmse_sk', nrmse(out_np_ss, in_np_ss, normalization='euclidean'))
#print('mse_sk', mse(out_np_ss, in_np_ss))
#print('re_sk', np.mean(np.mean((out_np_ss - in_np_ss) / (np.maximum(out_np_ss, EPSILON)))))

# print('####################################################################################')

# RE_ =RE()
# MSE_ =MSE()
# NRMSE_ =NRMSE()
# SSIM_ =SSIM()
# PSNR_ =PSNR()

#print('PSNR', "{:.4f}".format(PSNR_(out_, in_).item()))
#print('RE', "{:.4f}".format(RE_(out_, in_).item()))
#print('MSE', "{:.4f}".format(MSE_(out_, in_).item()))
#print('NRMSE', "{:.4f}".format(NRMSE_(out_, in_).item()))
#print('SSIM', "{:.4f}".format(SSIM_(out_, in_).item()))

# print(np.max(out_np_ss))
# print(np.min(out_np_ss))

