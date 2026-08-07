import os

import multiprocessing

from os.path import join
from pathlib import Path

import torch
import torchio as tio
import nibabel as nib
import torchvision
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import numpy as np
#from IPython import display


num_workers = multiprocessing.cpu_count()


import random
seed = 1234


#from Dataset.Hist_Norm import *

class SMARTload:
    def __init__(self, dataset_path, patch_size, max_length, samples_per_volume,  \
                 batch_size, save_pth, rawdata_dir, split = 'train', train_type = 'train_patches'):
        super(SMARTload,self).__init__()
        self.dataset_root = dataset_path
        self.split = split
        self.train_type = train_type
        self.patch_size = patch_size
        self.max_length = max_length
        self.samples_per_volume = samples_per_volume
        #self.num_workers = self.num_workers
        self.batch_size = batch_size
        self.save_pth = save_pth
        self.rawdata = rawdata_dir  
        self.Augument = False
       
        assert self.split in ('train', 'val', 'test')
        assert self.train_type in ('train_whole_images', 'train_patches')
        


    def load(self):

        if self.split == 'test':
            sets = self.smart_loader(self.dataset_root)
            loader = DataLoader(sets, self.batch_size, shuffle=False)
        else:
            sets = self.smart_loader(self.dataset_root)
            loader = DataLoader(sets, self.batch_size, shuffle=True)

        return loader   
    
    
    
    def test(self, patient):
        assert self.split == 'test'
                
        loader_tst = DataLoader(patient, self.batch_size, shuffle=False)

        return loader_tst


    def smart_loader(self, dir):
                    
        if self.split == 'train':
            ld_imgs, sd_imgs, ld_imgs_id, sd_imgs_id = [], [], [], []
            train = self.splits()
            for (sds, lds, sds_id, lds_id) in train:
                sd_imgs.append(sds)
                ld_imgs.append(lds)
                sd_imgs_id.append(sds_id)
                ld_imgs_id.append(lds_id)
                
                    
            
        elif self.split == 'val':
            ld_imgs, sd_imgs, ld_imgs_id, sd_imgs_id = [], [], [], []
            val = self.splits()
            for (sds, lds, sds_id, lds_id) in val:
                sd_imgs.append(sds)
                ld_imgs.append(lds)
                sd_imgs_id.append(sds_id)
                ld_imgs_id.append(lds_id)
                
        else:
            ld_imgs, sd_imgs, ld_imgs_id, sd_imgs_id = [], [], [], []
            test = self.splits()
            for (sds, lds, sds_id, lds_id) in test:
                sd_imgs.append(sds)
                ld_imgs.append(lds)
                sd_imgs_id.append(sds_id)
                ld_imgs_id.append(lds_id)
                
                
        
        if self.split in ('train','val'):
            subjects = []
            for (ld_file, sd_file, ld_imgs_id, sd_imgs_id) in zip(ld_imgs, sd_imgs, ld_imgs_id, sd_imgs_id):
                
                subject = tio.Subject(
                    ld_subject=tio.ScalarImage(ld_file),
                    sd_subject=tio.ScalarImage(sd_file),
                )
                subjects.append(subject)
                
            #Norm_hist = Histogram( self.dataset_root, self.save_pth)
            #landmarks = Norm_hist.get_landmarks()
                
            if self.split == 'train' and self.Augument:
                _transform= tio.Compose([
                #tio.HistogramStandardization(landmarks),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                tio.RandomAffine(scales=(0.9, 1.2), degrees=15, translation=5, isotropic=False, center='image', image_interpolation='linear'),
                tio.RandomGamma(log_gamma = (-1,1), include = ['sd_subject']),
                tio.RandomFlip(axes=(0, 1, 2)),
                tio.RescaleIntensity((-1, 1)),
                ])
                
                _datasets_all = tio.SubjectsDataset(subjects, transform=_transform)
                print('Train dataset size with Aug:', len(_datasets_all), 'subjects')
            
            else:
                _transformVal= tio.Compose([
                #tio.HistogramStandardization(landmarks),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                tio.RescaleIntensity((-1, 1))
                
                ])
                _datasets_all = tio.SubjectsDataset(subjects, transform=_transformVal)
                
                if self.split == 'train':
                    print('Train dataset size w/o Aug:', len(_datasets_all), 'subjects')
                else:
                    print('Validation dataset size.:', len(_datasets_all), 'subjects')
                
                
        if self.split == 'test':
            test_subjects = []
            for (ld_file, sd_file, ld_imgs_id, sd_imgs_id) in zip(ld_imgs, sd_imgs, ld_imgs_id, sd_imgs_id):
                subject_tst = tio.Subject(
                    sd_subject = tio.ScalarImage(sd_file),
                    ld_subject = tio.ScalarImage(ld_file),
                    fname = sd_imgs_id,
                )
                test_subjects.append(subject_tst)
                
            _transformTest= tio.Compose([
                #tio.HistogramStandardization(landmarks),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                tio.RescaleIntensity((-1, 1))
                
                ])
                
            _datasets_all = tio.SubjectsDataset(test_subjects, transform=_transformTest)
            print('Test dataset size:', len(_datasets_all), 'subjects')
                

            
        #print(subject_name)
            
        #one_subject = _datasets[13]
        #one_subject.plot()
            
        #print(one_subject)
        #print(one_subject.ld_subject)
        #print(one_subject.sd_subject)

        if self.train_type == 'train_whole_images' and self.split in ('train','val','test'):
            in_datasets = _datasets_all

        elif self.train_type == 'train_patches' and self.split in ('train','val'):
            in_datasets = self.gen_patches(_datasets_all)
            
        return in_datasets
    
    
    
    def gen_patches(self, dataset):
        sampler = tio.data.UniformSampler(self.patch_size)

        if self.split == 'train':
            print('split == train')
            _patches = tio.Queue(
                subjects_dataset = dataset,
                max_length = self.max_length,
                samples_per_volume = self.samples_per_volume,
                sampler = sampler,
                num_workers = num_workers,
                shuffle_subjects = True,
                shuffle_patches = True,
            )

        elif self.split == 'val':
            print('split == val')
            _patches = tio.Queue(
                subjects_dataset = dataset,
                max_length = self.max_length,
                samples_per_volume = self.samples_per_volume,
                sampler = sampler,
                num_workers = num_workers,
                shuffle_subjects = True,
                shuffle_patches = True,
            )
            
        else:
            print('split == test')
            _patches = tio.Queue(
                subjects_dataset = dataset,
                max_length = self.max_length,
                samples_per_volume = self.samples_per_volume,
                sampler = sampler,
                num_workers = num_workers,
                shuffle_subjects = True,
                shuffle_patches = True,
            )
            
        return _patches
    
    
    def splits(self):
        lst = []
        with open(os.path.join(self.dataset_root, self.split + '_dirs'), "r") as dirfile:
            for line in dirfile:
                for dd in line.strip().split(';'):
                    lst.append(eval(dd))
        return lst    

    def getMaxandMin(self, sd_in, id):
        dt = tio.ScalarImage(sd_in).data
        aff = tio.ScalarImage(sd_in).affine
        
        max_dt = torch.max(dt)
        min_dt = torch.min(dt)
        
        return dt, aff, max_dt, min_dt, id
    
    def Inv_Rescale(self, predict_in, fname):
        sd_dir = os.path.join(self.rawdata, str(fname[0])+'.nii')
        Orig_sd = tio.ScalarImage(sd_dir)
        max_px = torch.max(Orig_sd.data).item()
        min_px = torch.min(Orig_sd.data).item()
        
        synth_in= predict_in.squeeze(0).cpu()
        
        synth_max, synth_min = torch.max(synth_in).item(), torch.min(synth_in).item()
        inv_rescale_img = tio.RescaleIntensity(
                out_min_max=(min_px, max_px), in_min_max=(synth_min,synth_max))
        
        return inv_rescale_img(synth_in)

    
    def Inv_Rescale_Res_save(self, predict_in, fname, save_dir):
        sd_dir = os.path.join(self.rawdata, str(fname)+'.nii.gz')
        #ld_Img = tio.ScalarImage(os.path.join(self.rawdata, str(fname[0].replace('sd','ld'))+'.nii'))
     
        Orig_sd = tio.ScalarImage(sd_dir)
        max_px = torch.max(Orig_sd.data).item()
        min_px = torch.min(Orig_sd.data).item()
        mean_px = torch.mean(Orig_sd.data).item()
        std_px = torch.max(Orig_sd.data).item()
        #print("Orig_sd:", 'mean_px', mean_px, 'std_px', std_px)
        print("Orig_sd:", 'max_px', max_px, 'min_px', min_px)
        
        synth_in = predict_in.squeeze(0).cpu()
        synth_in = (synth_in + 1) / 2

        #print("synth_in:", torch.max(synth_in).item(), torch.min(synth_in).item())

        normalized_pixel_value = synth_in * max_px
        print("normalized_pixel_value:", torch.max(normalized_pixel_value).item(), torch.min(normalized_pixel_value).item())
        
        # synth_max, synth_min = torch.max(synth_in).item(), torch.min(synth_in).item()
        # inv_rescale_img = tio.RescaleIntensity(
        #         out_min_max=(min_px, max_px), in_min_max=(synth_min,synth_max))

        # inv_Re = inv_rescale_img(synth_in)
        # #inv_Norm = self.Inv_Norm(inv_Re)

        self.save2tio(normalized_pixel_value, fname, save_dir)
        
        # normalized_pixel_value = (synth_in - mean_px) / std_px
        # print("normalized_pixel_value:", torch.max(normalized_pixel_value).item(), torch.min(normalized_pixel_value).item())

        
        to_standard = tio.Resample(sd_dir)

        subject = tio.Subject(
             synth_img = tio.ScalarImage(save_dir.replace('_pred','_pred_s2t'))
         )

        Res_ = to_standard(subject.synth_img)
        print("Res_:", torch.max(Res_.data).item(), torch.min(Res_.data).item())
        
        
        # # Calculate scaling factors
        # scale_factor_mean = mean_px /torch.mean(Res_.data).item() 
        # scale_factor_std = std_px / torch.std(Res_.data).item() 
        
        
        # print("Res_:", 'mean_px', torch.mean(Res_.data).item(), 'std_px', torch.std(Res_.data).item())
        # print("Res_:", 'max_px', torch.max(Res_.data).item(), 'min_px', torch.min(Res_.data).item())
        
        # # Normalize the output image using the scaling factors
        # normalized_output = Res_.data * scale_factor_std # * scale_factor_std
        # print('scale_factor_mean:', scale_factor_mean, "scale_factor_std:", scale_factor_std)
        
        # print("normalized_output:", 'mean_px', torch.mean(normalized_output).item(), 'std_px', torch.std(normalized_output).item())
        # print("normalized_output:", 'max_px', torch.max(normalized_output).item(), 'min_px', torch.min(normalized_output).item())

        self.save2tio(Res_.data, fname, save_dir)

        # print('---- saved '+str(fname).replace('_sd','_pred')+' to file ----')

        #--------------------------------------------------------------------------------------------------------------------------
        
        #inv_ = inv_rescale_img(xxx)

        #synth_img = tio.ScalarImage(save_dir.replace('_pred','_pred_s2t'))
        #self.save2tio(inv_.numpy(), fname, save_dir)

        #self.save2tio(xxx, fname, save_dir)
        
        #return 
    
    def save2nifty(self, predicts, fname, fname_out): 
        Orig_sd = tio.ScalarImage(os.path.join(self.rawdata, str(fname)+'.nii.gz'))
        print("Orig_sd", Orig_sd)
        ref_a = Orig_sd.affine
        
        data = predicts.squeeze().cpu().detach().numpy()
        final_img = nib.Nifti1Image(data, ref_a)
        nib.save(final_img, fname_out.replace('_pred','_pred_s2n'))
        
    def save2tio(self, predicts, fname, fname_out):
        Orig_sd = tio.ScalarImage(os.path.join(self.rawdata, str(fname)+'.nii.gz'))
        tio.ScalarImage(tensor=predicts, affine=Orig_sd.affine).save(fname_out.replace('_pred','_pred_s2t'))
        
    def ResampleImg(self, predicts, fname, save_dir):
        sd_Img = tio.ScalarImage(os.path.join(self.rawdata, str(fname[0])+'.nii'))
        ld_Img = tio.ScalarImage(os.path.join(self.rawdata, str(fname[0].replace('sd','ld'))+'.nii'))
    
        synth = tio.ScalarImage(tensor=predicts, affine=ld_Img.affine)  #ld_Img.affine
        
        to_standard = tio.Resample(sd_Img)
        return to_standard(synth)
    
    def Inv_Norm(self, predicts):
        predicts: torch.Tensor
        values = predicts.clone().float()
        mean, std = values.mean(), values.std()
        if std == 0:
            return None
        predicts *= std
        predicts += mean
        
        return predicts

        
def prepare_load(loader):
    sd = loader['sd_subject'][tio.DATA]
    ld = loader['ld_subject'][tio.DATA]
    sd_affine = loader['sd_subject'][tio.AFFINE]
    ld_affine = loader['ld_subject'][tio.AFFINE]
    
    return sd, ld, sd_affine, ld_affine
    
def Rescale(sd_in, ld_in, n = 'double', type = 'Normalize'):
    sd_in= torch.squeeze(sd_in, 0).cpu()
    ld_in= torch.squeeze(ld_in, 0).cpu()
    sd_out_max, sd_out_min = torch.max(sd_in).item(), torch.min(sd_in).item()
    ld_out_max, ld_out_min = torch.max(ld_in).item(), torch.min(ld_in).item()
    
    if n == 'double':
        if type == 'Normalize':
            rescale_sd = tio.RescaleIntensity(
                out_min_max=(-1, 1), in_min_max=(sd_out_min, sd_out_max))
            rescale_ld = tio.RescaleIntensity(
                out_min_max=(-1, 1), in_min_max=(ld_out_min, ld_out_max))
            norm_sd, norm_ld = rescale_sd(sd_in), rescale_ld(ld_in)
            norm_sd, norm_ld  = torch.unsqueeze(norm_sd, 0), torch.unsqueeze(norm_ld, 0)
            
        elif type == 'Denormalize':
            rescale_sd = tio.RescaleIntensity(
                out_min_max=(sd_out_min, sd_out_max), in_min_max=(-1,1))
            rescale_ld = tio.RescaleIntensity(
                out_min_max=(ld_out_min, ld_out_max), in_min_max=(-1,1))
            norm_sd, norm_ld = rescale_sd(sd_in), rescale_ld(ld_in)
            norm_sd, norm_ld = torch.unsqueeze(norm_sd, 0), torch.unsqueeze(norm_ld, 0)
        
        norm = norm_sd, norm_ld
    
    elif n == 'single':
        if type == 'Normalize':
            rescale_sd = tio.RescaleIntensity(
                out_min_max=(-1, 1), in_min_max=(sd_out_min, sd_out_max))
            norm_sd = rescale_sd(sd_in)
            norm_sd = torch.unsqueeze(norm_sd,0)
            
        elif type == 'Denormalize':
            rescale_sd = tio.RescaleIntensity(
                out_min_max=(sd_out_min, sd_out_max), in_min_max=(-1,1))
            norm_sd = rescale_sd(sd_in)
            norm_sd = torch.unsqueeze(norm_sd,0)
        
        norm = norm_sd
        
    return norm      

    
# dir = '/home/uanazodo/my_envs/SMART3D/Dataset/data' 

# we = SMARTload(self, split = 'train', train_type = 'train_patches')
# we.load()

# v1 = smart_loader(dir, split = 'train', train_type = 'train_patches')

# v = DataLoader(v1, batch_size=3)

# print('Dataset size:', len(v), 'subjects')

# k = int(self.patch_size//4)
# one_batch = next(iter(v))
# batch_sd = one_batch['sd_subject'][tio.DATA][..., k]
# batch_ld = one_batch['ld_subject'][tio.DATA][..., k]
# slices = torch.cat((batch_sd, batch_ld))
# image_path_all = 'batch_patches_all.png'
# torchvision.utils.save_image(slices, image_path_all, nrow=3, normalize=True, scale_each=True)
# display.Image(image_path_all)

# dataset_train   = SMARTload(self, split = 'train', train_type = 'train_patches')
# dataloader_train = dataset_train.combinelist()

#loader_tst, patch_aggregator = dataloader_train 
#print('lenght:', len(loader_tst))




# with torch.no_grad():
#     perf_results = []
#     #for i in len(loader)
#     for ii, loader in enumerate(tqdm(loader_tst)):
#         #print(loader)
#         _sdimgs = loader['sd_subject'][tio.DATA]
#         _ldimgs = loader['ld_subject'][tio.DATA]
#         sd_affine = loader['sd_subject'][tio.AFFINE]
#         ld_affine = loader['ld_subject'][tio.AFFINE]
                
#         patch_locations = loader[tio.LOCATION]
                
#         idx = loader['fname']
#         patient_ID = idx                #.replace('_sd', '')
#         print('First:',patient_ID)
# print('Second:',patient_ID)
#print(_sdimgs.shape)
#print(_ldimgs.shape)


#loader_tst, patch_aggregator, patient_ID, sd_data, ld_data, sd_affine = dataloader_train
#print(patient_ID)
# def get_dim(x_y_z):
#     N = x_y_z.shape[0]
#     C = x_y_z.shape[1]
#     D = x_y_z.shape[2]
#     H = x_y_z.shape[3]
#     W = x_y_z.shape[4]
        
#     return N, C, D, H, W


# for index, imgs in enumerate(tqdm(dataloader_train)):
#     print(index)

#     sd_imgs, ld_imgs, _, _ = prepare_load(imgs)
#     print('ld_imgs',ld_imgs.shape)
    
#     N, C, D, H, W = get_dim(sd_imgs)
#     print(N, C, D, H, W)
    
