#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch 

class Opts(object):
    def __init__(self):
        #paths
        self.dataset_path = '/gpfs/fs0/scratch/u/uanazodo/uanazodo/Ray/SMART_V1.0/Dataset/data'
        self.save_model = '/gpfs/fs0/scratch/u/uanazodo/uanazodo/Ray/Best_Experimet/run_best'
        self.save_path = '/gpfs/fs0/scratch/u/uanazodo/uanazodo/Ray/Best_Experimet/model_pl_l1/save_smart3d'
        self.rawdata_dir = '/gpfs/fs0/scratch/u/uanazodo/uanazodo/Ray/vintage'

        
        #patches
        self.patch_size = 32 #generate patches of size  (default: 32x32x32)
        self.max_length = 432 #Maximum number of patches that can be stored in the queue (default: 300)
        self.samples_per_volume = 216  #Default number of patches to extract from each volume (default: )
        self.batch_size= 1  #input batch size for training (default: 1)
        
        
        
        #self.plt_hist = [True, False]
        
        self.gpu = True #default=True, help='use gpu'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attnP='Atnnp4'
        self.config='SSAM4'
        self.attn_adv='Attn_adv3'


        ## set optimizer
        self.lr = 0.0002 #learning rate (default: 0.0001)
        self.b1 = 0.5 #betas coefficients b1  (default: 0.5)
        self.b2 = 0.999 #betas coefficients b2  (default: 0.999)
        self.lambda_adv = 5e-3 # 0.005
        self.lambda_pixel = 1e-2 # 0.02
        #self.is_training = 'Training'
        
        ## set parameters
        self.epochs = 400 #number of epochs to train (default: 200)
        self.decay_epoch = 100 #100  
        self.n_critic = 1 #number of iterations of critic(discriminator) (default: 5)
        self.checkpoint_interval = 1 #checkpoint_interval (default: 5)
        
        #test
        self.ind_epoch = 203.0
