 #!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import torch
import torchio as tio
import time, datetime
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torchvision.utils import save_image
#


import model16_8 as models
from utils_smart import misc, metrics, weights, losses_, model_save
torch.autograd.set_detect_anomaly(True)

from param import Opts
opt = Opts()
#from Augs.diff_augs import diff_transform as transform
from Dataset.dataset import *

class Net():

    def __init__(self):
        
        self.generator  = models.generator(config=opt.config) #config=opt.config
        self.adversary =  models.adversary()
        #print(opt.config)

        
        self.generator = self.generator.to(opt.device)
        self.adversary = self.adversary.to(opt.device)

        if torch.cuda.device_count() > 1:
            self.generator = torch.nn.DataParallel(self.generator)
            self.adversary = torch.nn.DataParallel(self.adversary)
            
        self.criterion = metrics.FID_vgg19()
        
        # self.generator = self.generator.apply(weights.init_model)
        # self.adversary = self.adversary.apply(weights.init_model)
        
        
    def loss(self, synth_img, sd_imgs, _ldimgs, adversary):
            # Losses generator
        adversarial_loss   = losses_.GANLoss().to(opt.device) #nn.MSELoss().to(opt.device)  #nn.BCEWithLogitsLoss() #nn.MSELoss().to(opt.device) #try nn.BCELoss() 
        criterion = nn.L1Loss().to(opt.device) 
        
        synth_loss = criterion(synth_img, sd_imgs)
        
        adv_real = torch.cat([_ldimgs, sd_imgs], dim=1)
                
        loss_GAN = adversarial_loss(adversary(adv_real), True)
        totalGAN_loss = loss_GAN + (100 * synth_loss)

        # adversary loss 
        adv_fake = torch.cat([_ldimgs, synth_img], dim=1).detach()
        realAdv_loss = adversarial_loss(adversary(adv_real), True)
        fakeAdv_loss = adversarial_loss(adversary(adv_fake), False) 
        totalAdv_loss = (realAdv_loss + fakeAdv_loss) / 2
                
        return totalGAN_loss, totalAdv_loss

    def train(self):
        # if not os.path.isdir(opt.save_path+'/'+str('train')+'/'):
        #     misc.mkNew_dir(opt.save_path+'/'+str('train')+'/')
            
        # if not os.path.isdir(opt.save_path+'/'+str('val')+'/'):
        #     misc.mkNew_dir(opt.save_path+'/'+str('val')+'/')

        logger_train_mean = misc.Logger(os.path.join(opt.save_path+'/'+str('train')+'/'+'train_log_mean_model_pl_l1.txt'), title='')
        logger_train_perf = misc.Logger(os.path.join(opt.save_path+'/'+str('train')+'/'+'train_log_perf_model_pl_l1.txt'), title='')
        #logger_exp = misc.Logger(os.path.join(opt.save_path+'/'+str('train')+'/'+'loss_experiment.txt'), title='')
        
        logger_val_perf = misc.Logger(os.path.join(opt.save_path+'/'+str('val')+'/'+'val_log_perf_model_pl_l1.txt'), title='')

        logger_train_mean.set_names(['Run epoch', 'D Loss', 'G Loss'])
        logger_train_perf.set_names(['Run epoch', 'PSNR', 'SSIM', 'NRMSE', 'MSE'])

        logger_val_perf.set_names(['Run epoch', 'PSNR', 'SSIM', 'NRMSE', 'MSE'])

        #logger_exp.set_names(['synth_loss', 'l1_loss', 'Voxel_ratio', 'decayfactor_div_', 'decayfactor_', 'epoch_inv_'])
        
        
        self.generator = self.generator.apply(weights.weights_init_normal)
        self.adversary = self.adversary.apply(weights.weights_init_normal)
        

         #Optimizers
        A_optimizer = torch.optim.Adam(self.adversary.parameters(), lr=opt.lr,betas=(opt.b1, opt.b2))
        G_optimizer = torch.optim.Adam(self.generator.parameters(),lr=opt.lr,betas=(opt.b1, opt.b2))

         # Learning rate update schedulers
        lr_scheduler_G  = torch.optim.lr_scheduler.LambdaLR(G_optimizer, lr_lambda=misc.LambdaLR(opt.epochs, 0, opt.decay_epoch).step) #misc.LambdaLR
        lr_scheduler_D  = torch.optim.lr_scheduler.LambdaLR(A_optimizer, lr_lambda=misc.LambdaLR(opt.epochs, 0, opt.decay_epoch).step)


        # Load data        
        dataset_train   = SMARTload(opt.dataset_path, opt.patch_size, opt.max_length, opt.samples_per_volume, \
            opt.batch_size, opt.save_path, opt.rawdata_dir, split = 'train', train_type = 'train_whole_images')
        dataloader_train = dataset_train.load()

        batches_done = 0
        prev_time    = time.time()
        # ---------------------------- *training * ---------------------------------   
        for epoch in range(1, (opt.epochs+1)):
            
            train_perf_m, train_perf_s = [], []
            losses_m, losses_s = [], []
            val_perf_s = []
            
            self.generator.train()
            self.adversary.train()
            
            for index, imgs in enumerate(dataloader_train):
                   
                sdIMGS, ldIMGS, _, _ = prepare_load(imgs)
                
                if opt.gpu:
                    _sdimgs, _ldimgs = sdIMGS.to(opt.device ,dtype=torch.float32), ldIMGS.to(opt.device, dtype=torch.float32)

                #for _ in range(opt.n_critic):
                    # generator forward pass
                synth_img = self.generator(_ldimgs)
                synth_img = synth_img.to(opt.device)

                # Train adversary
                A_optimizer.zero_grad()
                adversarial_loss   = losses_.GANLoss().to(opt.device)
                adv_real = torch.cat([_ldimgs, _sdimgs], dim=1)
                    
                # adversary loss 
                adv_fake = torch.cat([_ldimgs, synth_img], dim=1).detach()
                realAdv_loss = adversarial_loss(self.adversary(adv_real), True)
                fakeAdv_loss = adversarial_loss(self.adversary(adv_fake), False) 
                totalAdv_loss = (realAdv_loss + fakeAdv_loss) / 2
    
                totalAdv_loss.backward(retain_graph=True)
                A_optimizer.step()

                 
                # Train generator
                G_optimizer.zero_grad()
                synth_imgG = self.generator(_ldimgs).to(opt.device)
                adversarial_loss   = losses_.GANLoss().to(opt.device) 
                criterion_pixel = torch.nn.L1Loss().to(opt.device)
                
                # Measure pixel-wise loss against ground truth
                loss_pixel = criterion_pixel(synth_imgG, _sdimgs)
                
                synth_loss = self.criterion(synth_imgG, _sdimgs)
                adv_real = torch.cat([_ldimgs, _sdimgs], dim=1)
                        
                loss_GAN = adversarial_loss(self.adversary(adv_real), True)
                #totalGAN_loss = loss_GAN + (100 * synth_loss)
                
                totalGAN_loss = synth_loss + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
                totalGAN_loss.backward(retain_graph=True)
                G_optimizer.step()

                

                    # time
                batches_left = opt.epochs * len(dataloader_train) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / opt.n_critic)
                prev_time = time.time()
                            
                        #print('Epoch:', epoch, '| D_loss: %.6f' % loss_D.item(),'| G_loss: %.6f' % loss_G.item())
                #print('Train::','\r[Epoch %d/%d]:' % (epoch, opt.epochs),'[Batch %d/%d]:' % (index, len(dataloader_train)), '| D_loss: %.6f' % totalAdv_loss.item(),'| G_loss: %.6f' % totalGAN_loss.item(),'ETA: %s' %time_left)

                errors_train = metrics.ErrorMetrics(_sdimgs, synth_img)
                train_perf_m.append([errors_train['PSNR'],errors_train['SSIM'],errors_train['NRMSE'],errors_train['MSE']])
                #train_perf_s.append([epoch, errors_train['PSNR'],errors_train['SSIM'],errors_train['NRMSE'],errors_train['MSE']])
                
                losses_m.append([totalAdv_loss.item(), totalGAN_loss.item()])
                #logger_exp.append([synth_loss.item(), l1_loss_.item(), Voxel_ratio_.item(), decayfactor_div_.item(), decayfactor_.item(), epoch_inv_])
                #losses_s.append([epoch, totalAdv_loss.item(), totalGAN_loss.item()])
                 
            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D.step()
            
            lss = metrics.means(losses_m, ss ='train')
            perfs = metrics.means(train_perf_m, ss ='val')
            
            totalAdv_loss, totalGAN_loss = lss
            psnr_, ssim_, nrmse_, mse_ = perfs
            
            logger_train_mean.append([epoch, totalAdv_loss, totalGAN_loss])
            losses_s.append([epoch, totalAdv_loss, totalGAN_loss])
            logger_train_perf.append([epoch, psnr_, ssim_, nrmse_, mse_])
            train_perf_s.append([epoch, psnr_, ssim_, nrmse_, mse_])
     
            losses_m.clear()
            train_perf_m.clear()
                    
                    # Save model checkpoints
            # if epoch > 200 and (epoch) % opt.checkpoint_interval == 0:
            #     torch.save(self.generator.state_dict(),opt.save_path+'/'+str('train')+'/generator_%d.pkl' % (epoch))
            #     torch.save(self.adversary.state_dict(),opt.save_path+'/'+str('train')+'/adversary_%d.pkl' % (epoch))
                
            ql_m = self.val(epoch, self.generator)
            psnr, ssim, nrmse, mse = ql_m
            logger_val_perf.append([epoch, psnr, ssim, nrmse, mse])
            val_perf_s.append([epoch, psnr, ssim, nrmse, mse])

            current_script_path = __file__
            file_name_ex = os.path.basename(current_script_path)
            file_name = file_name_ex.replace(".py", "")
            print("file_name:", file_name)

            model_saver = model_save.ModelSaver(generator=self.generator, opt=opt, file_name=file_name)
            model_saver.save_best_models() 

        # torch.save((torch.tensor(train_perf_s)), opt.save_path+'/train_perf_s.pth')
        # torch.save((torch.tensor(losses_s)), opt.save_path+'/losses_s.pth')  
        # torch.save((torch.tensor(val_perf_s)), opt.save_path+'/val_perf_s.pth')    
        # misc.Loss_plt(train_perf_s, losses_s, val_perf_s, opt.save_path, metrics=1)
        # misc.Loss_plt(train_perf_s, losses_s, val_perf_s, opt.save_path, metrics=2)
            
           
    def val(self, epoch, generator):
        generator.eval()
        print('#--------- Validating for epoch:',epoch)
        
        # Load data        
        dataset_val   = SMARTload(opt.dataset_path, opt.patch_size, opt.max_length, opt.samples_per_volume, \
                                  opt.batch_size, opt.save_path, opt.rawdata_dir, split = 'val', train_type = 'train_whole_images')
        dataloader_val = dataset_val.load()
        
        val_perf_m = []
        # ---------------------------- *validation * ---------------------------------   
        for index, imgs in enumerate(dataloader_val):
            
            sdIMGS, ldIMGS, sd_affine, ld_affine = prepare_load(imgs)
            
            if opt.gpu:
                _sdimgs, _ldimgs = sdIMGS.to(opt.device), ldIMGS.to(opt.device)

                   
            with torch.no_grad():
                synth_img = generator(_ldimgs) # generator forward pass
                synth_img = synth_img.to(opt.device)
                
            # if torch.max(synth_img) > 1:
            #     rescale = tio.RescaleIntensity((-1, 1))
            #     synth_img = rescale(synth_img)
            
            errors_val = metrics.ErrorMetrics(_sdimgs, synth_img)
            
            val_perf_m.append([errors_val['PSNR'],errors_val['SSIM'],errors_val['NRMSE'],errors_val['MSE']])
            #val_perf_s.append([epoch, errors_val['PSNR'],errors_val['SSIM'],errors_val['NRMSE'],errors_val['MSE']])
            	
            #print('Val::', '\r[Epoch %d/%d]:' % (epoch, opt.epochs),'| [SSIM]:--', errors_val['SSIM'], '| [PSNR]:--', errors_val['PSNR'],'| [NRMSE]:--', errors_val['NRMSE'],'| [MSE]:--', errors_val['MSE'])

            #print(index)
            # Save Image
            # if epoch > 200:
            #     fname ='Val'
            #     if index == 1:
            #         print(index)
            #         metrics.summarize_performance(opt.save_path +'/'+str('val')+'/', opt, _ldimgs, _sdimgs, synth_img, ld_affine, sd_affine, fname, epoch)
        
        ql_m = metrics.means(val_perf_m, ss ='val')
            
        val_perf_m.clear()
        
        return ql_m
        

            

    def test(self): 
        if not os.path.isdir(opt.save_path+'/'+str('test')+'/'):
            misc.mkNew_dir(opt.save_path+'/'+str('test')+'/')
            
            
        logger_test_perf = misc.Logger(os.path.join(opt.save_path+'/'+str('test')+'/'+'test_log_perf_main16_8.txt'), title='')
        logger_test_perf.set_names(['PSNR', 'SSIM', 'NRMSE', 'MSE', 'FID', 'SNR', 'CNR'])

        self.generator.load_state_dict(torch.load(os.path.join(opt.save_model, 'saved_model_model_pl_l1', f'generator_PSNR_{opt.ind_epoch}#.pkl')),strict=False) 
        self.generator.eval() 
        
        # Load data        
        dataset_test   = SMARTload(opt.dataset_path, opt.patch_size, opt.max_length, opt.samples_per_volume, \
                                  opt.batch_size, opt.save_path, opt.rawdata_dir, split = 'test', train_type = 'train_whole_images')
        dataloader_test = dataset_test.load()
        

        for index, imgs in enumerate(dataloader_test):

            sd_subject, ld_subject, fname = imgs['sd_subject'][tio.DATA], imgs['ld_subject'][tio.DATA], imgs['fname']
            
            fname = fname[0]
            print(fname)
          
            save_dir = os.path.join(opt.save_path+'/'+str('test')+'/'+ str(fname).replace('_sd','_pred')+'_epoch_'+str(int(opt.ind_epoch))+'.nii.gz')
            
            if opt.gpu:
                _ldimgs = ld_subject.to(opt.device)
                _sdimgs = sd_subject.to(opt.device)
                
                  
            with torch.no_grad():    
                test_ouputs = self.generator(_ldimgs)
                synth_img = test_ouputs.to(opt.device)
                
                # errors_test = metrics.ErrorMetrics(_sdimgs, synth_img)
                # logger_test_perf.append([float(errors_test['PSNR']), float(errors_test['SSIM']), float(errors_test['NRMSE']), float(errors_test['MSE']), float(errors_test['FID']), float(errors_test['SNR']), float(errors_test['CNR'])])
                # #print([errors_test['PSNR'],errors_test['SSIM'],errors_test['NRMSE'],errors_test['MSE'], errors_test['FID'], errors_test['SNR'], errors_test['CNR']])
                test_ouputs_InvResc = dataset_test.Inv_Rescale_Res_save(synth_img, fname, save_dir)

   
        
        
        
        
    
    


 


 
