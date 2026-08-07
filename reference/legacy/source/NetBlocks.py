import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm



from helpers.Attns import SSCblockaddall

class encode_Block(nn.Module):

    def __init__(self, input_nc, filt_num, InstanceNorm3d=True):
        super(encode_Block,self).__init__()
        self.InstanceNorm3d = InstanceNorm3d
        self.input_nc = input_nc
        self.filt_num = filt_num

        self.Enc_blk1 = spectral_norm(nn.Conv3d(self.input_nc, self.filt_num, kernel_size=4, stride=2, padding=1, bias=False)) #change Kernel to 7x7 stride to 1
        self.Enc_blk2 = nn.LeakyReLU(0.2, inplace=True)
        self.Enc_blk3 = spectral_norm(nn.Conv3d(self.filt_num, self.filt_num, kernel_size=3, stride=1, padding=1, bias=False)) #change stride to 2
        self.Enc_blk4 = nn.InstanceNorm3d(filt_num)

    def forward(self, x):
        if self.InstanceNorm3d:
            nn1 = nn.Sequential(self.Enc_blk1, self.Enc_blk4, self.Enc_blk2, self.Enc_blk3, self.Enc_blk4, self.Enc_blk2)(x)
        else:
            nn1 = nn.Sequential(self.Enc_blk1, self.Enc_blk2, self.Enc_blk3, self.Enc_blk2)(x)
        
        return nn1
        
class encode_Block1(nn.Module):

    def __init__(self, input_nc, filt_num, InstanceNorm3d=True):
        super(encode_Block1,self).__init__()
        self.InstanceNorm3d = InstanceNorm3d
        self.input_nc = input_nc
        self.filt_num = filt_num

        self.Enc_blk1 = spectral_norm(nn.Conv3d(self.input_nc, self.filt_num, kernel_size=7, stride=1, padding=3, bias=False)) #change Kernel to 7x7 stride to 1
        self.Enc_blk2 = nn.LeakyReLU(0.2, inplace=True)
        self.Enc_blk3 = spectral_norm(nn.Conv3d(self.filt_num, self.filt_num, kernel_size=4, stride=2, padding=1, bias=False)) #change stride to 2
        self.Enc_blk4 = nn.InstanceNorm3d(filt_num)

    def forward(self, x):
        if self.InstanceNorm3d:
            nn1 =  nn.Sequential(self.Enc_blk1, self.Enc_blk4, self.Enc_blk2, self.Enc_blk3, self.Enc_blk4, self.Enc_blk2)(x)
        else:
            nn1 =  nn.Sequential(self.Enc_blk1, self.Enc_blk2, self.Enc_blk3, self.Enc_blk2)(x)
        
        return nn1
        


class discriminator_block(nn.Module):

    def __init__(self, input_nc, filt_num, InstanceNorm3d=True):
        super(discriminator_block,self).__init__()
        self.InstanceNorm3d = InstanceNorm3d
        self.input_nc = input_nc
        self.filt_num = filt_num

        self.Enc_blk1 = spectral_norm(nn.Conv3d(self.input_nc, self.filt_num, kernel_size=4, stride=2, padding=1, bias=False))
        self.Enc_blk2 = nn.LeakyReLU(0.2, inplace=True)
        self.Enc_blk4 = nn.InstanceNorm3d(self.filt_num)

    def forward(self, x):
        if self.InstanceNorm3d:
            nn1 = nn.Sequential(self.Enc_blk1, self.Enc_blk4, self.Enc_blk2)(x)
        else:
            nn1 = nn.Sequential(self.Enc_blk1, self.Enc_blk2)(x)
            
        return nn1



class decode_Block(nn.Module):

    def __init__(self, input_nc, filt_num, dropout3d=True):
        super(decode_Block,self).__init__()
        self.dropout3d = dropout3d
        self.input_nc = input_nc
        self.filt_num = filt_num
        

        act_fnO = nn.ReLU(inplace=True)

        self.Dc_blk1 = spectral_norm(nn.ConvTranspose3d(self.input_nc, self.filt_num, kernel_size=4, stride=2, padding=1, bias=False))
        self.Dc_blk2 = act_fnO
        self.Dc_blk3 = nn.Dropout3d(0.5)
        self.Dc_blk4 = nn.InstanceNorm3d(self.filt_num)


    def forward(self, x, concat_in):
        if self.dropout3d:
            conv2 = nn.Sequential(self.Dc_blk1,  self.Dc_blk4, self.Dc_blk3, self.Dc_blk2)(x)
            conv2 = torch.cat([conv2, concat_in], 1)
       
        else:
            conv2 = nn.Sequential(self.Dc_blk1, self.Dc_blk4, self.Dc_blk2)(x)
            conv2 = torch.cat([conv2, concat_in], 1)
        
        return conv2 
        

class up_size(nn.Module):

    def __init__(self, input_nc, filt_num):
        super(up_size,self).__init__()
        self.input_nc = input_nc
        self.filt_num = filt_num
    
        act_fnO = nn.ReLU(inplace=True)
        self.Dc_blk1 = spectral_norm(nn.ConvTranspose3d(self.input_nc, self.filt_num, kernel_size=4, stride=2, padding=1, bias=False))
        self.Dc_blk2 = act_fnO
        self.Dc_blk4 = nn.InstanceNorm3d(self.filt_num)


    def forward(self, x):
        conv2 = nn.Sequential(self.Dc_blk1, self.Dc_blk4, self.Dc_blk2)(x)
        return conv2 


class bottleneck(nn.Module):
    def __init__(self, input_nc, filt_num, InstanceNorm3d=True):
        super(bottleneck,self).__init__()
        self.InstanceNorm3d = InstanceNorm3d
        self.input_nc = input_nc
        self.filt_num = filt_num

        self.Enc_blk1 = spectral_norm(nn.Conv3d(self.input_nc, self.filt_num, kernel_size=4, stride=2, padding=1, bias=False))
        self.Enc_blk2 = nn.LeakyReLU(0.2, inplace=True)
        self.Enc_blk3 = spectral_norm(nn.Conv3d(self.filt_num, self.filt_num, kernel_size=3, stride=1, padding=1, bias=False))
        self.Enc_blk4 = nn.InstanceNorm3d(filt_num)
        self.Enc_blk5 = spectral_norm(nn.ConvTranspose3d(self.filt_num, self.filt_num, kernel_size=3, stride=1, padding=0, bias=False))
        
    def forward(self, x):
        if self.InstanceNorm3d:
            _in = nn.Sequential(self.Enc_blk1, self.Enc_blk4, self.Enc_blk2, self.Enc_blk3, self.Enc_blk4, self.Enc_blk2)(x)
            _out = self.Enc_blk5(_in)

        else:
            _in = nn.Sequential(self.Enc_blk1, self.Enc_blk2, self.Enc_blk3, self.Enc_blk2)(x)
            _out = self.Enc_blk5(_in)
            
        return _out
                  



class Attention_Block_dwn(nn.Module):
    
    def __init__(self, input_nc, filter_num, config='SSAM1'): 
        super(Attention_Block_dwn, self).__init__()

        self.config=config
        self.input_nc = input_nc
        self.filter_num  = filter_num
        self.block_name = SSCblockaddall(self.input_nc, self.filter_num, self.config)
        

    def forward(self, x):
        
        y = self.block_name(x)

        return y


class Attention_Block_up(nn.Module):
    
    def __init__(self, input_nc, filter_num, config='SSAM1'): 
        super(Attention_Block_up, self).__init__()

        self.config=config
        self.input_nc = input_nc
        self.filter_num  = filter_num
        self.block_name = SSCblockaddall(self.input_nc, self.filter_num , self.config)
        #self.encode =nn.Conv3d(self.input_nc,self.input_nc, kernel_size=4, stride=1, padding=1, bias=False)
        

    def forward(self, x1):
        
        y1 = self.block_name(x1)

        return y1

class final_layer(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(final_layer, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        self.final_conv = nn.ConvTranspose3d(self.input_nc, self.output_nc, kernel_size=4, stride=2, padding=1) #change Kernel to 7
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.final_conv(x)
        return self.tanh(x)

