#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Wed July 28 21:36:19 2022
#Created by Raymond Confidence


import torch
import torch.nn as nn
from torchsummary import summary

from helpers import NetBlocks as block #helpers.
final_layer = block.final_layer
encode_Block = block.encode_Block
decode_Block = block.decode_Block
bottleneck = block.bottleneck
up_size = block.up_size
encode_Block1 = block.encode_Block1
Attention_Block_up = block.Attention_Block_up
Attention_Block_dwn = block.Attention_Block_dwn
discriminator_block = block.discriminator_block
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class generator(nn.Module):

    def __init__(self, config='SSAM1'):
        super(generator,self).__init__()

        self.i_nc = 1 #input channel
        self.o_nc = 1 #output channe;
        self.filt_num = 32 # Number of fliters applied
        self.config = config
        
                    # ~~~ Downsizing Paths ~~~~~~ #
        
                    # ~~~ Encoding Paths ~~~~~~ #
        
        #######################################################################
        # LD-Encoder
        #######################################################################
        self.encode_1_0 = encode_Block1(self.i_nc, self.filt_num, InstanceNorm3d=False) #1,16
        # 64 X 64 X 64
        
        self.encode_2_0 = encode_Block(self.filt_num, self.filt_num*2) #16,32
        # 32 X 32 X 32

        self.encode_3_0 = encode_Block(self.filt_num*2, self.filt_num*4) #32,64
        # 16 X 16 X 16
    
        self.encode_4_0 = encode_Block(self.filt_num*4, self.filt_num*8) #64, 128
        # 8 X 8 X 8

        self.encode_5_0 = encode_Block(self.filt_num*8, self.filt_num*16) #128, 256
        # 4 X 4 X 4
        
        self.encode_6_0 = encode_Block(self.filt_num*16, self.filt_num*16) #128, 256
        # 2 X 2 X 2
   
        self.encode_7_0 = encode_Block(self.filt_num*16, self.filt_num*16, InstanceNorm3d=False) # 256, 256
        # 1 X 1 X 1
        
        #######################################################################
        # ~~~ Attention Path ~~~~~~ #
       #######################################################################
       
        self.fu_encoder_1 = Attention_Block_dwn(self.filt_num*4, self.filt_num*4, self.config)
        
        self.fu_encoder_2 = Attention_Block_dwn(self.filt_num*8, self.filt_num*8, self.config)
        
        #self.fu_encoder_3 = Attention_Block_dwn(self.filt_num*16, self.filt_num*16, self.config)

       #######################################################################
        # ~~~ Decoding Path ~~~~~~ #
       #######################################################################
        self.decod_1_0 = decode_Block(self.filt_num*16, self.filt_num*16) #2
        #2 X 2 X 2
        
        self.decod_2_0 = decode_Block(self.filt_num*32, self.filt_num*16) #4
        # 4 X 4 X 4
        
        self.decod_3_0 = decode_Block(self.filt_num*32, self.filt_num*8) #8
        # 8 X 8 X 8
        
        self.decod_4_0 = decode_Block(self.filt_num*16, self.filt_num*4, dropout3d=False) #16
        # 16 X 16 X 16
        
        self.decod_5_0 = decode_Block(self.filt_num*8, self.filt_num*2, dropout3d=False)   #32
        # 32 X 32 X 32
        
        self.decod_6_0 = decode_Block(self.filt_num*4, self.filt_num, dropout3d=False)   #64
        # 64 X 64 X 64
        
        self.out = final_layer(self.filt_num*2, self.o_nc) #128
        
        

    
    def forward(self,input):

        input = input.to(device,dtype=torch.float32) #(128, 128, 128)
        ## print(input.shape)
        
        
        #------- Reduce image size ------
        encoder_1_0 = self.encode_1_0(input) #(64, 64, 64)
    
        # ##############################
        
        # -----  First Level -------- 
        encoder_2_0 = self.encode_2_0(encoder_1_0)  #(32, 32, 32)
        # print("encoder_2_0", encoder_2_0.shape)
    
        # -----  Second Level --------
        
        encoder_3_0 = self.encode_3_0(encoder_2_0) # (16, 16, 16)
        f_block_1  = self.fu_encoder_1(encoder_3_0)
        # print("encoder_3_0", encoder_3_0.shape)
        
        # -----  Third Level --------
        encoder_4_0 = self.encode_4_0(f_block_1) # (8, 8, 8)
        f_block_2  = self.fu_encoder_2(encoder_4_0)  
        # print("encoder_4_0", encoder_4_0.shape)
     
         # -----  Fourth Level --------
        encoder_5_0 = self.encode_5_0(f_block_2) # (4, 4, 4)
        #f_block_3  = self.fu_encoder_3(encoder_5_0) 
        # print("encoder_5_0", encoder_5_0.shape)

         # -----  Fifth Level --------
        encoder_6_0 = self.encode_6_0(encoder_5_0) # (2, 2, 2) 
        # print("encoder_6_0", encoder_6_0.shape)
        
        # -----  sith Level --------
        encoder_7_0 = self.encode_7_0(encoder_6_0) # (1, 1, 1)
        # print("encoder_7_0", encoder_7_0.shape)
      
        #######################################################################                                                                                                
        # ~~~~~~ Decoding 
        decoder_1_0 = self.decod_1_0(encoder_7_0,encoder_6_0)   # (2, 2, 2)
        # print("decoder_1_0", decoder_1_0.shape)
        
        decoder_2_0 = self.decod_2_0(decoder_1_0,encoder_5_0)   # (4, 4, 4)
        # print("decoder_2_0", decoder_2_0.shape)
        
        decoder_3_0 = self.decod_3_0(decoder_2_0,encoder_4_0)   # (8, 8, 8)
        # print("decoder_3_0", decoder_3_0.shape)
        
        decoder_4_0 = self.decod_4_0(decoder_3_0,encoder_3_0)   # (16, 16, 16)
        # print("decoder_4_0", decoder_4_0.shape)
       
        decoder_5_0 = self.decod_5_0(decoder_4_0,encoder_2_0)   #(32, 32, 32)
        # print("decoder_5_0", decoder_5_0.shape)
  
        decoder_6_0 = self.decod_6_0(decoder_5_0,encoder_1_0)   #(64, 64, 64)
        # print("decoder_6_0", decoder_6_0.shape)
      
        decod_out = self.out(decoder_6_0)  # (128, 128, 128)
        # print("decod_out", decod_out.shape)
                        
        return decod_out


class adversary(nn.Module):
    def __init__(self,i_nc=2, filt_num=32):
        super(adversary,self).__init__()

        self.i_nc = i_nc
        self.filt_num = filt_num
        

        self.adv_1_0 = discriminator_block(self.i_nc, self.filt_num, InstanceNorm3d=True)
        self.adv_2_0 = discriminator_block(self.filt_num, self.filt_num*2)
        self.adv_3_0 = discriminator_block(self.filt_num*2, self.filt_num*4)
        self.adv_4_0 = nn.Sequential(nn.Conv3d(self.filt_num*4, self.filt_num*8, kernel_size=4, stride=1, padding = 1), nn.LeakyReLU(0.2, inplace=True))
        self.final = nn.Sequential(nn.Conv3d(self.filt_num*4, 1, kernel_size=4, padding = 1), nn.Sigmoid()) #, nn.Sigmoid()
        


    def forward(self, x):
        x = x.to(device,dtype=torch.float) #(2,192,192,192)
   
        adv1 = self.adv_1_0(x) #1, 16 -- 96
        adv2 = self.adv_2_0(adv1) #16, 32 -- 48
        adv3 = self.adv_3_0(adv2) #32, 64 -- 24
 
        final = self.final(adv3) #64, 128 -- 12
        ## print(torch.max(final))


        return final

# if __name__ == '__main__':
#     import torch
#     from torch.autograd import Variable
#     #from torchsummaryX import summary
   

#     torch.cuda.set_device(0)
#     generator = generator()

#     net = generator.cuda().eval()

#     if torch.cuda.device_count() > 1:
#         Nnet = torch.nn.DataParallel(net)

#     data = Variable(torch.randn(1, 1, 192, 192, 192)).cuda()

#     out = net(data)
#     summary(net,(1,192,192,192))
#     # print("out size: {}".format(out.size()))