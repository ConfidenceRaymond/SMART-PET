import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from helpers.SSIMattn import self_ssim3d




cuda = True if torch.cuda.is_available() else False
FloatTensor   = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor    = torch.cuda.LongTensor if cuda else torch.LongTensor








class SelfAttention(nn.Module):
    """
    input:N*C*D*H*W
    """
    def __init__(self, in_nc, filter_num):
        super().__init__()
   
        self.gama = nn.Parameter(torch.tensor([0.0]))


        self.in_nc = in_nc
        self.filter_num = filter_num
        self.conv3d_3 = nn.Sequential(
            # Conv3d output:N*C*D*H*W
            nn.Conv3d(in_channels=self.in_nc, out_channels=self.filter_num, kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.filter_num),
            nn.ReLU(inplace=True),
        )


        self.conv3d_1 = nn.Sequential(
            # Conv3d output:N*C*D*H*W
            nn.Conv3d(in_channels=self.in_nc, out_channels=self.filter_num, kernel_size=1),
            nn.InstanceNorm3d(self.filter_num),
            nn.ReLU(inplace=True),
        )


    def get_dim(self, x_y_z):
        N = x_y_z.shape[0]
        C = x_y_z.shape[1]
        D = x_y_z.shape[2]
        H = x_y_z.shape[3]
        W = x_y_z.shape[4]
       
        return N, C, D, H, W
       


    def Cal_Patt(self, k_x, q_x, v_x, N, C, D, H, W):
        """
        input:N*C*D*H*W
        """
        k_x_flatten = k_x.reshape((N, C, D, 1, H * W))
        q_x_flatten = q_x.reshape((N, C, D, 1, H * W))
        v_x_flatten = v_x.reshape((N, C, D, 1, H * W))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 2, 4, 3), k_x_flatten)


        r_x = F.softmax(sigma_x, dim=4, dtype=torch.float32)
        Patt = torch.matmul(v_x_flatten, r_x).reshape(N, C, D, H, W)
        return Patt


   
    def Cal_Datt(self, k_x, q_x, v_x, N, C, D, H, W):
        """
        input:N*C*D*H*W
        """


        k_x_flatten = k_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        q_x_flatten = q_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        v_x_flatten = v_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 2, 3, 5, 4), k_x_flatten)
        r_x = F.softmax(sigma_x, dim=5, dtype=torch.float32)
        Datt = torch.matmul(v_x_flatten, r_x).reshape(N, C, H, W, D)
        return Datt.permute(0, 1, 4, 2, 3)


   
    def forward(self, x):
        N, C, D, H, W = self.get_dim(x)
        v_x = self.conv3d_3(x)
        k_x = self.conv3d_1(x)
        q_x = self.conv3d_1(x)
   
       
        Patt = self.Cal_Patt(k_x, q_x, v_x, N, C, D, H, W)
        Datt = self.Cal_Datt(k_x, q_x, v_x, N, C, D, H, W)
       
        Y = self.gama*(Patt + Datt) + x
       
        return Y








class CA_Attention(nn.Module):
    def __init__(self, in_nc, filter_num):
        super(CA_Attention, self).__init__()


        self.in_nc = in_nc
        self.filter_num = filter_num
       
        self.sigmoid = nn.Sigmoid()
        self.AglobalAvgPool = nn.AdaptiveAvgPool3d(1)
        self.AglobalMaxPool = nn.AdaptiveMaxPool3d(1)
       
        self.b=1
        self.gamma=2
        reduction_ratio = 16
       
        self.t = int(abs((math.log(self.in_nc,2)+self.b)/self.gamma))
        self.kernel_size = self.t if self.t%2 else self.t+1
       
        self.kernel = 7
        self.inner_nc = int(filter_num // reduction_ratio)


        self.conv1x1 = nn.Conv3d(self.in_nc, 1, kernel_size=1, padding='same', bias=False)
        self.conv1 = nn.Conv3d(2, 1, kernel_size=self.kernel, padding=self.kernel//2, bias=False)


        self.covv = nn.Sequential(
            nn.Conv3d(self.in_nc, self.inner_nc, kernel_size=self.kernel_size, padding='same', bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.inner_nc, self.in_nc, kernel_size=self.kernel_size, padding='same', bias=False),
            nn.ReLU(inplace=True))


       
    def forward(self, x):
        ''' Channel attention '''
        avgO_ = self.AglobalAvgPool(x)
       
        avgO = self.covv(avgO_)
        maxO = self.AglobalMaxPool(x)
        maxO = self.covv(maxO)
        addO = torch.add(avgO, maxO)
        ca = self.sigmoid(addO)
        ca = ca.view(ca.size(0), ca.size(1), 1, 1, 1)
        ca = torch.mul(ca, x)
        ''' Spatial attention '''
        avg_out = torch.mean(x, dim=1, keepdim=True, dtype=torch.float32)
        mx_out, _ = torch.max(x, dim=1, keepdim=True)
        cat_out = torch.cat((avg_out, mx_out), dim=1)
        mn_mx_out = self.conv1(cat_out)


        conv1x1= self.conv1x1(x)
        sa = self.sigmoid(torch.add(mn_mx_out, conv1x1))
        sa = sa.view(sa.size(0), 1, sa.size(2), sa.size(3), sa.size(4))
       
        ''' Channel & Spatial attention '''
        csa = torch.mul(sa, ca)
   
        return csa




class similarityattention3d(nn.Module):
    def __init__(self, in_nc, filter_num):
        super(similarityattention3d, self).__init__()
        self.in_nc = in_nc
        self.filter_num = filter_num
        self.kernel = 7
        self.convs = nn.Sequential(
            # Conv3d output:N*C*D*H*W
            nn.Conv3d(in_channels=self.in_nc, out_channels=self.filter_num, kernel_size=self.kernel, padding=self.kernel//2, bias=False),
            nn.InstanceNorm3d(self.filter_num),
            nn.ReLU(inplace=True),
        )
        self.conv1_1 = nn.Conv3d(self.in_nc, 1, kernel_size=self.kernel, padding=self.kernel//2, bias=False)
     
         
    def forward(self, x):


        similarity_in = self_ssim3d(x)
        convs = self.convs(similarity_in)
        conv1_1 = self.conv1_1(similarity_in)
        similarity_out = torch.sigmoid(torch.add(convs, conv1_1))
        similarity_out = torch.mul(similarity_out, x)


        return similarity_out


class SSCblockaddall(nn.Module):
    def __init__(self, in_nc, filter_num, config='SSAM1'):
        super(SSCblockaddall, self).__init__()
       
        self.config = config
        self.filter_num=filter_num
        self.ssa = SelfAttention(in_nc, self.filter_num) #Self attention
        self.csa = CA_Attention(in_nc, self.filter_num)
        self.sam3d = similarityattention3d(in_nc, filter_num)
        self.conc = nn.Conv3d(in_nc*3,in_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.conc5 = nn.Conv3d(in_nc*2,in_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.encode = nn.Conv3d(in_nc,in_nc, kernel_size=3, stride=1, padding=1, bias=False)




    def forward(self, x):
       
        if self.config == 'SSAM1':
            sam3d_out, csa_out = self.sam3d(x) * x, self.csa(x) * x
           
            mul_SSAM = sam3d_out * csa_out
            sum_SSAM = sam3d_out + csa_out
           
            cat_SSAM = torch.cat((mul_SSAM,sum_SSAM),dim=1)


            out = self.conc5(cat_SSAM)
       
        elif self.config == 'SSAM2':
            ssa_out, sam3d_out, csa_out = self.ssa(x) * x, self.sam3d(x) * x, self.csa(x) * x
           
            mul_all = ssa_out * sam3d_out * csa_out
     
            out = self.encode(mul_all)
           
        elif self.config == 'SSAM3':
           
            sam3d_out, csa_out = self.sam3d(x) * x, self.csa(x) * x
           
            sum_SSAM = sam3d_out + csa_out


            out = self.encode(sum_SSAM)
           
        elif self.config == 'SSAM4':
            ssa_out, sam3d_out, csa_out = self.ssa(x) * x, self.sam3d(x) * x, self.csa(x) * x
           
            sum_all = ssa_out + sam3d_out + csa_out
     
            out = self.encode(sum_all)
                       
        elif self.config == 'SSAM5':
            ssa_out = self.ssa(x) * x
            sam3d_out, csa_out = self.sam3d(ssa_out) * ssa_out, self.csa(ssa_out) * ssa_out
                     
            cat_SSAM = torch.cat((sam3d_out,csa_out),dim=1)
            out = self.conc5(cat_SSAM)
                   
        elif self.config == 'SSAM6':
            sam3d_out, csa_out = self.sam3d(x) * x, self.csa(x) * x
           
            cat_SSAM = sam3d_out * csa_out


            out = self.encode(cat_SSAM)
           
        elif self.config == 'CSA':
            csa_out = self.csa(x) * x


            out = self.encode(csa_out)
           
        elif self.config == 'SAM':
            sam3d_out = self.sam3d(x) * x


            out = self.encode(sam3d_out)
           
        elif self.config == 'SSA':
            ssa_out = self.ssa(x) * x


            out = self.encode(ssa_out)
           
        else:
            raise ValueError('Invalid Attention provided')


        return out