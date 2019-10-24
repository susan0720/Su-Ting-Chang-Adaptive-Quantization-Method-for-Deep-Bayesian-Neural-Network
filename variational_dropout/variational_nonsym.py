import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
scale1 = 4
scale2 = 4
scale3 = 2
scale4 = 2
scale5 = 8
scale6 = 8
scale7 = 10
scale8 = 10
def gaussian_window( theta, c):
    a= (theta-c)**2
    b=-0.5 *((0.075)**-2)*a
    val = t.exp(t.clamp(b,max=75))
    
#     val[val != val] = 0 
#     if math.isnan(t.sum(theta)):
# #         val_list = val.reshape(1,-1).tolist()
#         print(c)
#         print(t.max(theta),t.min(theta))
#         print(hi)
    return val
def kllu(log_alpha):
    k = [0.63576, 1.87320, 1.48695]
    first_term = k[0] * F.sigmoid(k[1] + k[2] * log_alpha)
    second_term = 0.5 * t.log(1 + t.exp(-log_alpha))
    return -(first_term - second_term - k[0])
class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)
class VariationalDropout(nn.Module):
    def __init__(self, input_size, out_size, log_sigma2=-10, threshold=3):
        """
        :param input_size: An int of input size
        :param log_sigma2: Initial value of log sigma ^ 2.
               It is crusial for training since it determines initial value of alpha
        :param threshold: Value for thresholding of validation. If log_alpha > threshold, then weight is zeroed
        :param out_size: An int of output size
        """
        super(VariationalDropout, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.theta = Parameter(t.FloatTensor(input_size, out_size))
        self.bias = Parameter(t.Tensor(out_size))

        self.log_sigma2 = Parameter(t.FloatTensor(input_size, out_size).fill_(log_sigma2))
        self.sz = input_size*out_size
        self.s1 = Parameter(t.Tensor([scale1]))
        self.s2 = Parameter(t.Tensor([scale2]))
        self.s3 = Parameter(t.Tensor([scale3]))
        self.s4 = Parameter(t.Tensor([scale4]))
#         self.code = t.Tensor([0.2,0,-0.2,0.2, -0.2])
        self.code = t.Tensor([0.5,-0.5,0.5,-0.5])
        self.reset_parameters()

        self.k = [0.63576, 1.87320, 1.48695]

        self.threshold = threshold

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)

        self.theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def clip(input, to=3):
        input = input.masked_fill(input < -to, -to)
        input = input.masked_fill(input > to, to)

        return input

    def kld(self, idx):
        if math.isnan(t.sum(self.theta.data)):
            print(t.max(self.theta.data),t.min(self.theta.data))
            print(hi)
                   
        window1 = gaussian_window(self.theta*self.s1,0.5)
        window2 = gaussian_window(self.theta*self.s2,-0.5)
        window3 = gaussian_window(self.theta*self.s3,0.5)
        window4 = gaussian_window(self.theta*self.s4,-0.5)

        log_alpha1 = self.clip(self.log_sigma2 + 2*t.log(self.s1)- t.log((self.theta*self.s1-0.5) ** 2))
#         log_alpha2 = self.log_sigma2 - t.log((self.theta) ** 2)
        log_alpha3 = self.clip(self.log_sigma2 + 2*t.log(self.s2)- t.log((self.theta*self.s2+0.5) ** 2))
        log_alpha4 = self.clip(self.log_sigma2 + 2*t.log(self.s3)- t.log((self.theta*self.s3-0.5) ** 2))
        log_alpha5 = self.clip(self.log_sigma2 + 2*t.log(self.s4)- t.log((self.theta*self.s4+0.5) ** 2))
           
        F_KLLU1 = kllu(log_alpha1)
#         F_KLLU2 = kllu(log_alpha2)
        F_KLLU3 = kllu(log_alpha3)
        F_KLLU4 = kllu(log_alpha4)
        F_KLLU5 = kllu(log_alpha5)
#         F_KL = F_KLLU1*window1 + F_KLLU3*window2 + F_KLLU2*(1-window1-window2-window3-window4) + F_KLLU4*window3 + F_KLLU5*window4
        F_KL = F_KLLU1*window1 + F_KLLU3*window2 + F_KLLU4*window3 + F_KLLU5*window4 
        return F_KL.sum() / (self.sz)
    
    def forward(self, input, train):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """
#         theta = self.theta.data
#         _theta = t.abs(theta) == float('inf')  # -inf的位置为1
#         self.theta.data = self.theta.data.masked_fill_(_theta,0) # 将-inf填充为0
        c1 = (self.theta-0.5/self.s1)**2
#         print(t.max(c1))
#         c2 = (self.theta)**2
        c3 = (self.theta+0.5/self.s2)**2
        c4 = (self.theta-0.5/self.s3)**2
        c5 = (self.theta+0.5/self.s4)**2
#         mean = t.min(t.min(t.min(t.min(c1,c2),c3),c4),c5)
#         c = t.stack((c1,c2,c3,c4,c5),0)
        mean = t.min(t.min(t.min(c1,c5),c3),c4)
        c = t.stack((c1,c3,c4,c5),0)
        idx = t.argmin(c,0)
        kld = self.kld(idx)
        if not train:
            """
            mask = log_alpha > self.threshold
            return t.addmm(self.bias, input, self.theta.masked_fill(mask, 0))
            """
            theta_q = self.theta.data.clone()
#             theta_q[:] = (idx==0).float()*self.code[idx].cuda()/self.s1 + (idx==1).float()*self.code[idx].cuda() + (idx==2).float()*self.code[idx].cuda()/self.s2+ (idx==3).float()*self.code[idx].cuda()/self.s3+ (idx==4).float()*self.code[idx].cuda()/self.s4
            theta_q[:] = (idx==0).float()*self.code[idx].cuda()/self.s1 + (idx==1).float()*self.code[idx].cuda()/self.s2 +(idx==2).float()*self.code[idx].cuda()/self.s3 + (idx==3).float()*self.code[idx].cuda()/self.s4
            mu = t.mm(input, theta_q)
            
    
            return mu + self.bias, kld
            
            
#         weight = self.theta*self.s1*(idx==0).float() + self.theta*(idx==1).float()+ self.theta*self.s2*(idx==2).float()+ self.theta*self.s3*(idx==3).float() +self.theta*self.s4*(idx==4).float()
        
#         weight_std = self.log_sigma2.exp()*(self.s1**2)*(idx==0).float() + self.log_sigma2.exp()*(idx==1).float()+ self.log_sigma2.exp()*(self.s2**2)*(idx==2).float()+ self.log_sigma2.exp()*(self.s3**2)*(idx==3).float()+ self.log_sigma2.exp()*(self.s4**2)*(idx==4).float()
        
        weight = self.theta*self.s1*(idx==0).float() + self.theta*self.s2*(idx==1).float()+ self.theta*self.s3*(idx==2).float() + self.theta*self.s4*(idx==3).float()
        
        weight_std = self.log_sigma2.exp()*(self.s1**2)*(idx==0).float() + self.log_sigma2.exp()*(self.s2**2)*(idx==1).float() + self.log_sigma2.exp()*(self.s3**2)*(idx==2).float() + self.log_sigma2.exp()*(self.s4**2)*(idx==3).float()
        mu = t.mm(input, weight)
        std = t.sqrt(t.mm(input ** 2, weight_std) + 1e-6)

        eps = Variable(t.randn(*mu.size()))
        if input.is_cuda:
            eps = eps.cuda()
        out = std * eps + mu + self.bias
        return out, kld


class VariationalDropoutCNN(nn.Module):
    def __init__(self, img_row, in_channel, out_channel, kernel_size, stride=1,padding=0, dilation=1, groups=1, log_sigma2=-10, threshold=3):
        """
        :param input_channel: An int of input channel
        :param log_sigma2: Initial value of log sigma ^ 2.
               It is crusial for training since it determines initial value of alpha
        :param threshold: Value for thresholding of validation. If log_alpha > threshold, then weight is zeroed
        :param out_channel: An int of output channel
        """
        super(VariationalDropoutCNN, self).__init__()
        self.m = img_row
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.theta = Parameter(t.Tensor(out_channel, in_channel // groups, kernel_size, kernel_size))
        self.bias = Parameter(t.Tensor(out_channel, self.m-kernel_size+1, self.m-kernel_size+1))
        self.sz = out_channel * (in_channel // groups) * kernel_size**2
        self.log_sigma2 = Parameter(t.FloatTensor(out_channel, in_channel // groups, kernel_size, kernel_size).fill_(log_sigma2))
        self.s1 = Parameter(t.Tensor([scale1]))
        self.s2 = Parameter(t.Tensor([scale2]))
        self.s3 = Parameter(t.Tensor([scale3]))
        self.s4 = Parameter(t.Tensor([scale4]))
#         self.code = t.Tensor([0.2,0,-0.2,0.2,-0.2])
        self.code = t.Tensor([0.5,-0.5,0.5,-0.5])
        self.reset_parameters()

        self.k = [0.63576, 1.87320, 1.48695]

        self.threshold = threshold

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_channel)

        self.theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def clip(input, to=3):
        input = input.masked_fill(input < -to, -to)
        input = input.masked_fill(input > to, to)

        return input

    
    
    def kld(self, idx):
                           
        window1 = gaussian_window(self.theta*self.s1,0.5)
        window2 = gaussian_window(self.theta*self.s2,-0.5)
        window3 = gaussian_window(self.theta*self.s3,0.5)
        window4 = gaussian_window(self.theta*self.s4,-0.5)

        log_alpha1 = self.clip(self.log_sigma2 + 2*t.log(self.s1)- t.log((self.theta*self.s1-0.5) ** 2))
#         log_alpha2 = self.log_sigma2 - t.log((self.theta) ** 2)
        log_alpha3 = self.clip(self.log_sigma2 + 2*t.log(self.s2)- t.log((self.theta*self.s2+0.5) ** 2))
        log_alpha4 = self.clip(self.log_sigma2 + 2*t.log(self.s3)- t.log((self.theta*self.s3-0.5) ** 2))
        log_alpha5 = self.clip(self.log_sigma2 + 2*t.log(self.s4)- t.log((self.theta*self.s4+0.5) ** 2))
                           
        F_KLLU1 = kllu(log_alpha1)
#         F_KLLU2 = kllu(log_alpha2)
        F_KLLU3 = kllu(log_alpha3)
        F_KLLU4 = kllu(log_alpha4)
        F_KLLU5 = kllu(log_alpha5)
#         F_KL = F_KLLU1*window1 + F_KLLU3*window2 + F_KLLU2*(1-window1-window2-window3-window4) + F_KLLU4*window3 + F_KLLU5*window4
        F_KL = F_KLLU1*window1 + F_KLLU3*window2 + F_KLLU4*window3 + F_KLLU5*window4 
        return F_KL.sum() / (self.sz)

    def forward(self, input, train):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """
#         theta = self.theta.data
#         _theta = t.abs(theta) == float('inf')   # inf的位置为1
#         self.theta.data = self.theta.data.masked_fill_(_theta,0) # 将-inf填充为0
        c1 = (self.theta-0.5/self.s1)**2
#         c2 = (self.theta)**2
        c3 = (self.theta+0.5/self.s2)**2
        c4 = (self.theta-0.5/self.s3)**2
        c5 = (self.theta+0.5/self.s4)**2
#         mean = t.min(t.min(t.min(t.min(c1,c2),c3),c4),c5)
#         c = t.stack((c1,c2,c3,c4,c5),0)
        mean = t.min(t.min(t.min(c1,c5),c3),c4)
        c = t.stack((c1,c3,c4,c5),0)
        idx = t.argmin(c,0)
        
        
        kld = self.kld(idx)

        if not train:
            """
            mask = log_alpha > self.threshold
            return F.conv2d( input, weight = self.theta.masked_fill(mask, 0), stride=self.stride, 
                          padding=self.padding,dilation=self.dilation, groups=self.groups)
            """
            
            theta_q = self.theta.data.clone()
            theta_q = self.theta.data.clone()
#             theta_q[:] = (idx==0).float()*self.code[idx].cuda()/self.s1 + (idx==1).float()*self.code[idx].cuda() + (idx==2).float()*self.code[idx].cuda()/self.s2+ (idx==3).float()*self.code[idx].cuda()/self.s3+ (idx==4).float()*self.code[idx].cuda()/self.s4
            theta_q[:] = (idx==0).float()*self.code[idx].cuda()/self.s1 + (idx==1).float()*self.code[idx].cuda()/self.s2 +(idx==2).float()*self.code[idx].cuda()/self.s3 + (idx==3).float()*self.code[idx].cuda()/self.s4
            mu = F.conv2d(input, weight = theta_q, stride=self.stride, 
                          padding=self.padding,dilation=self.dilation, groups=self.groups)
            
    
            return mu+self.bias , kld
#         weight = self.theta*self.s1*(idx==0).float() + self.theta*(idx==1).float()+ self.theta*self.s2*(idx==2).float()+ self.theta*self.s3*(idx==3).float() +self.theta*self.s4*(idx==4).float()
#         weight_std = self.log_sigma2.exp()*(self.s1**2)*(idx==0).float() + self.log_sigma2.exp()*(idx==1).float()+ self.log_sigma2.exp()*(self.s2**2)*(idx==2).float()+ self.log_sigma2.exp()*(self.s3**2)*(idx==3).float()+ self.log_sigma2.exp()*(self.s4**2)*(idx==4).float()
        weight = self.theta*self.s1*(idx==0).float() + self.theta*self.s2*(idx==1).float()+ self.theta*self.s3*(idx==2).float() + self.theta*self.s4*(idx==3).float()
        
        weight_std = self.log_sigma2.exp()*(self.s1**2)*(idx==0).float() + self.log_sigma2.exp()*(self.s2**2)*(idx==1).float() + self.log_sigma2.exp()*(self.s3**2)*(idx==2).float() + self.log_sigma2.exp()*(self.s4**2)*(idx==3).float()
        mu = F.conv2d(input, weight = weight, stride=self.stride, 
                          padding=self.padding,dilation=self.dilation, groups=self.groups)
        std = t.sqrt(F.conv2d(input ** 2, weight = weight_std, stride=self.stride, 
                          padding=self.padding,dilation=self.dilation, groups=self.groups) + 1e-6)

        eps = Variable(t.randn(*mu.size()))
        if input.is_cuda:
            eps = eps.cuda()
        out = std * eps + mu + self.bias
        
        return out, kld
       