import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
scale = 1.
def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1) :
    """
    An method for calculating KL divergence between two Normal distribtuion.
    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).
   
    """
    kl = log_sigma_1 - log_sigma_0 + (t.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5
    return kl.sum()
class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)
class VariationalDropout(nn.Module):
    def __init__(self, input_size, out_size, log_sigma2=-8, threshold=3):
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
        self.prior_theta = 0.
        self.prior_log_sigma2 = -2.
        self.log_sigma2 = Parameter(t.FloatTensor(input_size, out_size).fill_(log_sigma2))
        self.sz = input_size*out_size
        self.s =  Parameter(t.Tensor([scale]))
        self.code = t.Tensor([0.2,0,-0.2])
                           
        self.reset_parameters()

        self.k = [0.63576, 1.87320, 1.48695]

        self.threshold = threshold

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)

        self.theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def clip(self):
        self.log_sigma2.masked_fill(self.log_sigma2 < -10, -10)
        self.log_sigma2.masked_fill(self.log_sigma2 > 1, 1)
        self.theta.data = t.where(self.theta<(-0.2-0.3679*t.sqrt(self.log_sigma2.exp())), (-0.2-0.3679*t.sqrt(self.log_sigma2.exp())),self.theta)
        self.theta.data = t.where(self.theta>(0.2+0.3679*t.sqrt(self.log_sigma2.exp())), (0.2+0.3679*t.sqrt(self.log_sigma2.exp())),self.theta)
#         self.theta.masked_fill(self.theta < (-0.2-0.3679*t.sqrt(self.log_sigma2.exp())), (-0.2-0.3679*t.sqrt(self.log_sigma2.exp())))
#         self.theta.masked_fill(self.theta > (0.2+0.3679*t.sqrt(self.log_sigma2.exp())), (0.2+0.3679*t.sqrt(self.log_sigma2.exp())))
       
    def clip__(input, to=8):
        input = input.masked_fill(input < -to, -to)
        input = input.masked_fill(input > to, to)

        return input
#     def kllu(self,log_alpha):
#         first_term = self.k[0] * F.sigmoid(self.k[1] + self.k[2] * log_alpha)
#         second_term = 0.5 * t.log(1 + t.exp(-log_alpha))
#         return -(first_term - second_term - self.k[0])
    def kld(self, mean, idx):
                           
        window1 = gaussian_window(mean*self.s,0.2)
        window2 = gaussian_window(mean*self.s,-0.2)
        
        log_alpha1 = self.log_sigma2 + 2*t.log(self.s)- t.log((mean*self.s-0.2) ** 2)
        log_alpha2 = self.log_sigma2 + 2*t.log(self.s)- t.log((mean*self.s) ** 2)
        log_alpha3 = self.log_sigma2 + 2*t.log(self.s)- t.log((mean*self.s+0.2) ** 2)
                           
        F_KLLU1 = kllu(log_alpha1)
        F_KLLU2 = kllu(log_alpha2)
        F_KLLU3 = kllu(log_alpha3)
#         print(F_KLLU1)
#         print(F_KLLU2)
#         print(F_KLLU3)
#         print(hi)
        F_KL = F_KLLU1*window1 + F_KLLU3*window2 + F_KLLU2*(1-window1-window2)
        return F_KL.sum() / (self.sz)
    
    def forward(self, input, train, noquan):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """
        self.clip(self)
        c1 = (self.theta*self.s-0.2)**2
        c2 = (self.theta*self.s)**2
        c3 = (self.theta*self.s+0.2)**2
        mean = t.min(t.min(c1,c2),c3)
        c = t.stack((c1,c2,c3),0)
        idx = t.argmin(c,0)
#         print(idx)
       
        if not train and not noquan:
            """
            mask = log_alpha > self.threshold
            return t.addmm(self.bias, input, self.theta.masked_fill(mask, 0))
            """
            theta_q = self.theta.data.clone()
            theta_q[:] = self.code[idx].cuda()/self.s
#             mask = log_alpha > self.threshold
            
            mu = t.mm(input, theta_q)
            kld = t.sum((theta_q-self.theta)**2)
    
            return mu + self.bias, kld
        if noquan:
            kld = 0
            """
            mask = log_alpha > self.threshold
            return t.addmm(self.bias, input, self.theta.masked_fill(mask, 0))
            """
            theta_q = self.theta.data.clone()
            mu = t.mm(input, theta_q)
            
            return mu + self.bias, kld    
        kld = _kl_loss(self.theta,self.log_sigma2,self.prior_theta,self.prior_log_sigma2)/self.sz
        mu = t.mm(input, self.theta*self.s)
        std = t.sqrt(t.mm(input ** 2, self.s**2*self.log_sigma2.exp()) + 1e-6)

        eps = Variable(t.randn(*mu.size()))
        if input.is_cuda:
            eps = eps.cuda()

        return std * eps + mu + self.bias, kld

    def max_alpha(self):
        log_alpha = self.log_sigma2 - self.theta ** 2
        return t.max(log_alpha.exp())

class VariationalDropoutCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,padding=0, dilation=1, groups=1, log_sigma2=-8, threshold=3):
        """
        :param input_channel: An int of input channel
        :param log_sigma2: Initial value of log sigma ^ 2.
               It is crusial for training since it determines initial value of alpha
        :param threshold: Value for thresholding of validation. If log_alpha > threshold, then weight is zeroed
        :param out_channel: An int of output channel
        """
        super(VariationalDropoutCNN, self).__init__()
#         self.m = img_row
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.theta = Parameter(t.Tensor(out_channel, in_channel // groups, kernel_size, kernel_size))
        self.prior_theta = 0.
        self.prior_log_sigma2 = -2.
#         self.bias = Parameter(t.Tensor(out_channel, in_channel // groups, kernel_size, kernel_size))
#         self.bias = Parameter(t.Tensor(out_channel, self.m-kernel_size+1, self.m-kernel_size+1))
        self.sz = out_channel * (in_channel // groups) * kernel_size**2
        self.log_sigma2 = Parameter(t.FloatTensor(out_channel, in_channel // groups, kernel_size, kernel_size).fill_(log_sigma2))
        self.s = Parameter(t.Tensor([scale]))
        self.code = t.Tensor([0.2,0,-0.2])
                           
        self.reset_parameters()

        self.k = [0.63576, 1.87320, 1.48695]

        self.threshold = threshold

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_channel)

        self.theta.data.uniform_(-stdv, stdv)
#         self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def clip_logsig(input):
        input = input.masked_fill(input < -10, -10)
        input = input.masked_fill(input > 1, 1)

        return input
    def clip(self):
        self.log_sigma2.masked_fill(self.log_sigma2 < -10, -10)
        self.log_sigma2.masked_fill(self.log_sigma2 > 1, 1)
        self.theta.data = t.where(self.theta<(-0.2-0.3679*t.sqrt(self.log_sigma2.exp())), (-0.2-0.3679*t.sqrt(self.log_sigma2.exp())),self.theta)
        self.theta.data = t.where(self.theta>(0.2+0.3679*t.sqrt(self.log_sigma2.exp())), (0.2+0.3679*t.sqrt(self.log_sigma2.exp())),self.theta)
#         self.theta.masked_fill(self.theta < (-0.2-0.3679*t.sqrt(self.log_sigma2.exp())), (-0.2-0.3679*t.sqrt(self.log_sigma2.exp())))
#         self.theta.masked_fill(self.theta > (0.2+0.3679*t.sqrt(self.log_sigma2.exp())), (0.2+0.3679*t.sqrt(self.log_sigma2.exp())))
        

    
    
    def kld(self, idx):
                           
        window1 = gaussian_window(self.theta*self.s,0.2)
        window2 = gaussian_window(self.theta*self.s,-0.2)

        log_alpha1 = self.log_sigma2 + 2*t.log(self.s)- t.log((self.theta*self.s-0.2) ** 2)
        log_alpha2 = self.log_sigma2 + 2*t.log(self.s)- t.log((self.theta*self.s) ** 2)
        log_alpha3 = self.log_sigma2 + 2*t.log(self.s)- t.log((self.theta*self.s+0.2) ** 2)
                           
        F_KLLU1 = kllu(log_alpha1)
        F_KLLU2 = kllu(log_alpha2)
        F_KLLU3 = kllu(log_alpha3)
        F_KL = F_KLLU1*window1 + F_KLLU3*window2 + F_KLLU2*(1-window1-window2)
        return F_KL.sum() / (self.sz)

    def forward(self, input, train,noquan):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """
        self.clip()
        c1 = (self.theta*self.s-0.2)**2
        c2 = (self.theta*self.s)**2
        c3 = (self.theta*self.s+0.2)**2
        mean = t.min(t.min(c1,c2),c3)
        c = t.stack((c1,c2,c3),0)
        idx = t.argmin(c,0)
        
        
        if not train and not noquan:
            """
            mask = log_alpha > self.threshold
            return F.conv2d( input, weight = self.theta.masked_fill(mask, 0), stride=self.stride, 
                          padding=self.padding,dilation=self.dilation, groups=self.groups)
            """
            
            theta_q = self.theta.data.clone()
            
            theta_q[:] = self.code[idx].cuda()/self.s
            mu = F.conv2d(input, weight = theta_q, stride=self.stride, 
                          padding=self.padding,dilation=self.dilation, groups=self.groups)
            
            kld = t.sum((theta_q-self.theta)**2)
            return mu,kld#+self.bias , kld
        if noquan:
            kld=0
            theta_q = self.theta.data.clone()
            mu = F.conv2d(input, weight = theta_q, stride=self.stride, 
                          padding=self.padding,dilation=self.dilation, groups=self.groups)
            
    
            return mu,kld#+self.bias , kld
        kld = _kl_loss(self.theta,self.log_sigma2, self.prior_theta, self.prior_log_sigma2)/self.sz
        mu = F.conv2d(input, weight = self.theta*self.s, stride=self.stride, 
                          padding=self.padding,dilation=self.dilation, groups=self.groups)
        std = t.sqrt(F.conv2d(input ** 2, weight = self.log_sigma2.exp()*self.s**2, stride=self.stride, 
                          padding=self.padding,dilation=self.dilation, groups=self.groups) + 1e-6)

        eps = Variable(t.randn(*mu.size()))
        if input.is_cuda:
            eps = eps.cuda()
        return std * eps + mu,kld# + self.bias , kld

    def max_alpha(self):
        log_alpha = self.log_sigma2 - (self.theta-0.2) ** 2
        return t.max(log_alpha.exp())