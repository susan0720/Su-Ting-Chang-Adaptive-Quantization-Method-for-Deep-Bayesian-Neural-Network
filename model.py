import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt
from variational_dropout.variational_quantization import VariationalDropout, VariationalDropoutCNN, FlattenLayer
from layers import *

class DenseNet(nn.Module):
    """
    Densely Connected Convolutional Neural Network [1].

    Connects each layer to every other layer in a feed-forward 
    fashion. This alleviates the vanishing-gradient problem, 
    strengthens feature propagation, encourages feature reuse, and 
    substantially reduces the number of parameters.

    Architecture
    ------------
    * Initial Convolution Layer
    * DenseBlock - TransitionLayer (x2)
    * DenseBlock - Global Avg Pooling
    * Fully Connected
    * Softmax
    
    When we say we have a DenseNet of L layers, L is computed as 
    follows:
    - There are 3 Dense blocks, each with n layers.
    - There is an initial conv layer, and final fully-connected layer.
    - There are 2 Transition layers, each with 1 layer.
    Hence, L = 3*n + 2 + 2 = 3*n + 4.

    This is equivalent to saying (L - 4) must be divisible by 3.

    References
    ----------
    - [1]: Huang et. al., https://arxiv.org/abs/1608.06993
    """
    def __init__(self, 
                 num_blocks, 
                 num_layers_total, 
                 growth_rate, 
                 num_classes, 
                 bottleneck, 
                 p, 
                 theta):
        """
        Initialize the DenseNet network. He. et al weight initialization 
        is used (scaling by sqrt(2/n) to make variance 2/n).

        Params
        ------
        - num_blocks: (int) number of dense blocks in the network. On the CIFAR 
          datasets, this is set to 3 while on ImageNet, it's set to 4.
        - num_layers_total: (int) total number of layers L in the network. L must
          follow the following equation: L = 3*n + 4 where n is the number of
          layers in each dense block.
        - growth_rate: (int) this is k in the paper. Number of feature maps produced
          by each convolution in the dense blocks. 
        - num_classes: (int) number of output classes in the dataset.
        - bottleneck: (bool) specifies if the bottleneck variant of the dense block is
          to be used. 
        - p: (float) dropout rate. Used on non-augmented versions of the datasets.
        - theta: (float) compression factor in the range [0, 1]. In the paper, a value
          of 0.5 is used when bottleneck is used.
        """
        super(DenseNet, self).__init__()

        # ensure L relationship talked above 
        error_msg = "[!] Total number of layers must be 3*n + 4..."
        assert (num_layers_total - 4) % 3 == 0, error_msg

        # compute L, the number of layers in each dense block
        # if bottleneck, we need to adjust L by a factor of 2
        num_layers_dense = int((num_layers_total - 4) / 3)
        if bottleneck:
            num_layers_dense = int(num_layers_dense / 2)

        # ================================== #
        # initial convolutional layer
        out_channels = 16
        if bottleneck:
            out_channels = 2 * growth_rate
        self.conv = VariationalDropoutCNN(3,
                              out_channels, 
                              kernel_size=3,
                              padding=1)
        # ================================== #

        # ================================== #
        # dense blocks and transition layers 
        blocks = []
        for i in range(num_blocks - 1):
            # dense block
            dblock = DenseBlock(num_layers_dense, 
                                out_channels, 
                                growth_rate, 
                                bottleneck, 
                                p)
            blocks.append(dblock)

            # transition block
            out_channels = dblock.out_channels
            trans = TransitionLayer(out_channels, theta, p)
            blocks.append(trans)
            out_channels = trans.out_channels
        # ================================== #

        # ================================== #
        # last dense block does not have transition layer
        dblock = DenseBlock(num_layers_dense, 
                            out_channels, 
                            growth_rate, 
                            bottleneck, 
                            p)
        blocks.append(dblock)
        self.block = nn.Sequential(*blocks)
        self.out_channels = dblock.out_channels
        # ================================== #

        # ================================== #
        # fully-connected layer
        self.fc = VariationalDropout(self.out_channels, num_classes)
        # ================================== #

        # ================================== #
        # He et. al weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        # ================================== #

    def forward(self, input, train=False, noquan=False):
        """
        Run the forward pass of the DenseNet model.
        """
        kld = 0
        out, kld_ = self.conv(input, train, noquan)
        kld = kld+ kld_
#         print(out.shape)
        for i, layer in enumerate(self.block):
            
#             print(i,layer)
#             if hasattr(layer,'block') or hasattr(layer,'pool'):
            out, kld = layer(out,kld, train, noquan)
            out = F.relu(out)                
           
#         out, kld_ = self.block(out)#, train, noquan)
#         kld = kld+ kld_
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.out_channels)
        out, kld_ = self.fc(out, train, noquan)
        kld = kld+ kld_
        return out, kld
    def loss(self, **kwargs):
        if kwargs['train']:
            out, kld = self(kwargs['input'], kwargs['train'], kwargs['noquan'])
            return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average']), kld
        out,kld = self(kwargs['input'], kwargs['train'], kwargs['noquan'])
        return out,F.cross_entropy(out, kwargs['target'], size_average=kwargs['average'])