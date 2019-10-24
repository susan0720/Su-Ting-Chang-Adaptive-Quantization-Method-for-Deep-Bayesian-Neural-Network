import torch.nn as nn
import torch.nn.functional as F

from variational_dropout.variational_nonsym import VariationalDropout, VariationalDropoutCNN,FlattenLayer


class VariationalDropoutLeNet(nn.Module):
    def __init__(self):
        super(VariationalDropoutLeNet, self).__init__()

        self.layers = nn.ModuleList([
#                 VariationalDropoutCNN(28,1,6, 5, stride=1)
#                 ,nn.BatchNorm2d(6)
#                 ,nn.ReLU()
#                 ,nn.MaxPool2d(kernel_size=2, stride=2)
#                 ,VariationalDropoutCNN(12,6, 16, 5, stride=1)
#                 ,nn.BatchNorm2d(16)
#                 ,nn.ReLU()
#                 ,nn.MaxPool2d(kernel_size=2, stride=2)
#                 ,FlattenLayer(4 * 4 * 16)
#                 ,VariationalDropout(4 * 4 * 16, 120)
#                 ,nn.BatchNorm1d(120)
#                 ,nn.ReLU()
#                 ,VariationalDropout(120, 84)
#                 ,nn.BatchNorm1d(84)
#                 ,nn.ReLU()
#                 ,VariationalDropout(84, 10)
            VariationalDropoutCNN(32,3,6, 5, stride=1)
            ,nn.BatchNorm2d(6)
            ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2)
            ,VariationalDropoutCNN(14,6, 16, 5, stride=1)
            ,nn.BatchNorm2d(16)
            ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2)
            ,FlattenLayer(5 * 5 * 16)
            ,VariationalDropout(5 * 5 * 16, 120)
            ,nn.BatchNorm1d(120)
            ,nn.ReLU()
            ,VariationalDropout(120, 84)
            ,nn.BatchNorm1d(84)
            ,nn.ReLU()
            ,VariationalDropout(84, 10)
        ])

    def forward(self, input, train=False):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :param train: An boolean value indicating whether forward propagation called when training is performed
        :return: An float tensor with shape of [batch_size, 10]
                 filled with logits of likelihood and kld estimation
        """

        result = input
        if train:
            kld = 0

            for i, layer in enumerate(self.layers):
                if hasattr(layer,'kld'):
                    result, kld = layer(result, train)
                    result = F.sigmoid(result)
                    kld += kld
                else:
                    result = layer(result)

            return result, kld

        for i, layer in enumerate(self.layers):
            if hasattr(layer,'kld'):
                result, kld = layer(result, train)
                result = F.sigmoid(result)
            else:
                result = layer(result)
        return result

    def loss(self, **kwargs):
        if kwargs['train']:
            out, kld = self(kwargs['input'], kwargs['train'])
            return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average']), kld
        out = self(kwargs['input'], kwargs['train'])
        return out,F.cross_entropy(out, kwargs['target'], size_average=kwargs['average'])
