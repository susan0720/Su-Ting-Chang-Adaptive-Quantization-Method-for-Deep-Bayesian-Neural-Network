import argparse
import os
import torch as t
import torch.nn as nn
import math
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets
from models.variational_dropout_lenet_nonsym import VariationalDropoutLeNet
from models import *
import matplotlib.pyplot as plt
import numpy as np
t.backends.cudnn.enabled=False
val = float('nan')
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--num-epochs', type=int, default=195, metavar='NI',
                        help='num epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                        help='batch size (default: 70)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--mode', type=str, default='vardropout', metavar='M',
                        help='training mode (default: simple)')
    args = parser.parse_args()

#     writer = SummaryWriter(args.mode)

    assert args.mode in ['simple', 'dropout', 'vardropout'], 'Invalid mode, should be in [simple, dropout, vardropout]'
    Model = {
        'simple': SimpleModel,
        'dropout': DropoutModel,
        'vardropout': VariationalDropoutLeNet
    }
    Model = Model[args.mode]
    dataset = datasets.CIFAR10(root='../data',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=True)
    train_dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    dataset = datasets.CIFAR10(root='../data',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=False)
    test_dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = Model()
    print(model)
    
    if args.use_cuda:
        model.cuda()
        
    m=[]
    m.append(model.layers[0].s1)
    m.append(model.layers[4].s1)
    m.append(model.layers[9].s1)
    m.append(model.layers[12].s1)
    m.append(model.layers[15].s1)
    m.append(model.layers[0].s2)
    m.append(model.layers[4].s2)
    m.append(model.layers[9].s2)
    m.append(model.layers[12].s2)
    m.append(model.layers[15].s2)
    new_params=[]
    for p in model.parameters():
        if len(p)!=1:
            new_params.append(p)
    optimizer = Adam([
                {'params': new_params},
                {'params': m, 'lr': args.learning_rate*0.01}
                
            ], args.learning_rate, eps=1e-6)
    
    cross_enropy_averaged = nn.CrossEntropyLoss(size_average=True)
    best_acc = 0.
    coef = 0
    for epoch in range(args.num_epochs):
        
        for iteration, (input, target) in enumerate(train_dataloader):
            
            input = Variable(input)
            target = Variable(target)

            if args.use_cuda:
                input, target = input.cuda(), target.cuda()

            optimizer.zero_grad()

            loss = None
            if args.mode == 'simple':
                loss = model.loss(input=input, target=target, average=True)
            elif args.mode == 'dropout':
                loss = model.loss(input=input, target=target, p=0.4, average=True)
            else:
                likelihood, kld = model.loss(input=input, target=target, train=True, average=True)
#                 coef = min((1+epoch) / 40., 1.)
                
                loss = likelihood + kld * coef
                
            if math.isnan(loss):
                print('train epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()))
                print(likelihood , kld )
                l1=model.layers[0].theta.data.cpu().view(1,-1).numpy()
                l2=model.layers[4].theta.data.cpu().view(1,-1).numpy()
                l3=model.layers[9].theta.data.cpu().view(1,-1).numpy()
                l4=model.layers[12].theta.data.cpu().view(1,-1).numpy()
                l5=model.layers[15].theta.data.cpu().view(1,-1).numpy()
                s1=model.layers[0].log_sigma2.data.cpu().view(1,-1).numpy()
                s2=model.layers[4].log_sigma2.data.cpu().view(1,-1).numpy()
                s3=model.layers[9].log_sigma2.data.cpu().view(1,-1).numpy()
                s4=model.layers[12].log_sigma2.data.cpu().view(1,-1).numpy()
                s5=model.layers[15].log_sigma2.data.cpu().view(1,-1).numpy()
                scale1=model.layers[0].s.data.cpu().numpy()
                scale2=model.layers[4].s.data.cpu().numpy()
                scale3=model.layers[9].s.data.cpu().numpy()
                scale4=model.layers[12].s.data.cpu().numpy()
                scale5=model.layers[15].s.data.cpu().numpy()
                np.savez('../lenet_cifar10_nan.npz',mean1 = l1,mean2 = l2,mean3 = l3,mean4 = l4,mean5 = l5
                     ,logsig1 = s1,logsig2 = s2,logsig3 = s3,logsig4 = s4,logsig5 = s5
                     ,scale1=scale1
                     ,scale2=scale2
                     ,scale3=scale3
                     ,scale4=scale4
                     ,scale5=scale5)
                print(hi)
            loss.backward()
#             print(iteration)
            for p in model.parameters():
                p.grad.data.clamp_(-1,1)
                _grad = t.isnan(p.grad.data)  # -inf的位置为1
#                 if t.sum(_grad)!=0:
#                     print(epoch, iteration,p)
                p.grad.data = p.grad.data.masked_fill_(_grad,0) # 将-inf填充为0
#                 print(t.max(p.grad.data))
            optimizer.step()

            if iteration % 30 == 0:
                scale1=model.layers[0].s1.data.cpu().numpy()
                scale2=model.layers[4].s1.data.cpu().numpy()
                scale3=model.layers[9].s1.data.cpu().numpy()
                scale4=model.layers[12].s1.data.cpu().numpy()
                scale5=model.layers[15].s1.data.cpu().numpy()
                scale1_2=model.layers[0].s2.data.cpu().numpy()
                scale2_2=model.layers[4].s2.data.cpu().numpy()
                scale3_2=model.layers[9].s2.data.cpu().numpy()
                scale4_2=model.layers[12].s2.data.cpu().numpy()
                scale5_2=model.layers[15].s2.data.cpu().numpy()
                print('train epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()))
                print(scale1
                        ,scale2
                        ,scale3
                        ,scale4
                        ,scale5 )
                print(scale1_2
                          ,scale2_2
                          ,scale3_2
                          ,scale4_2
                          ,scale5_2)

            
        loss = 0
        total = 0
        correct = 0
        for input, target in test_dataloader:
            input = Variable(input)#.view(-1, 784)
            target = Variable(target)
            if args.use_cuda:
                input, target = input.cuda(), target.cuda() 
            if args.mode == 'simple':
                loss += model.loss(input=input, target=target, average=False).cpu().data.numpy()
            elif args.mode == 'dropout':
                loss += model.loss(input=input, target=target, p=0., average=False).cpu().data.numpy()
            else:
                outputs, loss = model.loss(input=input, target=target, train=False, average=False)
                loss += loss.data
            _, predicted = t.max(outputs.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
        acc =  correct.data.numpy()/total *100.0
        print(acc,correct,total)
        print('valid epoch {}, loss {}, acc {}%'.format(epoch, loss,acc))
        print('-----------------------------------------')
        if epoch<15:
            coef+=1/15
        if epoch>=50:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']-(0.001/200)
            optimizer.param_groups[1]['lr'] =optimizer.param_groups[0]['lr']*0.01 
        
        if acc>=best_acc :
            l1=model.layers[0].theta.data.cpu().view(1,-1).numpy()
            l2=model.layers[4].theta.data.cpu().view(1,-1).numpy()
            l3=model.layers[9].theta.data.cpu().view(1,-1).numpy()
            l4=model.layers[12].theta.data.cpu().view(1,-1).numpy()
            l5=model.layers[15].theta.data.cpu().view(1,-1).numpy()
            s1=model.layers[0].log_sigma2.data.cpu().view(1,-1).numpy()
            s2=model.layers[4].log_sigma2.data.cpu().view(1,-1).numpy()
            s3=model.layers[9].log_sigma2.data.cpu().view(1,-1).numpy()
            s4=model.layers[12].log_sigma2.data.cpu().view(1,-1).numpy()
            s5=model.layers[15].log_sigma2.data.cpu().view(1,-1).numpy()
            scale1=model.layers[0].s1.data.cpu().numpy()
            scale2=model.layers[4].s1.data.cpu().numpy()
            scale3=model.layers[9].s1.data.cpu().numpy()
            scale4=model.layers[12].s1.data.cpu().numpy()
            scale5=model.layers[15].s1.data.cpu().numpy()
            scale1_2=model.layers[0].s2.data.cpu().numpy()
            scale2_2=model.layers[4].s2.data.cpu().numpy()
            scale3_2=model.layers[9].s2.data.cpu().numpy()
            scale4_2=model.layers[12].s2.data.cpu().numpy()
            scale5_2=model.layers[15].s2.data.cpu().numpy()
            np.savez('../lenet_cifar10_nonsym.npz',mean1 = l1,mean2 = l2,mean3 = l3,mean4 = l4,mean5 = l5
                 ,logsig1 = s1,logsig2 = s2,logsig3 = s3,logsig4 = s4,logsig5 = s5
                 ,scale1=scale1
                 ,scale2=scale2
                 ,scale3=scale3
                 ,scale4=scale4
                 ,scale5=scale5
                 ,scale1_2=scale1_2
                 ,scale2_2=scale2_2
                 ,scale3_2=scale3_2
                 ,scale4_2=scale4_2
                 ,scale5_2=scale5_2
                     ,acc=acc)
            best_acc = acc

    l1=model.layers[0].theta.data.cpu().view(1,-1).numpy()
    l2=model.layers[4].theta.data.cpu().view(1,-1).numpy()
    l3=model.layers[9].theta.data.cpu().view(1,-1).numpy()
    l4=model.layers[12].theta.data.cpu().view(1,-1).numpy()
    l5=model.layers[15].theta.data.cpu().view(1,-1).numpy()
    s1=model.layers[0].log_sigma2.data.cpu().view(1,-1).numpy()
    s2=model.layers[4].log_sigma2.data.cpu().view(1,-1).numpy()
    s3=model.layers[9].log_sigma2.data.cpu().view(1,-1).numpy()
    s4=model.layers[12].log_sigma2.data.cpu().view(1,-1).numpy()
    s5=model.layers[15].log_sigma2.data.cpu().view(1,-1).numpy()
    scale1=model.layers[0].s1.data.cpu().numpy()
    scale2=model.layers[4].s1.data.cpu().numpy()
    scale3=model.layers[9].s1.data.cpu().numpy()
    scale4=model.layers[12].s1.data.cpu().numpy()
    scale5=model.layers[15].s1.data.cpu().numpy()
    scale1_2=model.layers[0].s2.data.cpu().numpy()
    scale2_2=model.layers[4].s2.data.cpu().numpy()
    scale3_2=model.layers[9].s2.data.cpu().numpy()
    scale4_2=model.layers[12].s2.data.cpu().numpy()
    scale5_2=model.layers[15].s2.data.cpu().numpy()
    np.savez('../lenet_cifar10_epoch200.npz',mean1 = l1,mean2 = l2,mean3 = l3,mean4 = l4,mean5 = l5
         ,logsig1 = s1,logsig2 = s2,logsig3 = s3,logsig4 = s4,logsig5 = s5
         ,scale1=scale1
         ,scale2=scale2
         ,scale3=scale3
         ,scale4=scale4
         ,scale5=scale5
         ,scale1_2=scale1_2
         ,scale2_2=scale2_2
         ,scale3_2=scale3_2
         ,scale4_2=scale4_2
         ,scale5_2=scale5_2
             ,acc=acc)
