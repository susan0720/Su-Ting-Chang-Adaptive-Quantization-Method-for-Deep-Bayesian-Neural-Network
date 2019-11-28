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
from model import DenseNet
from models import *
import matplotlib.pyplot as plt
import numpy as np
t.backends.cudnn.enabled=False
val = float('nan')
t.cuda.set_device(0)
def str2bool(v):
    return v.lower() in ('true', '1')
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--num-epochs', type=int, default=195, metavar='NI',
                        help='num epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 70)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--mode', type=str, default='vardropout', metavar='M',
                        help='training mode (default: simple)')
    # network params
    parser.add_argument('--num_blocks', type=int, default=3,
                            help='# of Dense blocks to use in the network')
    parser.add_argument('--num_layers_total', type=int, default=40,
                     help='Total # of layers in the network')
    parser.add_argument('--growth_rate', type=int, default=12,
                     help='Growth rate (k) of the network')
    parser.add_argument('--bottleneck', type=str2bool, default=False,
                     help='Whether to use bottleneck layers')
    parser.add_argument('--compression', type=float, default=1.0,
                        help='Compression factor theta in the range [0, 1]')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                            help='Dropout rate used with non-augmented datasets')
    args = parser.parse_args()

#     writer = SummaryWriter(args.mode)

    assert args.mode in ['simple', 'dropout', 'vardropout'], 'Invalid mode, should be in [simple, dropout, vardropout]'
    Model = {
        'simple': SimpleModel,
        'dropout': DropoutModel,
        'vardropout': DenseNet
    }
    Model = Model[args.mode]
    dataset = datasets.CIFAR10(root='../data',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=True)
    train_dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,num_workers=4,                           pin_memory=True)

    dataset = datasets.CIFAR10(root='../data',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=False)
    test_dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4,                           pin_memory=True)

    model = Model(args.num_blocks, 
                 args.num_layers_total, 
                 args.growth_rate, 
                 100,#.num_classes, 
                 args.bottleneck, 
                 args.dropout_rate, 1.0)#args.compression)
    print(model)
#     print(h)
    if args.use_cuda:
        model.cuda()
        
#     m=[]
#     m.append(model.layers[0].s)
#     m.append(model.layers[2].s)
#     m.append(model.layers[5].s)
#     m.append(model.layers[6].s)
# #     m.append(model.layers[15].s)
#     new_params=[]
#     for p in model.parameters():
#         if len(p)!=1:
#             new_params.append(p)
            
    optimizer = Adam(model.parameters() , args.learning_rate)#, eps=1e-6)
#     optimizer = Adam([{'params': new_params},{'params': m, 'lr': args.learning_rate*0.01}], lr=args.learning_rate)
        
    
    cross_enropy_averaged = nn.CrossEntropyLoss(size_average=True)
    best_acc = 0
    coef = 0
    for epoch in range(args.num_epochs):
        if epoch >=50 :
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']-(0.001/100)
#         optimizer.param_groups[1]['lr'] =optimizer.param_groups[0]['lr']*0.01 
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
                likelihood, kld = model.loss(input=input, target=target, train=True, average=True, noquan=False)
#                 coef = min((1+epoch) / 40., 1.)
                loss = likelihood + kld * coef
                
            
            loss.backward()
#             for num,p in enumerate(model.parameters()):
#                 print(num,p.shape)
#                 if num ==5:
#                     print(h)
            for num,p in enumerate(model.parameters()):
#                 print(p,p.grad.shape)
#                 print(num)
                p.grad.data.clamp_(-1,1)
                _grad = t.isnan(p.grad.data)  # -inf的位置为1
#                 if t.sum(_grad)!=0:
#                     print(epoch, iteration,p)
                p.grad.data = p.grad.data.masked_fill_(_grad,0)
            
            optimizer.step()

            if iteration % 100 == 0:
#                 scale1=model.layers[0].s.data.cpu().numpy()
#                 scale2=model.layers[2].s.data.cpu().numpy()
#                 scale3=model.layers[5].s.data.cpu().numpy()
#                 scale4=model.layers[6].s.data.cpu().numpy()
#                 scale5=model.layers[15].s.data.cpu().numpy()
                print('train epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()))
#                 print(scale1
#                         ,scale2
#                         ,scale3
#                         ,scale4
# #                         ,scale5
#                      )

            
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
                outputs, loss = model.loss(input=input, target=target, train=False, average=False, noquan=False)
                loss += loss.data
            _, predicted = t.max(outputs.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
        acc =  100.*correct/total
        print('valid epoch {}, loss {}, acc {}%, correct {}, total {}'.format(epoch, loss,acc,correct,total))
        print('-----------------------------------------')
        if epoch<15:
            coef+=1/15
        if correct>=best_acc :
            state={
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_valid_acc': correct}

            t.save(state, '../checkpoint/densenet_'+str(args.num_layers_total)+'_CIFAR10.t7')
            best_acc = correct
    l1=model.layers[0].theta.data.cpu().view(1,-1).numpy()
    l2=model.layers[2].theta.data.cpu().view(1,-1).numpy()
    l3=model.layers[5].theta.data.cpu().view(1,-1).numpy()
    l4=model.layers[6].theta.data.cpu().view(1,-1).numpy()
#     l5=model.layers[15].theta.data.cpu().view(1,-1).numpy()
    s1=model.layers[0].log_sigma2.data.cpu().view(1,-1).numpy()
    s2=model.layers[2].log_sigma2.data.cpu().view(1,-1).numpy()
    s3=model.layers[5].log_sigma2.data.cpu().view(1,-1).numpy()
    s4=model.layers[6].log_sigma2.data.cpu().view(1,-1).numpy()
#     s5=model.layers[15].log_sigma2.data.cpu().view(1,-1).numpy()
    scale1=model.layers[0].s.data.cpu().numpy()
    scale2=model.layers[2].s.data.cpu().numpy()
    scale3=model.layers[5].s.data.cpu().numpy()
    scale4=model.layers[6].s.data.cpu().numpy()
#     scale5=model.layers[15].s.data.cpu().numpy()
    np.savez('../lenet_mnist_epoch200.npz',mean1 = l1,mean2 = l2,mean3 = l3,mean4 = l4
#              ,mean5 = l5
         ,logsig1 = s1,logsig2 = s2,logsig3 = s3,logsig4 = s4
#              ,logsig5 = s5
         ,scale1=scale1
         ,scale2=scale2
         ,scale3=scale3
         ,scale4=scale4
#          ,scale5=scale5
             ,acc=correct)
    #         if acc > best_acc:
#             print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
#             state = {
#                 'net':model.cuda(),
#                 'acc':acc,
#                 'epoch':epoch,
#             }
# #             if not os.path.isdir('checkpoint'):
# #                 os.mkdir('checkpoint')
#             save_point = './checkpoint/'
#             if not os.path.isdir(save_point):
#                 os.mkdir(save_point)
#             t.save(state, save_point+'mnist_lenet.t7')
#             best_acc = acc
#     writer.close()
