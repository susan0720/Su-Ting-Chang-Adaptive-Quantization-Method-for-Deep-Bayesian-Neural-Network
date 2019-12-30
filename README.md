# Adaptive Quantization Method for Deep Bayesian Neural Network
## Introduction
Quantizaiton on weight is part of model compression issue. 
This paper proposes a $M-ary$ adaptive quantization method for deep Bayesian neural networks. 
That is to use deterministic values to represent stochastic weights of the model. 
To reduce the performance loss, the representation value that maximize the likelihood is chosen because it keeps the probabilistic property after quantization. 
Besides, the quantization scheme is decided data-driven, which explores in larger parameter space than hand-crafted models. 
This benefits the model to achieve better results in classification. 
The adaptive quantization method can be generalized for M levels, decided before training. 
This method improve the quantization for Bayesian neural network, and can be applied on different network structure, making it possible to utilize complex models on mobile device. 
## Result
After the training stage, the means of weights densely cluster at three locations in M=3 case. The histogram showes that the distribution over weights is "multiple spike-and-slab". Also, the locations of spikes are asymmetric about zero.
![image](https://github.com/susan0720/Su-Ting-Chang-Adaptive-Quantization-Method-for-Deep-Bayesian-Neural-Network/blob/master/densenet-M3.png)


The quantized loss is the difference between quantized weights and full-precision weights, and it converges fastly.
![image](https://github.com/susan0720/Su-Ting-Chang-Adaptive-Quantization-Method-for-Deep-Bayesian-Neural-Network/blob/master/comparison.png)
## File description
"train_densenet.py" is the main file that runs adaptive quantization method on deep Bayesian DenseNet with CIFAR10.

"train_nonsymLeNet.py" is the main file that runs adaptive quantization method on deep LeNet with MNIST.

"layers.py" is the module of Bayesian convolutional layer and fully connected layer.
## Setting
* Hardware:
  * CPU: Intel Core i7-4930K @3.40 GHz
  * RAM: 64 GB DDR3-1600
  * GPU: GeForce GTX 1080ti
* pytorch 
* Dataset
  * MNIST
  * CIFAR10
  * CIFAR100
