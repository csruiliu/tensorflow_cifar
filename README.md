# Train CNN models with CIFAR10 on TensorFlow #

All the models are tested on TensorFlow 1.15 and should be compatible with TF 1.x

## AlexNet ##

The implementation of AlexNet follows the architecture proposed in the [paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). The current hardware offering is powerful enough, so the original two GPUs version's AlexNet is usually implemented in a unified fashion. ***AlexNet won the ImageNet 2012 contest (Classification and Localization), it is the first time to apply CNN to a large scale dataset, and the eye-catching results significantly promoted the development of CNN***

## DenseNet ##

The implementations of DenseNet [121, 169, 201, 264] follow the architecture proposed in the [paper](https://arxiv.org/pdf/1608.06993.pdf). To fit the CIFAR-10, I slightly modify the kernel size of the front convolutional layer from 7x7 to 3x3, and omit the first 3x3 max pooling layer. ***DenseNet paper won the CVPR 2017 best paper. Its idea is to strengthen the feature reuse (any two specific layers can communicate through a shortcut).***

## EfficientNet ##

The implementations of EfficientNet follows the architecture of EfficientNetB0 proposed in [paper](https://arxiv.org/pdf/1905.11946.pdf). ***It proposed a compound scaling method that uses a compound coefficient φ to uniformly scales network width, depth, and resolution in a principled way. Essentially, EfficientNet attempted to find a sweet spot in the tradeoff between accuracy and model size***. 

## Inception ## 

The implementation of Inception (aka GoogLeNet, Inception-V1) follows the architecture presented in the [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf). I remove the front 7x7 conv layer and 3x3 max pooling layer to fit CIFAR-10. ***GoogLeNet won the ImageNet 2014 contest (Classification and Detection). Its inception unit consists of parallel several conv layers, which can reduce the model size without hurting accuracy. It shows that multiple small conv kernels can achieve similar performance to a big conv kernel in a way.***

## LeNet ## 

The implementation of LeNet (aka LeNet-5) follows the architecture presented in the [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). It is a simple but practical model. ***It was the original of the modern CNNs in a way and promoted the development of deep learning.***

## MobileNet ##

The implementations of MobileNet basically follows the architecture proposed in the [paper](https://arxiv.org/pdf/1704.04861.pdf). I slightly modify the strides of the front conv2d layer to 1 and remove the avgpool 7x7 layer. ***MobileNet is designed for mobile applications so its size has to be small enough. The key idea in MobileNet is the depthwise separable convolution.***

## MobileNet V2 ##

The implementations of MobileNetV2 basically follow the architecture proposed in the [paper](https://arxiv.org/pdf/1801.04381.pdf). To fit the CIFAR-10, I slightly modify the strides of the front conv2d layer to 1, the strides of the second bottleneck to 1. The avgpool 7x7 has been omitted. ***The successor of MobileNet, the key contribution of V2 is the Inverted Residual Block.***

## ResNet ##

The implementations of ResNet [18, 34, 50, 101, 152] follow the proposed architecture of the original [paper](https://arxiv.org/abs/1512.03385). To fit the CIFAR-10, I slightly modify the kernel size of `conv_1` from 7x7 to 3x3 and omit the first 3x3 max pooling layer. ***ResNet is the champion of the ImageNet 2015 (Classification and Localization), which first introduced residual block with shortcut connection to increase the model depth.***

## ResNeXt ##

The implementations of ResNeXt follow the proposed architecture of the original [paper](https://arxiv.org/pdf/1611.05431.pdf). Specifically, I follow the original implemention of ResNeXt on CIFAR-10, ResNeXt-29. ***ResNeXt is the runner-up of the ImageNet 2016 (Classification). It takes advantage of grouped convolution and abstracted a pattern of split-transform-merge.***

## ShuffleNet ##

The imlementations of ShuffleNet follows the architecture proposed in the [paper](https://arxiv.org/pdf/1707.01083.pdf), but omits the max pooling layer. ***ShuffleNet is designed for mobile devices with very limited computing power (like MobileNet). The key ideas are pointwise group convolu and channel shuffle. The former can reduce computation complexity of 1x1 convolutions (a costly operation in CNN), the latter allows each group conv gets input from the other groups so that the information can flow across feature channels.*** 

## ShuffleNet V2 ##

The imlementations of ShuffleNet V2 follows the architecture proposed in the [paper](https://arxiv.org/pdf/1807.11164.pdf), but omits the max pooling layer. ***ShuffleNet V2 claimed that we should use the direct metric (actual runtime on target platform) rather than the indirect metric to compute model complexities. It also proposed 4 principles for designing models. G1. Equal channel width minimizes memory access cost; G2. Excessive group convolution increases MAC; G3. Network fragmentation reduces degree of parallelism; G4. Element-wise operations (ReLu, Add, etc) are non-negligible.*** 

## SqueezeNet ##

The imlementations of SqueezeNet follows the architecture proposed in the [paper](https://arxiv.org/pdf/1602.07360.pdf). I slightly change the strides of the first conv layer to 2. ***Many CNN models focused on improving accuracy, SqueezeNet was proposed to achieve an acceptable accuracy but use a small number of parameters. So, it took the size side of a trade-off between accuracy and size.***

## VGG ##

The imlementations of VGG [11, 13, 16, 19] follow the architecture proposed in the [paper](https://arxiv.org/pdf/1409.1556.pdf). ***VGG is the champion of the ImageNet 2014 (Localization) and a runner-up of the ImageNet 2014 (Classification). It is a simple but practical deep learning model, and its results showed that a deeper model could achieve better performance and illustrated the importance of a small conv kernel.***

## Xception ##

The imlementation of Xception basically follows the architecture proposed in the [paper](https://arxiv.org/pdf/1610.02357.pdf). However, to fit CIFAR-10, I remove the front conv layer and all max pooling layer, and change the stride of shortcut layers to 1x1. ***Xception is similar to MobileNet and promotes the development of Depthwise Conv + Pointwise Conv.***

## ZFNet ##

The imlementation of ZFNet follows the architecture proposed in the [paper](https://arxiv.org/pdf/1311.2901.pdf), but omits Layer 4 and maxpooling layer after Layer 5. ZFNet had the same architecture as AlexNet but used smaller conv kernels and strides for the first two conv layers. ***ZFNet is the champion of the ImageNet 2013. The key contribution is to visualize the model.*** 