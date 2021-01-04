# Train CNN models with CIFAR10 on TensorFlow #

All the models are tested on TensorFlow 1.15 and should be compatible with TF 1.x

## AlexNet ##

The implementation of AlexNet follows the architecture proposed in the [paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). To fit the CIFAR-10, I modify the filters and kernel size of conv layer, for example, the filters and kernel of the first conv layer is changed to 64 and 3.

## DenseNet ##

The implementations of DenseNet [121, 169, 201, 264] follow the architecture proposed in the [paper](https://arxiv.org/pdf/1608.06993.pdf). To fit the CIFAR-10, I slightly modify the kernel size of the front convolutional layer from 7x7 to 3x3, and omit the first 3x3 max pooling layer. 

## GoogLeNet ## 

The implementation of GoogLeNet follows the architecture presented in the [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf). I remove the front 7x7 conv layer and 3x3 max pooling layer to fit CIFAR-10. 

## LeNet ## 

The implementation of LeNet follows the architecture presented in the [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). It is a simple but practical model. 

## MobileNet ##

The implementations of MobileNet basically follows the architecture proposed in the [paper](https://arxiv.org/pdf/1704.04861.pdf). I slightly modify the strides of the front conv2d layer to 1 and remove the avgpool 7x7 layer. 

## MobileNet V2 ##

The implementations of MobileNetV2 basically follow the architecture proposed in the [paper](https://arxiv.org/pdf/1801.04381.pdf). To fit the CIFAR-10, I slightly modify the strides of the front conv2d layer to 1, the strides of the second bottleneck to 1. The avgpool 7x7 has been omitted. 

## ResNet ##

The implementations of ResNet [18, 34, 50, 101, 152] follow the proposed architecture of the original [paper](https://arxiv.org/abs/1512.03385). To fit the CIFAR-10, I slightly modify the kernel size of `conv_1` from 7x7 to 3x3 and omit the first 3x3 max pooling layer. 

## VGG ##

The imlementations of VGG [11, 13, 16, 19] follow the architecture proposed in the [paper](https://arxiv.org/pdf/1409.1556.pdf).


