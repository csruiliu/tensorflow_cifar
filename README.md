# Train CNN models with CIFAR10 on TensorFlow #

## ResNet ##

The model follows the architecture of the original [paper](https://arxiv.org/abs/1512.03385). To fit the CIFAR-10, I slightly modify the kernel size of `conv_1` from 7x7 to 3x3 and omit the first 3x3 max pooling layer.  