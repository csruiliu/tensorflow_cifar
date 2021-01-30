import tensorflow as tf
import argparse
import numpy as np

from models.resnet import ResNet
from models.densenet import DenseNet
from models.mobilenet_v2 import MobileNetV2
from models.mobilenet import MobileNet
from models.vgg import VGG
from models.lenet import LeNet
from models.inception import Inception
from models.alexnet import AlexNet
from models.resnext import ResNeXt
from models.xception import Xception
from models.squeezenet import SqueezeNet
from models.zfnet import ZFNet
from models.efficientnet import EfficientNet
from models.shufflenet import ShuffleNet
from models.shufflenet_v2 import ShuffleNetV2

from tools.dataset_loader import load_cifar10_keras
from tools.model_tools import train_model
from tools.model_tools import evaluate_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batchsize', action='store', type=int,
                        help='set the training batch size')
    parser.add_argument('-o', '--opt', action='store', type=str,
                        choices=['Adam', 'SGD', 'Adagrad', 'Momentum'],
                        help='set the training optimizer')
    parser.add_argument('-r', '--lr', action='store', type=float,
                        help='set the learning rate')
    parser.add_argument('-e', '--epoch', action='store', type=int,
                        help='set the number of training epoch')

    args = parser.parse_args()
    batch_size = args.batchsize
    opt = args.opt
    lr = args.lr
    num_epoch = args.epoch

    # load CNN model
    # model = ResNet(residual_layer=18, num_classes=10)
    # model = DenseNet(residual_layer=121, num_classes=10)
    # model = MobileNetV2(num_classes=10)
    # model = MobileNet(num_classes=10)
    # model = VGG(conv_layer=16, num_classes=10)
    # model = LeNet(num_classes=10)
    # model = Inception(num_classes=10)
    # model = AlexNet(num_classes=10)
    # model = ZFNet(num_classes=10)
    # model = ResNeXt(cardinality=8, num_classes=10)
    # model = Xception(num_classes=10)
    # model = SqueezeNet(num_classes=10)
    # model = EfficientNet(num_classes=10)
    # model = ShuffleNet(num_groups=2, num_classes=10)
    model = ShuffleNetV2(complexity=1, num_classes=10)

    feature_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    label_ph = tf.placeholder(tf.int32, [None, 10])

    logit = model.build(feature_ph)
    train_op = train_model(logit, label_ph, opt, lr)
    eval_op = evaluate_model(logit, label_ph)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # load cifar10 dataset
    train_feature, train_label, eval_feature, eval_label = load_cifar10_keras()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_feature = train_label.shape[0]
        num_batch = num_feature // batch_size
        rest_feature = num_feature - batch_size * num_batch

        # train the model
        for e in range(num_epoch):
            # shuffle the training data
            shf_indices = np.arange(num_feature)
            np.random.shuffle(shf_indices)
            train_feature = train_feature[shf_indices]
            train_label = train_label[shf_indices]

            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))

                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size
                train_feature_batch = train_feature[batch_offset:batch_end]
                train_label_batch = train_label[batch_offset:batch_end]
                sess.run(train_op, feed_dict={feature_ph: train_feature_batch, label_ph: train_label_batch})

            if rest_feature != 0:
                print('the rest train feature: {}, train them now'.format(rest_feature))
                rest_feature_batch = train_feature[-rest_feature:]
                rest_label_batch = train_label[-rest_feature:]
                sess.run(train_op, feed_dict={feature_ph: rest_feature_batch, label_ph: rest_label_batch})
            else:
                print('no train feature left for this epoch')

        print('start evaluation phrase')
        acc_sum = 0
        eval_batch_size = 50
        num_batch_eval = eval_label.shape[0] // eval_batch_size
        for i in range(num_batch_eval):
            print('evaluation step %d / %d' % (i + 1, num_batch_eval))
            batch_offset = i * eval_batch_size
            batch_end = (i + 1) * eval_batch_size
            eval_feature_batch = eval_feature[batch_offset:batch_end]
            eval_label_batch = eval_label[batch_offset:batch_end]
            acc_batch = sess.run(eval_op, feed_dict={feature_ph: eval_feature_batch, label_ph: eval_label_batch})
            acc_sum += acc_batch

        acc_avg = acc_sum / num_batch_eval

    print('evaluation accuracy:{}'.format(acc_avg))
