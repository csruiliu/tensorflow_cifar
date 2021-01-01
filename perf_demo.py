import tensorflow as tf
import argparse

from models.resnet import ResNet
from models.densenet import DenseNet
from tools.dataset_loader import load_cifar10_keras


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

    # load cifar10 dataset
    train_feature, train_label, eval_feature, eval_label = load_cifar10_keras()

    # load CNN model
    # model = ResNet(residual_layer=18, num_classes=10)
    model = DenseNet(residual_layer=121, num_classes=10)

    feature_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    label_ph = tf.placeholder(tf.int32, [None, 10])

    logit = model.build(feature_ph)
    train_op = model.train(logit, label_ph, opt, lr)
    eval_op = model.evaluate(logit, label_ph)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label.shape[0] // batch_size
        rest_feature = train_label.shape[0] - batch_size * num_batch

        for e in range(num_epoch):
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

        acc_avg = sess.run(eval_op, feed_dict={feature_ph: eval_feature, label_ph: eval_label})

    print('evaluation accuracy:{}'.format(acc_avg))
