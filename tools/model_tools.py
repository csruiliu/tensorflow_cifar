import tensorflow as tf


def train_model(model, train_labels, opt='Momentum', lr=0.1):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(train_labels, model)
    train_loss = tf.reduce_mean(cross_entropy)
    # cross_entropy_cost = tf.reduce_mean(cross_entropy)
    # reg_loss = tf.losses.get_regularization_loss()
    # train_loss = cross_entropy_cost + reg_loss

    tf.trainable_variables()

    if opt == 'Adam':
        train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
    elif opt == 'SGD':
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(train_loss)
    elif opt == 'Adagrad':
        train_op = tf.train.AdagradOptimizer(lr).minimize(train_loss)
    elif opt == 'Momentum':
        train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(train_loss)
    else:
        raise ValueError('Optimizer is not recognized')

    return train_op


def evaluate_model(model, eval_labels):
    prediction = tf.equal(tf.argmax(model, -1), tf.argmax(eval_labels, -1))
    eval_op = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return eval_op

