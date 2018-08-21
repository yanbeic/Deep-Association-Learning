from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

from nets import nets_factory


FLAGS = tf.app.flags.FLAGS


TOWER_NAME = 'tower'
# A very small value to avoid numerical problems
EPISILON = 1e-8


def inference(images, num_classes):
    # Select the network
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=num_classes,
        weight_decay=FLAGS.weight_decay,
        is_training=True)
    logits, end_points = network_fn(images)

    # Add summaries for viewing model statics on TensorBoard.
    _activation_summaries(end_points)

    return [logits]


def inference_with_feature(images, num_classes, feature_name):
    # Select the network with features
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=num_classes,
        weight_decay=FLAGS.weight_decay,
        is_training=True)
    logits, end_points = network_fn(images)

    # Add summaries for viewing model statics on TensorBoard.
    _activation_summaries(end_points)
    # organize feature dimension
    features = end_points[feature_name]
    features = tf.squeeze(tf.squeeze(features, squeeze_dims=1), squeeze_dims=1)
    return [logits], features


def loss(logits, labels, name=None, weight=1.0):
    # cross entropy classification lsos
    tf.losses.softmax_cross_entropy(
        logits=logits[0], onehot_labels=labels,
        label_smoothing=FLAGS.label_smoothing,
        weights=weight, loss_collection=name)


def _activation_summary(x):
    # create summaries for activations in Tensorboard
    # (1) a summary that provides a histogram of activations.
    # (2) a summary that measure the sparsity of activations.
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _activation_summaries(endpoints):
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            _activation_summary(act)


def average_gradients(tower_grads):
    # Calculate the average gradient for each shared variable across all towers.
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        grad = tf.clip_by_value(grad, -FLAGS.grad_clip, FLAGS.grad_clip)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

