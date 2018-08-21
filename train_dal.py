from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import network
import association
import utils

from datetime import datetime
from scipy.io import savemat


# Flags governing network training
tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1',
                           'The name of the architecture to train.')
tf.app.flags.DEFINE_float('weight_decay', 0.00004,
                          'The weight decay on the model weights.')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           'If specified, restore this pretrained model (e.g. ImageNet pretrained).')
tf.app.flags.DEFINE_float('ema_decay', 0.9999,
                          'The decay to use for the moving average.')
tf.app.flags.DEFINE_float('grad_clip', 2.0,
                          'The gradient clipping threshold to stabilize the training.')
tf.app.flags.DEFINE_float('label_smoothing', 0.1,
                          'The amount of label smoothing.')
tf.app.flags.DEFINE_string('train_dir', '/tmp/tfmodel/',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('feature_dir', None,
                           'Directory where the pre-extracted features are stored for anchor initialization.')
# Flags governing data preprocessing
tf.app.flags.DEFINE_integer('num_readers', 4,
                            'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4,
                            'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            'Size of the queue of preprocessed images.')
tf.app.flags.DEFINE_integer('batch_size', 128,
                            'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            'Number of batches to run.')
# Flags governing the employed hardware
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'How many GPUs to use.')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')
# Flags governing dataset characteristics
tf.app.flags.DEFINE_string('dataset_name', 'MARS',
                           'The name of the dataset, either "MARS", "PRID2011" or "iLIDS-VID".')
tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('preprocessing_name', 'reidnet',
                           'The name of the preprocessing to use. ')
tf.app.flags.DEFINE_integer('image_size', 224,
                            'Train image size')
tf.app.flags.DEFINE_integer('num_classes', None,
                            'Number of classes.')
tf.app.flags.DEFINE_integer('num_samples', None,
                            'Number of classes.')
tf.app.flags.DEFINE_integer('num_cams', 2,
                            'Number of cameras.')
# Flags governing optimiser
tf.app.flags.DEFINE_string('optimizer', 'rmsprop',
                           'The name of the optimizer, either "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')
# Flags governing the anchor learning
tf.app.flags.DEFINE_string('feature_name', 'AvgPool_1a',
                           'Name of the feature layer.')
tf.app.flags.DEFINE_integer('feature_dim', 1024,
                            'Dimension of feature vector.')
tf.app.flags.DEFINE_integer('warm_up_epochs', 2,
                            'Number of epochs to start tracklet association.')
tf.app.flags.DEFINE_float('margin', 0.5,
                          'Margin of triplet loss.')
tf.app.flags.DEFINE_float('eta', 0.5,
                          'Learning rate to update anchors.')


FLAGS = tf.app.flags.FLAGS


# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.


def _tower_loss(images, labels, cams, num_classes,
                reuse_variables=None, start_sign=0.0):

    # Build inference graph.
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        _, features = \
            network.inference_with_feature(
                images, num_classes, FLAGS.feature_name)

    # Build anchor learning graph & compute loss
    final_loss, anchors = \
        association.learning_graph(
            features, labels, cams, reuse_variables, start_sign)

    # Assemble all of the losses for the current tower only.
    losses =[final_loss]
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    tf.summary.scalar('triplet_loss', final_loss)

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

    return total_loss, anchors


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # split the batch across GPUs.
        assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
            'Batch size must be divisible by number of GPUs')

        start_sign_placeholder = tf.placeholder(tf.bool, name='start_sign')

        images, labels, cams = utils.prepare_data('train')

        # Split the batch of images and labels for towers.
        images_splits = tf.split(images, FLAGS.num_gpus, 0)
        labels_splits = tf.split(labels, FLAGS.num_gpus, 0)
        cams_splits = tf.split(cams, FLAGS.num_gpus, 0)

        num_classes = FLAGS.num_classes + 1
        global_step = slim.create_global_step()

        # Create an optimizer that performs gradient descent.
        if FLAGS.optimizer == 'rmsprop':
            # Calculate the learning rate schedule.
            num_batches_per_epoch = (FLAGS.num_samples / FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                            global_step,
                                            decay_steps,
                                            FLAGS.learning_rate_decay_factor,
                                            staircase=True)
            opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                            momentum=RMSPROP_MOMENTUM,
                                            epsilon=RMSPROP_EPSILON)
        elif FLAGS.optimizer =='sgd':
            boundaries = [int(1/2 * float(FLAGS.max_steps))]
            boundaries = list(np.array(boundaries, dtype=np.int64))
            values = [0.01, 0.001]
            lr = tf.train.piecewise_constant(global_step, boundaries, values)
            opt = tf.train.MomentumOptimizer(learning_rate=lr,
                                             momentum=0.9,
                                             use_nesterov=True)

        tower_grads = []
        anchors_op = []
        reuse_variables = None
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (network.TOWER_NAME, i)) as scope:
                    with slim.arg_scope(slim.get_model_variables(scope=scope), device='/cpu:0'):
                        # Calculate the loss for one tower of the model.
                        loss, anchors = \
                            _tower_loss(images_splits[i], labels_splits[i], cams_splits[i],
                                        num_classes, reuse_variables, start_sign_placeholder)

                        anchors_op.append(anchors)

                    # Reuse variables for the next tower.
                    reuse_variables = True

                    batchnorm = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                    batchnorm = [var for var in batchnorm if not 'Logits' in var.name]

                    trainable_var = tf.trainable_variables()
                    trainable_var = [var for var in trainable_var if not 'Logits' in var.name]

                    grads = opt.compute_gradients(loss, var_list=trainable_var)
                    tower_grads.append(grads)

        # synchronize gradients across all towers
        grads = network.average_gradients(tower_grads)
        gradient_op = opt.apply_gradients(grads, global_step=global_step)

        var_averages = tf.train.ExponentialMovingAverage(FLAGS.ema_decay, global_step)
        var_average = tf.trainable_variables()
        var_average = [var for var in var_average if not 'Logits' in var.name]
        var_op = var_averages.apply(var_average)

        batchnorm_op = tf.group(*batchnorm)
        train_op = tf.group(gradient_op, var_op, batchnorm_op)

        saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)
        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # continue training from existing model
        if FLAGS.pretrained_model_checkpoint_path:
            var_to_restore = [var for var in trainable_var if not 'Logits' in var.name]
            restorer = tf.train.Saver(var_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        tf.train.start_queue_runners(sess=sess)
        step_1_epoch = int(float(FLAGS.num_samples)/float(FLAGS.batch_size))

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, _, loss_value = \
                sess.run([train_op, anchors_op, loss],
                         feed_dict={start_sign_placeholder:
                                    step>=step_1_epoch*FLAGS.warm_up_epochs})

            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.4f '
                              '(%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

            if (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step+1)


def main(_):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
