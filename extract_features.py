from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np
import scipy.io as sio
import sys
import utils

from tensorflow.contrib import slim
from nets import nets_factory


tf.app.flags.DEFINE_string('dataset_name', None,
                           'The name of the dataset to load.')
tf.app.flags.DEFINE_string('store_name', 'test',
                           'The name of the train/test split to store as mat files.')
tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('feature_dir', None,
                           'The directory where the feature files are stored.')
tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1',
                           'The name of the architecture to train.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_string('feature_type', None,
                           'The type of feature you are going to extract.')
tf.app.flags.DEFINE_string('preprocessing_name', 'reidnet',
                           'The name of the preprocessing to use.')
tf.app.flags.DEFINE_float('ema_decay', 0.9999,
                          'The decay to use for the moving average.')
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('image_size', 224,
                            'Train image height')
tf.app.flags.DEFINE_integer('num_readers', 1, 
                            'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 1,
                            'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('feature_dim', 1024,
                            'The dimension of feature vector.')
tf.app.flags.DEFINE_integer('num_classes', None,
                            'Number of classes for the network')
tf.app.flags.DEFINE_integer('num_samples', None,
                            'Number of test samples')
tf.app.flags.DEFINE_integer('num_matfiles', 1,
                            'The number of mat files to be converted.')


FLAGS = tf.app.flags.FLAGS


def main(_):
    with tf.Graph().as_default():

        # prepare dataset
        numsamples = FLAGS.num_samples
        images, labels, _ = utils.prepare_data('test')

        # prepare network
        network_fn = nets_factory.get_network_fn(
                     FLAGS.model_name,
                     num_classes=FLAGS.num_classes,
                     is_training=False)

        with slim.arg_scope(nets_factory.arg_scopes_map[FLAGS.model_name]()):
            logits, endpoints = network_fn(images)
            activations = endpoints[FLAGS.feature_type]
            activations = tf.contrib.layers.flatten(activations)

        var_averages = tf.train.ExponentialMovingAverage(FLAGS.ema_decay)
        var_to_restore = var_averages.variables_to_restore()

        # only restores the feature extractors
        var_to_restore = {k: v for k, v in var_to_restore.iteritems() if not 'Logits' in k}
        saver = tf.train.Saver(var_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if (ckpt and ckpt.model_checkpoint_path):
                # Restores from checkpoint with absolute path.
                print('Restores from checkpoint path: '+ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Successfully loaded model from %s at step=%s.' %
                      (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                return

            np_features = np.zeros(shape=[FLAGS.feature_dim, numsamples], dtype=np.float32)
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                index = 0

                print('Starting extaction on (%s). \n' % 'test data.')
                while index < numsamples and not coord.should_stop():
                    np_labels, np_activations = sess.run([labels, activations])
                    np_features[:, index] = np.transpose(np_activations[0])
                    sys.stdout.write('\r>> Extracting image features %d/%d.' % (index + 1, numsamples))
                    sys.stdout.flush()
                    index += 1
                print('\n Finished extraction on (%s). \n' % 'test data.')

            except Exception as e:
                coord.request_stop(e)
            # When done, ask the threads to stop.
            coord.request_stop()
            # Wait for threads to finish.
            coord.join(threads, stop_grace_period_secs=10)

    # check if the directory to store features exists, else make a new one
    if not os.path.exists(FLAGS.feature_dir):
        os.mkdir(FLAGS.feature_dir)

    if FLAGS.num_matfiles == 1:
        # store feature as one mat file (when data scale is small)
        save_filename = FLAGS.store_name + '.mat'
        sio.savemat(FLAGS.feature_dir + save_filename,
                    {FLAGS.store_name: np_features})
    else:
        # store feature as multiple (two) mat files (when data scale is large)
        num_matfiles = FLAGS.num_matfiles
        feature_size = np_features.shape
        sample_size = int(feature_size[1]/num_matfiles)
        for i in range(num_matfiles):
            save_filename = FLAGS.store_name + str(i) + '.mat'
            if i != (num_matfiles-1):
                temp = np_features[:, int(i * sample_size):int((i + 1) * sample_size)]
            else:
                temp = np_features[:, int(i * sample_size):int(feature_size[1])]
            sio.savemat(FLAGS.feature_dir + save_filename, {FLAGS.store_name: temp})


if __name__ == '__main__':
    tf.app.run()
