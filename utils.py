from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import dataset_factory
from preprocessing import preprocessing_factory


FLAGS = tf.app.flags.FLAGS


def prepare_data(dataset_split_name):

    if dataset_split_name == 'train':
        train_sign = True
    else:
        train_sign = False

    # Select the dataset
    dataset = dataset_factory.get_dataset(
        name='data',
        split_name=dataset_split_name,
        dataset_dir=FLAGS.dataset_dir,
        num_classes=FLAGS.num_classes,
        num_samples=FLAGS.num_samples)

    # Create a dataset provider that loads data
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        shuffle=train_sign,
        common_queue_capacity=8 * FLAGS.batch_size,
        common_queue_min=4 * FLAGS.batch_size,)
    [image, label, cam] = provider.get(['image', 'tracklet_id', 'cam_id'])

    # define preprocessing function
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        FLAGS.preprocessing_name, is_training=train_sign)
    image = image_preprocessing_fn(image, FLAGS.image_size, int(FLAGS.image_size/2))

    # put data in a queue
    images, labels, cams = tf.train.batch(
        [image, label, cam],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=2 * FLAGS.num_preprocessing_threads * FLAGS.batch_size)

    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels, cams], capacity=2 * FLAGS.num_preprocessing_threads * FLAGS.batch_size)

    images, labels, cams = batch_queue.dequeue()

    return images, labels, cams


