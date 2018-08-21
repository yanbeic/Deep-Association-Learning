from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import dataset_utils

_FILE_PATTERN = '%sdata_*.tfrecord'


_ITEMS_TO_DESCRIPTIONS = {
    'image/path': 'The image path.',
    'image/encoded': 'The encoded image data.',
    'image/format': 'The image format.',
    'image/tracklet_id': 'A single integer denotes tracklet ID under each camera.',
    'image/cam_id': 'A single integer denotes camera ID (camera network ID).',
    'image/height': 'A single integer denotes image height.',
    'image/width': 'A single integer denotes image width.',
}


def get_split(split_name, dataset_dir, division_idx=None,
              file_pattern=None, reader=None, num_classes=None, num_samples=None):
    """Gets a dataset tuple with instructions for reading training data.

    Returns:
        A 'Dataset' namedtuple.

    Raises:
        ValueError: if 'split_name' is not the train split.
    """

    if division_idx is not None:
        raise ValueError('No division index is needed for the training dataset.')

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/tracklet_id': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/cam_id': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'tracklet_id': slim.tfexample_decoder.Tensor('image/tracklet_id'),
        'cam_id': slim.tfexample_decoder.Tensor('image/cam_id'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None

    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=num_classes,
        labels_to_names=labels_to_names)
