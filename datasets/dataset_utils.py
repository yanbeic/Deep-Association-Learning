from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import tensorflow as tf

from six.moves import urllib


LABELS_FILENAME = 'labels.txt'


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    # Check whether or not the dataset directory contains a label map file.
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    # Read the labels file and returns a mapping from ID to class name.
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'r') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names


def image_to_re_id_tfexample(image_path, image_data, image_format,
                             height, width,
                             tracklet_id, cam_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/path': bytes_feature(image_path),
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/tracklet_id': int64_feature(tracklet_id),
        'image/cam_id': int64_feature(cam_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))
