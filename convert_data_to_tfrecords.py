from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from datasets import convert_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_type', None,
                           'The type of the dataset to convert, need to be either "train" or "test".')
tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the image files are saved.')
tf.app.flags.DEFINE_string('output_dir', None,
                           'The directory where the output TFRecords are saved.')
tf.app.flags.DEFINE_string('filename', None,
                           'The txt file where the list all image files to be converted.')
tf.app.flags.DEFINE_integer('num_tfrecords', 1, 
                            'Number of tfrecords to convert.')


def main(_):
    # check if dir exits and make it
    directory = FLAGS.output_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    # start convert data to tfrecords
    convert_data.run(dataset_dir=FLAGS.dataset_dir, 
                     output_dir=FLAGS.output_dir,
                     filename=FLAGS.filename, 
                     data_type=FLAGS.data_type, 
                     num_tfrecords=FLAGS.num_tfrecords)


if __name__ == '__main__':
    tf.app.run()
