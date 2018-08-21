from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import math
import shutil
import tensorflow as tf
import numpy as np

from PIL import Image
from datasets import dataset_utils


def _write_to_tfrecord(filenames, tracklet_ids, cam_ids,
                       dataset_dir, output_dir, split_name, num_tfrecords):

    num_images = len(filenames)
    num_per_shard = int(math.ceil(len(filenames) / float(num_tfrecords)))

    with tf.Graph().as_default():

        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder)

        with tf.Session('') as sess:

            for shard_id in range(num_tfrecords):

                output_filename = _get_output_filename(
                    output_dir, split_name, shard_id, num_tfrecords)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:

                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, num_images)

                    # convert each image one by one to tfrecord
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, num_images, shard_id))
                        sys.stdout.flush()

                        # read image and label
                        image_path = filenames[i]
                        image = Image.open(dataset_dir+image_path)
                        img = np.array(image)
                        
                        height = img.shape[0]
                        width = img.shape[1]

                        tracklet_id = tracklet_ids[i]
                        cam_id = cam_ids[i]

                        # encode image to png string
                        png_string = sess.run(encoded_image,
                                              feed_dict={image_placeholder: image})

                        example = dataset_utils.image_to_re_id_tfexample(
                            image_path, png_string, 'png',
                            height, width,
                            tracklet_id, cam_id)
                        tfrecord_writer.write(example.SerializeToString())



def _get_output_filename(output_dir, split_name, shard_id, num_tfrecords):
  """Creates the output filename.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  output_filename = '%sdata_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_tfrecords)
  return os.path.join(output_dir, output_filename)



def _get_image_filenames_and_labels(filename, split_name):
    """
    :return: image filenames and labels
    """

    with open(filename) as f:
        content = f.readlines()

    # filename/path of image
    filenames = [temp.split(' ')[0] for temp in content]
    # tracklet id under each camera
    tracklet_ids = [int(temp.split(' ')[1]) for temp in content]
    # camera id
    cam_ids = [int(temp.split(' ')[2]) for temp in content]

    if split_name == 'train':
        # Shuffle the ordering of all traing image files.
        print('\nShuffling training dataset...')
        shuffled_index = list(range(len(filenames)))
        random.seed(12345)
        random.shuffle(shuffled_index)

        filenames = [filenames[i] for i in shuffled_index]
        tracklet_ids = [tracklet_ids[i] for i in shuffled_index]
        cam_ids = [cam_ids[i] for i in shuffled_index]

    return filenames, tracklet_ids, cam_ids



def run(dataset_dir, output_dir, filename, data_type, num_tfrecords):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The directory where the dataset is stored.
    output_dir: The directory where the tfrecords should be stored.
    filename: Name of a txt file that stores all the training data details.
  """

  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)
    print('\nNeed to download dataset first!')
    return

  # the name of the converted tfrecord
  TF_filename = _get_output_filename(output_dir, data_type, 0, num_tfrecords)

  if tf.gfile.Exists(TF_filename):
    print('\nDataset files already exist. Remove them and recreate a new directory.')
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

  # process the training data:
  filenames, tracklet_ids, cam_ids \
      = _get_image_filenames_and_labels(filename, data_type)

  _write_to_tfrecord(filenames, tracklet_ids, cam_ids,
                     dataset_dir, output_dir, data_type, num_tfrecords)

  unique_labels = list(set(tracklet_ids))
  unique_labels.sort()
  labels_to_write = dict(zip(range(len(unique_labels)), unique_labels))
  dataset_utils.write_label_file(labels_to_write, output_dir)

  print('\nFinished converting the training data!')
