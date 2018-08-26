from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.io as sio
import numpy as np

from datasets import tracket_num


FLAGS = tf.app.flags.FLAGS


# define one line functions for
# computing distances, losses and normalisation
pairwise_distance = lambda f1, f2, dim: tf.reduce_sum(tf.square(tf.subtract(f1, f2)), dim)

hinge_loss = lambda dist_pos, dist_neg: tf.reduce_mean(tf.maximum(dist_pos - dist_neg + FLAGS.margin, 0))

l2_norm = lambda x: tf.nn.l2_normalize(x, 1, 1e-10)

normalize = lambda v: v ** 2 / (np.sum(v ** 2, 1, keepdims=True))


def init_anchor(anchors_name, cam, num_trackets):
    # initialize a set of anchor under a certain camera
    # the following two ways of initialization lead to similar performance
    if FLAGS.feature_dir:
        # initialize by pre-extracted features
        filename = FLAGS.feature_dir + 'train' + str(cam + 1) + '.mat'
        print('load features ' + filename)
        mat_contents = sio.loadmat(filename)
        train_feature = normalize(mat_contents['train' + str(cam + 1)])
        return tf.get_variable(anchors_name,
                               dtype=tf.float32,
                               initializer=train_feature,
                               trainable=False)
    else:
        # initialize as 0
        return tf.get_variable(anchors_name,
                               [num_trackets, FLAGS.feature_dim],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0),
                               trainable=False)


def get_anchor(reuse_variables):
    num_trackets = tracket_num.get_tracket_num(FLAGS.dataset_name)
    print('number of trackets is '+str(num_trackets))

    # initialize the whole sets of anchors
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        intra_anchors = []
        cross_anchors = []
        for i in range(FLAGS.num_cams):
            anchors_name = 'intra_anchors'+str(i)
            intra_anchors.append(init_anchor(anchors_name, i, num_trackets[i]))

            anchors_name = 'cross_anchors' + str(i)
            cross_anchors.append(init_anchor(anchors_name, i, num_trackets[i]))

    return intra_anchors, cross_anchors


def association_loss(features_cam, intra_anchors_n, cross_anchors_n, labels_cam):
    """
    This function compute two association losses.
    :param features_cam: features under each camera
    :param intra_anchors_n: intra-anchor under each camera
    :param cross_anchors_n: cross-anchor across camera
    :param labels_cam: tracket id under each camera
    :return: final loss value
    """
    dist_pos_intra = []
    dist_pos_cross = []
    dist_neg = []

    # loop under each camera to calculate association loss
    for i in range(FLAGS.num_cams):
        # compute the distance to the positive pairs:
        # corresponding intra-anchors (under the same camera)
        dist_pos_intra.append(pairwise_distance(features_cam[i], tf.gather(intra_anchors_n[i], labels_cam[i]), 1))

        # compute the distance to the positive pairs:
        # corresponding cross-anchors (cross the other camera(s))
        dist_pos_cross.append(pairwise_distance(features_cam[i], tf.gather(cross_anchors_n[i], labels_cam[i]), 1))

        # tracklet association ranking to compute distance to negative pairs
        dist_neg.append(association_ranking(features_cam[i], labels_cam[i], intra_anchors_n[i]))

    # compute two association losses
    dist_pos_intra = tf.concat(dist_pos_intra, 0)
    dist_pos_cross = tf.concat(dist_pos_cross, 0)
    dist_neg = tf.concat(dist_neg, 0)

    # margin-based ranking loss
    association_loss1 = hinge_loss(dist_pos_intra, dist_neg)
    association_loss2 = hinge_loss(dist_pos_cross, dist_neg)
    final_loss = association_loss1 + association_loss2

    return final_loss


def association_ranking(features_cam, labels_cam, intra_anchors_n):
    # compute the distances to all anchors under the same camera
    dist_all = pairwise_distance(features_cam[:, tf.newaxis], intra_anchors_n, 2)

    with tf.device('/cpu:0'): # place ranking on cpu
        dist_min1, rank1 = tf.nn.top_k(-dist_all, k=1, sorted=True)

    # if rank1 not match the source tracklet,
    # then dist_neg is averaged top1 distances
    non_match = tf.not_equal(labels_cam, tf.squeeze(rank1, squeeze_dims=1))
    dist_mean = tf.ones_like(dist_min1[:, 0]) * tf.reduce_mean(-dist_min1[:, 0])
    dist_neg = tf.where(non_match, -dist_min1[:, 0], dist_mean)

    return dist_neg


def cyclic_ranking(intra_anchors_batch_n, same_anchors_n, other_anchors_n,
                   labels_cam, start_sign):
    # perform cyclic ranking to discover similar tracklets across cameras
    # (1) rank to the other camera
    dist = pairwise_distance(intra_anchors_batch_n[:, tf.newaxis], other_anchors_n, 2)

    with tf.device('/cpu:0'): # place ranking on cpu
        _, rank1 = tf.nn.top_k(-dist, k=1, sorted=True)

    # features of rank1 in another camera
    rank1_anchors = tf.gather(other_anchors_n, tf.squeeze(rank1, squeeze_dims=1))

    # (2) rank back to the original camera
    dist = pairwise_distance(rank1_anchors[:, tf.newaxis], same_anchors_n, 2)

    with tf.device('/cpu:0'): # place ranking on cpu
        _, rank1 = tf.nn.top_k(-dist, k=1, sorted=True)

    # (3) consistency condition
    consistent = tf.cast(tf.equal(tf.cast(labels_cam, dtype=tf.int32), tf.squeeze(rank1, squeeze_dims=1)), tf.int32)
    consistent = tf.cast(consistent * tf.cast(start_sign, tf.int32), tf.bool)

    return consistent, rank1_anchors


def update_intra_anchor(intra_anchors, intra_anchors_n, features_cam, labels_cam):
    # update intra-anchor for each camera
    for i in range(FLAGS.num_cams):
        # compute the difference between old anchors and the new given data
        diff = tf.gather(intra_anchors_n[i], labels_cam[i]) - features_cam[i]
        # update the intra-anchors under each camera
        intra_anchors[i] = tf.scatter_sub(intra_anchors[i], labels_cam[i], FLAGS.eta * diff)

    return intra_anchors


def update_cross_anchor(cross_anchors, intra_anchors_n, intra_anchors_batch_n,
                        labels_cam, start_sign):
    # update cross-anchor
    for i in range(FLAGS.num_cams):
        # other_anchors: all the anchors under other cameras
        other_anchors_n = []
        [other_anchors_n.append(intra_anchors_n[x]) for x in range(FLAGS.num_cams) if x is not i]
        other_anchors_n = tf.concat(other_anchors_n, 0)

        consistent, rank1_anchors = \
            cyclic_ranking(intra_anchors_batch_n[i], intra_anchors_n[i], other_anchors_n, labels_cam[i], start_sign)

        # if the consistency fulfills, update by
        # merging with the best-matched rank1 anchors in another camera
        update = tf.where(consistent, (intra_anchors_batch_n[i] + rank1_anchors) / 2, intra_anchors_batch_n[i])

        # update the associate centers under each camera
        cross_anchors[i] = tf.scatter_update(cross_anchors[i], labels_cam[i], update)

    return cross_anchors


def learning_graph(features, labels, cams, reuse_variables, start_sign):
    """
    This function build the learning graph to learn intra/cross camera
    anchors and compute two association losses
    :param features: extracted features of current image frames
    :param labels: tracket ids of current image frames
    :param cams: camera ids of current image frames
    :param reuse_variables:
    :param start_sign: when to start cross-camera tracklet association
    :return: final loss and updated anchors
    """
    # obtain the set of anchors under each camera
    intra_anchors, cross_anchors = get_anchor(reuse_variables)

    # offset the tracklet id to between 0 to N
    labels = tf.cast(labels-1, tf.int32)
    cams = cams-1
    features_n = l2_norm(features)

    # normalization of all features/anchors
    labels_cam = []
    features_cam = []
    intra_anchors_n = []
    cross_anchors_n = []

    for i in range(FLAGS.num_cams):
        # A list of boolean variable denotes the indices that contain camera i
        condition_cam = tf.equal(cams, i)
        # obtain the tracklet ids under camera i
        labels_cam.append(tf.boolean_mask(labels, condition_cam))
        # obtain features under camera i
        features_cam.append(tf.boolean_mask(features_n, condition_cam))
        # obtain the normalized intra-camera anchors
        intra_anchors_n.append(l2_norm(intra_anchors[i]))
        # obtain the normalized cross-camera anchors
        cross_anchors_n.append(l2_norm(cross_anchors[i]))

    # compute two association losses
    final_loss = association_loss(features_cam, intra_anchors_n, cross_anchors_n, labels_cam)

    # update the intra-anchors
    intra_anchors = update_intra_anchor(intra_anchors, intra_anchors_n, features_cam, labels_cam)

    # re-obtain the updated intra-anchors
    intra_anchors_n = []
    intra_anchors_batch_n = []
    for i in range(FLAGS.num_cams):
        intra_anchors_n.append(l2_norm(intra_anchors[i]))
        intra_anchors_batch_n.append(tf.gather(intra_anchors_n[i], labels_cam[i]))

    # update the cross-anchors
    cross_anchors = update_cross_anchor(cross_anchors, intra_anchors_n, intra_anchors_batch_n, labels_cam, start_sign)

    return final_loss, [intra_anchors, cross_anchors]
