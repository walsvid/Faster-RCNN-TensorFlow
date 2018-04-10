# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.client import timeline
import time


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, imdb, roidb, output_dir, log_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print('Computing bounding-box regression targets...')
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print('done')

        # For checkpoint
        self.saver = saver
        # Log summary writer
        self.writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph(), flush_secs=5)

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers:
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers:
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box = self.net.build_loss(ohem=False)
        summary_op = self.net.summarizer()

        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)

        # optimizer
        if cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
        else:
            momentum_lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                                     cfg.TRAIN.STEPSIZE, cfg.TRAIN.GAMMA, staircase=True)
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(momentum_lr, momentum)

        if cfg.TRAIN.WITH_CLIP:
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10.0)
            train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        else:
            train_op = opt.minimize(loss, global_step=global_step)

        # initialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print('Loading pretrained model weights from {:s}'.format(self.pretrained_model))
            self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):

            # get one batch
            blobs = data_layer.forward()

            # Make one SGD update
            feed_dict = {self.net.data: blobs['data'], self.net.im_info: blobs['im_info'],
                         self.net.keep_prob: 0.5, self.net.gt_boxes: blobs['gt_boxes']}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()

            fetch_list = [rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, summary_op, train_op]
            rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, summary_str,\
                _ = sess.run(fetches=fetch_list, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

            self.writer.add_summary(summary=summary_str, global_step=global_step.eval())

            if cfg.TRAIN.SOLVER == 'Adam':
                lr = opt._lr_t.eval()
            elif cfg.TRAIN.SOLVER == 'RMS' or cfg.TRAIN.SOLVER == 'Momentum':
                lr = opt._learning_rate_tensor.eval()
            else:
                raise NotImplementedError

            timer.toc()

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(int(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % cfg.TRAIN.DISPLAY == 0:
                print(
                    'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, '
                    'loss_box: %.4f, lr: %f' %
                    (iter + 1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value,
                     rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, lr))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    layer = RoIDataLayer(roidb, num_classes)
    return layer


def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after))
    return filtered_roidb


def train_net(network, imdb, roidb, output_dir, log_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb)
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, log_dir, pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(sess, max_iters)
        print('done solving')
