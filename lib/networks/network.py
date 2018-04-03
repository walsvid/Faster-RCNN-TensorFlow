import functools
import numpy as np
import tensorflow as tf
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import psroi_pooling_layer.psroi_pooling_op as psroi_pooling_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn.proposal_layer import proposal_layer as proposal_layer_py
from rpn.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.config import cfg


DEFAULT_PADDING = 'SAME'


def include_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator


@include_original
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, is_train=True):
        self.inputs = []
        self.layers = dict()
        self.is_train = is_train
        self.mode = 'TRAIN' if self.is_train else 'TEST'
        self.proposal_layer_name = 'rpn_rois' if self.is_train else 'rois'
        self.summary = []
        self.losses = {}
        self.bbox_weights, self.bbox_biases = None, None
        self.bbox_weights_assign, self.bbox_bias_assign = None, None

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load_v1(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path, encoding='latin1').item()  # type: dict
            for key in data_dict.keys():
                with tf.variable_scope(key, reuse=True):
                    # for subkey in data_dict[key]:
                    for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                        try:
                            # var = tf.get_variable(subkey)
                            # session.run(var.assign(data_dict[key][subkey]))
                            session.run(tf.get_variable(subkey).assign(data))
                            print("assign pretrain model " + subkey + " to " + key)
                        except ValueError:
                            print("ignore " + key)
                            if not ignore_missing:
                                raise

    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path, encoding='latin1').item()  # type: dict
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("assign pretrain model " + subkey + " to " + key)
                        except ValueError:
                            print("ignore " + key)
                            if not ignore_missing:
                                raise

    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer_name in args:
            if isinstance(layer_name, str):
                try:
                    layer_name = self.layers[layer_name]
                    print(layer_name)
                except KeyError:
                    print(self.layers.keys())
                    raise KeyError('Unknown layer name fed: %s' % layer_name)
            self.inputs.append(layer_name)
        return self

    def get_output(self, layer_name):
        try:
            _layer = self.layers[layer_name]
        except KeyError:
            print(self.layers.keys())
            raise KeyError('Unknown layer name fed: %s' % layer_name)
        return _layer

    def get_unique_name(self, prefix):
        _id = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, _id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv_1(self, inputs, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        self.validate_padding(padding)
        c_i = inputs.get_shape()[-1]
        c_i = int(c_i)
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = functools.partial(tf.nn.conv2d, strides=[1, s_h, s_w, 1], padding=padding)
        # convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i / group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group == 1:
                conv = convolve(inputs, kernel)
            else:
                input_groups = tf.split(3, group, inputs)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def conv(self, inputs, k_h, k_w, c_o, s_h, s_w, name, biased=True, relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = inputs.get_shape()[-1]
        convolve = functools.partial(tf.nn.conv2d, strides=[1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable,
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            conv = convolve(inputs, kernel)
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias)
                else:
                    return tf.nn.bias_add(conv, biases)
            else:
                if relu:
                    return tf.nn.relu(conv)
                else:
                    return conv

    @layer
    def relu(self, inputs, name):
        with tf.name_scope(name):
            return tf.nn.relu(inputs, name=name)

    @layer
    def max_pool(self, inputs, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        with tf.name_scope(name):
            self.validate_padding(padding)
            return tf.nn.max_pool(inputs,
                                  ksize=[1, k_h, k_w, 1],
                                  strides=[1, s_h, s_w, 1],
                                  padding=padding,
                                  name=name)

    @layer
    def avg_pool(self, inputs, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        with tf.name_scope(name):
            self.validate_padding(padding)
            return tf.nn.avg_pool(inputs,
                                  ksize=[1, k_h, k_w, 1],
                                  strides=[1, s_h, s_w, 1],
                                  padding=padding,
                                  name=name)

    @layer
    def roi_pool(self, inputs, pooled_height, pooled_width, spatial_scale, name):
        with tf.name_scope(name):
            # only use the first input
            if isinstance(inputs[0], tuple):
                inputs[0] = inputs[0][0]

            if isinstance(inputs[1], tuple):
                inputs[1] = inputs[1][0]

            print(inputs)
            return roi_pool_op.roi_pool(inputs[0], inputs[1],
                                        pooled_height,
                                        pooled_width,
                                        spatial_scale,
                                        name=name)[0]

    @layer
    def psroi_pool(self, inputs, output_dim, group_size, spatial_scale, name):
        with tf.name_scope(name):
            # only use the first input
            if isinstance(inputs[0], tuple):
                inputs[0] = inputs[0][0]

            if isinstance(inputs[1], tuple):
                inputs[1] = inputs[1][0]

            return psroi_pooling_op.psroi_pool(inputs[0], inputs[1],
                                               output_dim=output_dim,
                                               group_size=group_size,
                                               spatial_scale=spatial_scale,
                                               name=name)[0]

    @layer
    def proposal_layer(self, inputs, _feat_stride, anchor_scales, cfg_key, name):
        with tf.name_scope(name):
            if isinstance(inputs[0], tuple):
                inputs[0] = inputs[0][0]
            return tf.reshape(
                tf.py_func(proposal_layer_py, [inputs[0], inputs[1], inputs[2], cfg_key, _feat_stride, anchor_scales],
                           [tf.float32]), [-1, 5], name=name)

    @layer
    def anchor_target_layer(self, inputs, _feat_stride, anchor_scales, name):
        with tf.name_scope(name):
            if isinstance(inputs[0], tuple):
                inputs[0] = inputs[0][0]

            with tf.variable_scope(name) as scope:
                rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights \
                    = tf.py_func(anchor_target_layer_py,
                                 [inputs[0], inputs[1], inputs[2], inputs[3], _feat_stride, anchor_scales],
                                 [tf.float32, tf.float32, tf.float32, tf.float32])

                rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
                rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
                rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
                rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

                return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def proposal_target_layer(self, inputs, classes, name):
        with tf.name_scope(name):
            if isinstance(inputs[0], tuple):
                inputs[0] = inputs[0][0]
            with tf.variable_scope(name) as scope:
                rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights \
                    = tf.py_func(proposal_target_layer_py,
                                 [inputs[0], inputs[1], classes],
                                 [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

                rois = tf.reshape(rois, [-1, 5], name='rois')
                labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
                bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')
                bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
                bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name='bbox_outside_weights')

                return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @layer
    # TODO: need re-write reshape layer
    def reshape_layer(self, inputs, d, name):
        with tf.name_scope(name):
            input_shape = tf.shape(inputs)
            if name == 'rpn_cls_prob_reshape':
                return tf.transpose(tf.reshape(tf.transpose(inputs, [0, 3, 1, 2]), [input_shape[0], int(d), tf.cast(tf.cast(input_shape[1], tf.float32) / tf.cast(d, tf.float32) * tf.cast(input_shape[3], tf.float32), tf.int32), input_shape[2]]), [0, 2, 3, 1], name=name)
            else:
                return tf.transpose(tf.reshape(tf.transpose(inputs, [0, 3, 1, 2]), [input_shape[0], int(d), tf.cast(tf.cast(input_shape[1], tf.float32) * (tf.cast(input_shape[3], tf.float32) / tf.cast(d, tf.float32)), tf.int32), input_shape[2]]), [0, 2, 3, 1], name=name)

    @layer
    def lrn(self, inputs, radius, alpha, beta, name, bias=1.0):
        with tf.name_scope(name):
            return tf.nn.local_response_normalization(inputs,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias,
                                                      name=name)

    @layer
    def concat(self, inputs, axis, name):
        with tf.name_scope(name):
            return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def fc(self, inputs, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            input_shape = inputs.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(inputs, [0, 3, 1, 2]), [-1, dim])
            else:
                feed_in, dim = (inputs, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, inputs, name):
        with tf.name_scope(name):
            input_shape = tf.shape(inputs)
            if name == 'rpn_cls_prob':
                return tf.reshape(tf.nn.softmax(tf.reshape(inputs, [-1, input_shape[3]])),
                                  [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)
            else:
                return tf.nn.softmax(inputs, name=name)

    @layer
    def dropout(self, inputs, keep_prob, name):
        with tf.name_scope(name):
            return tf.nn.dropout(inputs, keep_prob, name=name)

    @layer
    def batch_normalization(self, inputs, name, relu=True, is_training=False):
        with tf.name_scope(name):
            """contribution by miraclebiu"""
            if relu:
                bn_layer = tf.layers.batch_normalization(inputs, scale=True, center=True, training=is_training, name=name)
                return tf.nn.relu(bn_layer)
            else:
                return tf.layers.batch_normalization(inputs, scale=True, center=True, training=is_training, name=name)

    @layer
    def spatial_reshape_layer(self, inputs, d, name):
        with tf.name_scope(name):
            input_shape = tf.shape(inputs)
            # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
            return tf.reshape(inputs, [input_shape[0], input_shape[1], -1, int(d)])

    @layer
    def spatial_softmax(self, inputs, name):
        with tf.name_scope(name):
            input_shape = tf.shape(inputs)
            # d = input.get_shape()[-1]
            return tf.reshape(tf.nn.softmax(tf.reshape(inputs, [-1, input_shape[3]])), [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def add(self, inputs, name):
        """contribution by miraclebiu"""
        with tf.name_scope(name):
            return tf.add(inputs[0], inputs[1], name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                                 dtype=tensor.dtype.base_dtype,
                                                 name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')

        return regularizer

    def smooth_l1_loss(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma
        deltas = bbox_inside_weights * (bbox_pred - bbox_targets)
        deltas_abs = tf.abs(deltas)
        smooth_l1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.square(deltas) * 0.5 * sigma2 * smooth_l1_sign
        smooth_l1_option2 = (deltas_abs - 0.5 / sigma2) * tf.abs(smooth_l1_sign - 1)
        smooth_l1_result = smooth_l1_option1 + smooth_l1_option2
        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)
        return outside_mul

    # def smooth_l1_loss_v1(self, sigma, deltas):
    #     sigma2 = sigma * sigma
    #     deltas_abs = tf.abs(deltas)
    #     smooth_l1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
    #     smooth_l1_option1 = tf.square(deltas) * 0.5 * sigma2 * smooth_l1_sign
    #     smooth_l1_option2 = (deltas_abs - 0.5 / sigma2) * tf.abs(smooth_l1_sign - 1)
    #     smooth_l1_result = smooth_l1_option1 + smooth_l1_option2
    #     return smooth_l1_result

    def build_loss(self, ohem=False):
        # RPN cls loss
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_label, -1))), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, tf.where(tf.not_equal(rpn_label, -1))), [-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # RPN bbox loss
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.get_output('rpn-data')[1], [0, 2, 3, 1])
        rpn_bbox_inside_weights = tf.transpose(self.get_output('rpn-data')[2], [0, 2, 3, 1])
        rpn_bbox_outside_weights = tf.transpose(self.get_output('rpn-data')[3], [0, 2, 3, 1])

        rpn_smooth_l1 = self.smooth_l1_loss(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))

        # R-CNN cls loss
        cls_score = self.get_output('cls_score')
        label = tf.reshape(self.get_output('roi-data')[1], [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # R-CNN bbox loss
        bbox_pred = self.get_output('bbox_pred')
        bbox_targets = self.get_output('roi-data')[2]
        bbox_inside_weights = self.get_output('roi-data')[3]
        bbox_outside_weights = self.get_output('roi-data')[4]

        smooth_l1 = self.smooth_l1_loss(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))

        # final loss
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n(regularization_losses) + loss

        self.losses['cross_entropy'] = cross_entropy
        self.losses['loss_box'] = loss_box
        self.losses['rpn_cross_entropy'] = rpn_cross_entropy
        self.losses['rpn_loss_box'] = rpn_loss_box
        self.losses['loss'] = loss

        return loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box

    def summarizer(self):
        self.summary.append(tf.summary.scalar('rpn_reg_loss', self.losses['rpn_loss_box']))
        self.summary.append(tf.summary.scalar('rpn_cls_loss', self.losses['rpn_cross_entropy']))
        self.summary.append(tf.summary.scalar('loss_box', self.losses['loss_box']))
        self.summary.append(tf.summary.scalar('loss_cls', self.losses['cross_entropy']))
        self.summary.append(tf.summary.scalar('total_loss', self.losses['loss']))

        return tf.summary.merge(self.summary)

    def bbox_normalization(self):
        # create ops and placeholders for bbox normalization process
        if self.is_train:
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

                self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
                self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

                self.bbox_weights_assign = weights.assign(self.bbox_weights)
                self.bbox_bias_assign = biases.assign(self.bbox_biases)