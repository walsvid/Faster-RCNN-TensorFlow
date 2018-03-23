import functools
import numpy as np
import tensorflow as tf
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn.proposal_layer import proposal_layer as proposal_layer_py
from rpn.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn.proposal_target_layer import proposal_target_layer as proposal_target_layer_py



DEFAULT_PADDING = 'SAME'


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
    def __init__(self, trainable=True):
        self.inputs = []
        self.layers = dict()
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, saver, ignore_missing=False):
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

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, inputs, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
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
    def relu(self, inputs, name):
        return tf.nn.relu(inputs, name=name)

    @layer
    def max_pool(self, inputs, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(inputs,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, inputs, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(inputs,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, inputs, pooled_height, pooled_width, spatial_scale, name):
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
    def proposal_layer(self, inputs, _feat_stride, anchor_scales, cfg_key, name):
        if isinstance(inputs[0], tuple):
            inputs[0] = inputs[0][0]
        return tf.reshape(
            tf.py_func(proposal_layer_py, [inputs[0], inputs[1], inputs[2], cfg_key, _feat_stride, anchor_scales],
                       [tf.float32]), [-1, 5], name=name)

    @layer
    def anchor_target_layer(self, inputs, _feat_stride, anchor_scales, name):
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
        input_shape = tf.shape(inputs)
        if name == 'rpn_cls_prob_reshape':
            return tf.transpose(tf.reshape(tf.transpose(inputs, [0, 3, 1, 2]), [input_shape[0],
                                                                                int(d), tf.cast(
                    tf.cast(input_shape[1], tf.float32) / tf.cast(d, tf.float32) * tf.cast(input_shape[3], tf.float32),
                    tf.int32), input_shape[2]]), [0, 2, 3, 1], name=name)
        else:
            return tf.transpose(tf.reshape(tf.transpose(inputs, [0, 3, 1, 2]), [input_shape[0],
                                                                                int(d), tf.cast(
                    tf.cast(input_shape[1], tf.float32) * (
                            tf.cast(input_shape[3], tf.float32) / tf.cast(d, tf.float32)), tf.int32),
                                                                                input_shape[2]]), [0, 2, 3, 1],
                                name=name)

    # @layer
    # def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
    #     return feature_extrapolating_op.feature_extrapolating(input,
    #                           scales_base,
    #                           num_scale_base,
    #                           num_per_octave,
    #                           name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
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
        input_shape = tf.shape(inputs)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(inputs, [-1, input_shape[3]])),
                              [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)
        else:
            return tf.nn.softmax(inputs, name=name)

    @layer
    def dropout(self, inputs, keep_prob, name):
        return tf.nn.dropout(inputs, keep_prob, name=name)
