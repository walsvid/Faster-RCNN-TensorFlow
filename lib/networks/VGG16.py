import tensorflow as tf
from networks.network import Network


class VGG16(Network):
    def __init__(self, is_train):
        super().__init__(is_train)
        self.inputs = []
        self.is_train = is_train

        self.n_classes = 21
        self.feat_stride = [16, ]
        self.anchor_scales = [8, 16, 32]

        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes})

        self.bbox_weights, self.bbox_biases = None, None
        self.bbox_weights_assign, self.bbox_bias_assign = None, None

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

    def setup(self):
        # ========= HeadNet ============
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
         .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
         .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3'))
        # ========= RPN ============
        (self.feed('conv5_3')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')
         .conv(1, 1, len(self.anchor_scales) * 3 * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score'))
        if self.is_train:
            (self.feed('rpn_cls_score', 'gt_boxes', 'im_info', 'data')
             .anchor_target_layer(self.feat_stride, self.anchor_scales, name='rpn-data'))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
         .conv(1, 1, len(self.anchor_scales) * 3 * 4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred'))

        # ========= RoI Proposal ============
        (self.feed('rpn_cls_score')
         .reshape_layer(2, name='rpn_cls_score_reshape')
         .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
         .reshape_layer(len(self.anchor_scales) * 3 * 2, name='rpn_cls_prob_reshape'))
        if self.is_train:
            (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
             .proposal_layer(self.feat_stride, self.anchor_scales, 'TRAIN', name='rpn_rois'))

            (self.feed('rpn_rois', 'gt_boxes')
             .proposal_target_layer(self.n_classes, name='roi-data'))
        else:
            (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
             .proposal_layer(self.feat_stride, self.anchor_scales, 'TEST', name='rois'))

        # ========= RCNN ============
        if self.is_train:
            (self.feed('conv5_3', 'roi-data')
             .roi_pool(7, 7, 1.0 / 16, name='pool_5')
             .fc(4096, name='fc6')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc7')
             .dropout(0.5, name='drop7')
             .fc(self.n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

            (self.feed('drop7')
             .fc(self.n_classes * 4, relu=False, name='bbox_pred'))
            self.bbox_normalization()
        else:
            (self.feed('conv5_3', 'rois')
             .roi_pool(7, 7, 1.0 / 16, name='pool_5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(self.n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

            (self.feed('fc7')
             .fc(self.n_classes * 4, relu=False, name='bbox_pred'))

        return self
