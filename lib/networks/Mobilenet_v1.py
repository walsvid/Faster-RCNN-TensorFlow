import tensorflow as tf
from networks.network import Network


class MobilenetV1(Network):
    def __init__(self, is_train):
        super().__init__(is_train)
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        # self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        # self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')

        self.n_classes = 21
        self.feat_stride = [16, ]
        self.anchor_scales = [8, 16, 32]

        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes})

    def setup(self):
        # ========= HeadNet ============
        (self.feed('data')
         .conv(3, 3, 32, 2, 2, biased=False, relu=False, name='conv1')
         .batch_normalization(relu=True, name='conv1_bn')
         .conv(3, 3, 32, 1, 1, biased=False, group=32, relu=False, name='conv2_1_dw')
         .batch_normalization(relu=True, name='conv2_1_dw_bn')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_1_sep')
         .batch_normalization(relu=True, name='conv2_1_sep_bn')
         .conv(3, 3, 64, 2, 2, biased=False, group=64, relu=False, name='conv2_2_dw')
         .batch_normalization(relu=True, name='conv2_2_dw_bn')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_2_sep')
         .batch_normalization(relu=True, name='conv2_2_sep_bn')
         .conv(3, 3, 128, 1, 1, biased=False, group=128, relu=False, name='conv3_1_dw')
         .batch_normalization(relu=True, name='conv3_1_dw_bn')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_1_sep')
         .batch_normalization(relu=True, name='conv3_1_sep_bn')
         .conv(3, 3, 128, 2, 2, biased=False, group=128, relu=False, name='conv3_2_dw')
         .batch_normalization(relu=True, name='conv3_2_dw_bn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv3_2_sep')
         .batch_normalization(relu=True, name='conv3_2_sep_bn')
         .conv(3, 3, 256, 1, 1, biased=False, group=256, relu=False, name='conv4_1_dw')
         .batch_normalization(relu=True, name='conv4_1_dw_bn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_1_sep')
         .batch_normalization(relu=True, name='conv4_1_sep_bn')
         .conv(3, 3, 256, 2, 2, biased=False, group=256, relu=False, name='conv4_2_dw')
         .batch_normalization(relu=True, name='conv4_2_dw_bn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_2_sep')
         .batch_normalization(relu=True, name='conv4_2_sep_bn')
         .conv(3, 3, 512, 1, 1, biased=False, group=512, relu=False, name='conv5_1_dw')
         .batch_normalization(relu=True, name='conv5_1_dw_bn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_1_sep')
         .batch_normalization(relu=True, name='conv5_1_sep_bn')
         .conv(3, 3, 512, 1, 1, biased=False, group=512, relu=False, name='conv5_2_dw')
         .batch_normalization(relu=True, name='conv5_2_dw_bn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_2_sep')
         .batch_normalization(relu=True, name='conv5_2_sep_bn')
         .conv(3, 3, 512, 1, 1, biased=False, group=512, relu=False, name='conv5_3_dw')
         .batch_normalization(relu=True, name='conv5_3_dw_bn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_sep')
         .batch_normalization(relu=True, name='conv5_3_sep_bn')
         .conv(3, 3, 512, 1, 1, biased=False, group=512, relu=False, name='conv5_4_dw')
         .batch_normalization(relu=True, name='conv5_4_dw_bn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_4_sep')
         .batch_normalization(relu=True, name='conv5_4_sep_bn')
         .conv(3, 3, 512, 1, 1, biased=False, group=512, relu=False, name='conv5_5_dw')
         .batch_normalization(relu=True, name='conv5_5_dw_bn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_5_sep')
         .batch_normalization(relu=True, name='conv5_5_sep_bn')
         .conv(3, 3, 512, 2, 2, biased=False, group=512, relu=False, name='conv5_6_dw')
         .batch_normalization(relu=True, name='conv5_6_dw_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv5_6_sep')
         .batch_normalization(relu=True, name='conv5_6_sep_bn')
         .conv(3, 3, 1024, 1, 1, biased=False, group=1024, relu=False, name='conv6_dw')
         .batch_normalization(relu=True, name='conv6_dw_bn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv6_sep')
         .batch_normalization(relu=True, name='conv6_sep_bn'))

        # ========= RPN ============
        (self.feed('conv6_sep_bn')
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
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
         .spatial_reshape_layer(len(self.anchor_scales) * 3 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(self.feat_stride, self.anchor_scales, self.mode, name=self.proposal_layer_name))

        if self.is_train:
            (self.feed('rpn_rois', 'gt_boxes')
             .proposal_target_layer(self.n_classes, name='roi-data'))

        feed_layer = 'roi-data' if self.is_train else 'rois'
        (self.feed('conv6_sep_bn', feed_layer)
         .roi_pool(7, 7, 1.0 / 16, name='roi_pool')
         .conv(1, 1, 1000, 1, 1, relu=False, name='fc7')
         .conv(1, 1, self.n_classes, 1, 1, relu=False, name='cls_score')
         .softmax(name='cls_prob'))

        (self.feed('fc7')
         .conv(1, 1, self.n_classes * 4, 1, 1, relu=False, name='bbox_pred'))

        if self.is_train:
            self.bbox_normalization()

        return self
