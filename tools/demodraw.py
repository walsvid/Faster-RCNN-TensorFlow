#!/usr/bin/env python

import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import random
from networks.factory import get_network

output_img_dir = './outputs_img'

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


# CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def draw_all_detection(im, class_name, dets, boxcolor, thresh=0.5):
    """
    Draw detected bounding boxes.
    """
    color_white = (255, 255, 255)
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        # print("inds = 0, return!")
        return
    for ind in inds:
        bbox = dets[ind, :4]
        score = dets[ind, -1]
        bbox = list(map(int, bbox))
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), boxcolor, thickness=2)
        cv2.putText(im, '%s %.3f' % (class_name, score), (bbox[0], bbox[1] + 10),
                    color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im


def demo(sess, net, image_name, color_set):
    """Detect object classes in an image using pre-computed object proposals."""
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    # im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class=

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
        boxcolor = color_set[cls]
        res = draw_all_detection(im, cls, dets, boxcolor, thresh=CONF_THRESH)
        if res is not None:  # only have detection results, then overlap the im
            im = res
        else:
            continue
            # write to image
        if im is not None:
            rect_image_name = ('rect_%s' % image_name)
            cv2.imwrite(os.path.join(output_img_dir, rect_image_name), im)
            print('save predicted image in %s' % os.path.join(output_img_dir, rect_image_name))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default='./')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    # saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    # saver.restore(sess, args.model)

    ckpt_path = os.path.dirname(args.model)
    print(ckpt_path)
    saver = tf.train.Saver()
    try:
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except :
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    # sess.run(tf.initialize_all_variables())

    print('\n\nLoaded network {:s}'.format(args.model))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = im_detect(sess, net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']

    color_set = {}
    for cls in CLASSES[1:]:
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        color_set[cls] = color

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name, color_set)

    plt.show()
