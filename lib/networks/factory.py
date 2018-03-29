# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

import networks.VGG16
import networks.Resnet50
import pdb
import tensorflow as tf


__sets = {}


def _register():
    __sets['VGGnet_train'] = networks.VGG16(is_train=True)
    __sets['VGGnet_test'] = networks.VGG16(is_train=False)
    __sets['Resnet50_train'] = networks.Resnet50(is_train=True)
    __sets['Resnet50_test'] = networks.Resnet50(is_train=False)


def get_network(name):
    """Get a network by name."""
    _register()
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    net = __sets[name].setup()
    return net


def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
