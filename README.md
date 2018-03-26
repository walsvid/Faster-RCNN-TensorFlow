# Faster-RCNN-TensorFlow
[![Language python](https://img.shields.io/badge/python-3.5%2C%203.6-blue.svg)](https://www.python.org) [![TensorFlow](https://img.shields.io/badge/tensorflow-1.3-orange.svg)](https://www.tensorflow.org) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) 

This is an experimental TensorFlow implementation of Faster-RCNN, based on the work of [smallcorgi](https://github.com/smallcorgi/Faster-RCNN_TF) and [rbgirshick](https://github.com/rbgirshick/py-faster-rcnn). I have converted the code to python3, future python2 will stop supporting it, and using python3 is an irreversible trend. And I deleted some useless files and legacy caffe code.

What's New:
- [x] Convert code to Python3
- [x] Make compile script adapt gcc-5
- [x] Visualization using tensorboard
- [x] PSRoI Pooling
- [ ] OHEM a.k.a Online Hard Example Miniing
- [ ] RoI Align
- [ ] More basenet

Reference:
### Acknowledgments: 

1. [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

2. [Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)

3. [ROI pooling](https://github.com/zplizzi/tensorflow-fast-rcnn)

4. [TFFRCNN](https://raw.githubusercontent.com/CharlesShang/TFFRCNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```bash
  git clone https://github.com/walsvid/Faster-RCNN-TensorFlow.git
  ```

2. Build the Cython modules
    ```bash
    ROOT = Faster-RCNN-TensorFlow
    cd ${ROOT}/lib
    make
    ```
 Compile cython and roi_pooling_op, you may need to modify make.sh for your platform.

 GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |


### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

Download model training on PASCAL VOC 2007  [[Google Drive]](https://drive.google.com/file/d/0ByuDEGFYmWsbZ0EzeUlHcGFIVWM/view)


To run the demo execute:
```bash
cd $ROOT
python ./tools/demo.py --model model_path
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Train model
1. Download the training, validation, test data and VOCdevkit

    ```bash
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    ```

2. Extract all of these tars into one directory named `VOCdevkit`

    ```bash
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_08-Jun-2007.tar
    ```

3. It should have this basic structure

    ```bash
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
    # ... and several other directories ...
    ```

4. Create symlinks for the PASCAL VOC dataset

    ```bash
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    
5. Download pre-trained ImageNet models

   Download the pre-trained ImageNet models [[Google Drive]](https://drive.google.com/file/d/0ByuDEGFYmWsbNVF5eExySUtMZmM/view?usp=sharing)
   
    ```bash
    mv VGG_imagenet.npy $FRCN_ROOT/data/pretrain_model/VGG_imagenet.npy
    ```

6. Run script to train and test model
    ```bash
    cd $FRCN_ROOT
    ./experiments/scripts/faster_rcnn_end2end.sh $DEVICE $DEVICE_ID VGG16 pascal_voc
    ```
  DEVICE is `cpu` or `gpu`

### Visualization
Just execute `tensorboard`.
```bash
tensorboard --logdir=./logs
```

Release:
`v0.3.0`