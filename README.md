# Faster-RCNN-TensorFlow
This is an experimental TensorFlow implementation of Faster-RCNN, based on the work of [smallcorgi](https://github.com/smallcorgi/Faster-RCNN_TF) and [rbgirshick](https://github.com/rbgirshick/py-faster-rcnn). I have converted the code to python3, future python2 will stop supporting it, and using python3 is an irreversible trend. And I deleted some useless files and legacy caffe code.

What's New:
- [x] Convert code to Python3
- [x] Make script adapt gcc-5
- [ ] OHEM a.k.a Online Hard Example Miniing
- [ ] ROI Align
- [ ] More basenet

Reference:
### Acknowledgments: 

1. [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

2. [Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)

3. [ROI pooling](https://github.com/zplizzi/tensorflow-fast-rcnn)

4. [TFFRCNN](https://raw.githubusercontent.com/CharlesShang/TFFRCNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  git clone https://github.com/walsvid/Faster-RCNN-TensorFlow.git
  ```

2. Build the Cython modules
    ```Shell
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

To run the demo
```shell
cd $ROOT
python ./tools/demo.py --model model_path
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

Release:
`v0.2.0`