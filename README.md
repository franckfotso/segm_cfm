# segm_cfm

Implementation of Convolutional Feature Masking (CFM)

Created by Romuald FOTSO

## Introduction:

CFM, which stands for Convolutional Feature Masking. This project is an attempt to implement the segmentation approach [**Convolutional Feature Masking for Joint Object and Stuff Segmentation**](https://arxiv.org/abs/1412.1283)). In the paper, they propose to use shape information via masking convolutional features. The CNN features are masked out from the convolutional maps and used to train classifiers for recognition. We have setup an end-end framework suitable for data pre-processing, training, test, evalutation and visualization. There are several proposals involved in this project: Selective Search, Multiscale Combinatorial Group ([**MCG**](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/resources/MCG_CVPR2014.pdf)) and Convolutional Oriented Boundaries ([**COB**](http://arxiv.org/abs/1701.04658)).


## Purposes:

Our main goals are:
  * Propose a framework to handle a custom dataset for a segmentation task
  * Propose a framework to handle a range bounding boxes or segments proposals such as Selective search, MCG or COB.
  * Extract segments & rois from VOC Pascal Segmentation schema (if there is no SBD data).
  * Generate selective search proposals (segments & rois) from images set.
  * Implement a data generator et layers required to load segments/masks in Convolutional Network.
  * Implement key layers involved CFM approach: ROIPooling & MaskPooling.
  * Perform some post-processing (mask weighing & voting) required for the mask rendering.
  * Test the CFM approach over 4 types of network config: 
     *  T1 (bbox cls + mask cls)
     *  T2 (bbox cls + mask cls + bbox pred)
     *  T3 (bbox cls + mask cls + bbox pred + mask pred) => our own layers
     *  T4 (bbox cls + mask cls + bbox pred + mask pred) => author's layers (CFM)

## Datasets:

We have use the VOC2012 dataset, parsed as SBD (Semantic Boundaries Dataset). In addition, we also the MCG and COB proposals related.
Required files are available here:
  * [SBD (VOC2012): train+val (12031 images)](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
  * [SBD (VOC2012): train+val (12031 MCG proposals)](https://data.vision.ee.ethz.ch/jpont/mcg/MCG-SBD-trainval-proposals.tgz)
  * [SBD (VOC2012): train+val (12031 COB proposals)](https://data.vision.ee.ethz.ch/kmaninis/share/COB/Precomputed/COB-SBD-trainval-proposals.tgz)

Those datasets were use train model over CNN nets: AlexNet, ZF & VGG16. We have explored many CNN nets in the way to see how flexible this approach is. Futhermore, we have provide all prototxt required so that you can test this project in a wide range of Nvidia GPU card ( > 2 Gb)

## Hardwares/Softwares:

OS: Ubuntu 16.04 64 bit
    GPU: Nvidia GTX 950M 4G
    Cuda 8.0
    CuDNN 4.0.7
    Python 2.7.12
    OpenCV 3.1.0

## Prerequisites:

## Installation:

## Experiments:

## Our results

## Contact

Please feel free to leave suggestions or comments to Romuald FOTSO (romyny9096@gmail.com)
