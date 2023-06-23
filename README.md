# Reproduction Paper of: 'GenISP: Neural ISP for Low-Light Machine Cognition'

Authors: 
- Thijs Exterkate 
- Tim Hoevenaar
- Jesse Tholen

This reproduction was part of the course CS4240 Deep Learning at Delft University of Technology

The original paper can be found at: https://arxiv.org/abs/2205.03688

## Introduction
Most studies in the computer vision community use image data processed by an in-camera traditional Image Signal Processor (ISP). However, traditional ISP pipelines are typically not tuned to process low-light data.

Hong et al: for low light conditions, object detectors using raw sensor data perform better than detectors using data that is processed by a traditional ISP pipelin. 

The paper proposes to train an ISP pipeline, which they call GenISP. The GenISP pipeline first operates on the raw RGB image, which is a sensor-specific color space to one camera. The sensor-specific color space is mapped to a device-independent color space. This enables the model to better generalize to unseen camera sensors because it operates on the image in a device-independent color space. The device-independent image is then processed through three neural network modules: ConvWB, ConvCC and Shallow ConvNet. This whole pipeline outputs an image that is optimized for any off-the-shelf object detector. Thus, an object detector does not require any fine-tuning to camera specific sensor data.


## Preprocessing of Raw Images

## Image Correction

## Training
[use of retinanet(?)]