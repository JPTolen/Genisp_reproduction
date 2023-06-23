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

The paper proposes to train an ISP pipeline, which they call GenISP. The GenISP pipeline adapts raw image data into a representation that is suitable for any pre-trained object detector. They map the camera sensor specific color space to a device independent color space. Therefore, the model can better generalize to unseen cameras because it operates on data in a device-independent color space.

In contrast, we propose to take advantage of Color
Space Transformation (CST) matrices that map the sensorspecific color space (raw-RGB) to a device-independent
color space (CIE XYZ)
This is that incorporates color space transformation to a device-independent color space

Thus, an object detector does not require any fine-tuning to specific sensor data in an individual camera.


The method avoids fine-tuning to specific sensor data by leveraging image metadata along with the RAW image files to transform the image to optimize object detection

## Preprocessing of Raw Images

## Image Correction

## Training
[use of retinanet(?)]