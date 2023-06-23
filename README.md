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

The paper proposes to train an ISP pipeline. This neural ISP adapts raw image data into a representation that is optimal for machine cognition so that a pre-trained object detector can be used without any need for fine-tuninig or re-training

## Preprocessing of Raw Images

## Image Correction

## Training
[use of retinanet(?)]