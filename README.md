# Sink_InSAR_CNN_LSTM
Code name: X-BBox
Author: Anurag Kulshrestha
Purpose: Code for extracting training tiles and using U-Net and CNN-LSTM to learn
and classify sinkhole related fringe patterns in wrapped interferograms.
Author: Anurag Kulshrestha
Date created: 03-07-2022
Last modified: 02-02-2023

Info: 
1. The traning datasets are created using the XBBox method defined in function: make_training_tiles.
2. The training samples and labels are stored with file names beginnnig with 'trainX_' and 'train_Y' respectively.
3. The models are trained using TSx spotlight data, and tested on Sentinel-1 data.
4. For interferometric processing of TSx-spotlight data, please see TSx_spotlight.py
5. Functions for reading doris derived datasets are written in the doris_read_data.py file
6. The models are declared in models.py


Abstract of related paper:

Many sinkholes are well characterized by elliptical Gaussian-shaped fringes
in wrapped Synthetic Aperture Radar (SAR) interferograms. Detection of
these patterns over large sinkhole-prone areas remains challenging, especially
due to the unavailability of training datasets. Over the past few years, Con-
volution Neural Networks (CNN) have proved to be powerful to learn and
detect spatial patterns in images. Similarly, Recurrent Neural Networks
(RNN), such as Long Short Term Memory (LSTM), have the capability of
learning hidden patterns in multi-temporal sequences. As a synergy, this
study proposes the use of spatial modelling with U-Net and spatio-temporal
modelling with Convolutional Neural Network-LSTM (CNN-LSTM). We ex-
tract training datasets from real SAR interferograms created using X-band
TerraSAR-X spotlight SAR datasets of resolution 0.23×0.94 m and augment
the data in scale-space using a novel method which we call Extract using
Bounding Boxes (XBBox). Using transfer learning, we test our trained
models on real C-band Sentinel-1 datasets of 20 × 4 m resolution. This
was done over a study site near Wink, Texas, USA, where large subsidence
was recorded around a sinkhole of ∼500 m diameter in 2015. We used 12
TerraSAR-X and 15 Sentinel-1 SAR images separately acquired between
April-2015 and March-2016. The results show that the sinkhole site was
detected successfully using U-Net with a weighted average F1-score of 0.98.
CNN-LSTM showed relatively lower, but still high accuracy with a weighted
average F1-score of 0.92. It was seen that sinkhole detection probability in-
creased with the increase of temporal epochs of the input dataset.
