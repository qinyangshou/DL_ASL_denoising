# DL_ASL_denoising
Update 06-04-24:
LFS is disabled. The data is now uploaded to Google drive:
https://drive.google.com/drive/folders/1TjVdJpD6zLgkhZDFKy1tQ70hy1_0hWeg?usp=drive_link

## This project is developed by Lab of Functional MRI Technology (LOFT, www.loft-lab.org)

## Work can be cited as:
Shou, Qinyang, Chenyang Zhao, Xingfeng Shao, Kay Jann, Hosung Kim, Karl G. Helmer, Hanzhang Lu, and Danny JJ Wang. "Transformer‐based deep learning denoising of single and multi‐delay 3D arterial spin labeling." Magnetic resonance in medicine 91, no. 2 (2024): 803-818.

## Descriptions
This repository contains some example training and evaluation data (under ./Data)
To train a new model: run model_train.py, change the model type and model parameters in the code

The "./Model" folder contains a collection of models compared in the paper including: 

SwinIR(Liang, Jingyun, et al. "Swinir: Image restoration using swin transformer." Proceedings of the IEEE/CVF international conference on computer vision. 2021.)

DWAN(Xie, Danfeng, et al. "Denoising arterial spin labeling perfusion MRI with deep machine learning." Magnetic resonance imaging 68 (2020): 95-105.)

EDSR(Lim, Bee, et al. "Enhanced deep residual networks for single image super-resolution." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017.)

TransUNet(Chen, Jieneng, et al. "Transunet: Transformers make strong encoders for medical image segmentation." arXiv preprint arXiv:2102.04306 (2021).)

UNETR(Hatamizadeh, Ali, et al. "Unetr: Transformers for 3d medical image segmentation." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2022.)

To evaluate the trained model: run model_evaluation.py, it will produce the output in Model_predictions
