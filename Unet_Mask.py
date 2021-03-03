#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:30:25 2020

@author: neamul
"""

import os
import numpy as np
import nibabel as nib
import time
import pandas as pd
import matplotlib.pyplot as plt

start =time.time()

filename=os.path.join("/home/neamul/Desktop/DataN/ZLST1957_AttenCT.nii")
th=0.105
def U_netMask(filename):
    img=nib.load(filename)
    aff=img.affine
    data=img.get_fdata(dtype='float32').squeeze()
    data=data.squeeze()
    #plt.imshow(data[90])
    data[data<th]=0
    data[data>=th]=1
    imgNew=nib.Nifti1Image(data,aff)
    imgname=input ("Enter the unet-Mask img Name:  ")
    imgNew.to_filename(os.path.join(imgname+'UMask.nii'))
    
U_netMask(filename)
#filename=os.path.join("/home/neamul/thesis_git/Unet_mask/AlobUMask.nii")
#img=nib.load(filename)
#aff=img.affine
#data=img.get_fdata(dtype='float32').squeeze()
#plt.imshow(data[90])
