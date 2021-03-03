#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:54:40 2020

@author: neamul
"""
import os
import numpy as np
import nibabel as nib
import time
import pandas as pd
import matplotlib.pyplot as plt

def shapee(filename):
    img=nib.load(filename)
    data=img.get_fdata()
    return (data.shape)


def U_netMask(filename):
    img=nib.load(filename)
    aff=img.affine
    data=img.get_fdata().squeeze()
    print (data.shape)
   # data=data.reshape(data.shape[0],192,192) 
    imgNew=nib.Nifti1Image(data,aff)
    imgname=input ("Enter the unet-Mask img Name:  ")
    imgNew.to_filename(os.path.join('zlst_'+imgname+'.nii'))
    
filename=os.path.join("/home/neamul/thesis_git/testresult/New/Combination4/Zlst/zlst_pred4Mask.nii")

U_netMask(filename)


a=os.path.join("/home/neamul/thesis_git/zlst_pred4Mask.nii")
shapee(a)
