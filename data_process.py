import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd

from nilearn import plotting
from nilearn import image
import cv2
import os
import glob

path1='Utrecht/'
path2='Amsterdam'
path3='Singapore/


def image_processing(file_path):
    data_path=os.listdir(file_path)
    flair_dataset=[]
    mask_dataset=[]
    t1_dataset=[]
    for i in data_path:
        img_path=os.path.join(file_path, i,'pre')
        mask_path=os.path.join(file_path,i)

        for name in glob.glob(img_path+'/FLAIR*'):
            flair_img=image.load_img(name)
            flair_data=flair_img.get_data()
            flair_data=np.transpose(flair_data, (1,0,2))
            flair_resized = cv2.resize(flair_data, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
            flair_dataset.append(flair_resized)

        for name in glob.glob(img_path+'/T1*'):
            t1_img=image.load_img(name)
            t1_data=t1_img.get_data()
            t1_data=np.transpose(t1_data, (1,0,2))
            t1_resized = cv2.resize(t1_data, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
            t1_dataset.append(t1_resized)
        
        for name in glob.glob(mask_path+'/wmh*'):
            mask_img=image.load_img(name)
            mask_data=mask_img.get_data()
            mask_data=np.transpose(mask_data, (1,0,2)) #transpose so orientation matches nilearn plot
            mask_resized=cv2.resize(mask_data, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
            ret, mask_binary=cv2.threshold(mask_resized,0.6,1,cv2.THRESH_BINARY)
            mask_dataset.append(mask_binary)

    mri_array=np.stack((np.array(flair_dataset), np.array(t1_dataset)), axis=-1)
    mask_array=np.array(mask_dataset)
    return mri_array, mask_array


utrecht_mri, utrecht_mask=image_processing(path1)
amsterdam_mri, amsterdam_mask=image_processing(path2)
singapore_mri, singapore_mask=image_processing(path3)


np.save('utrecht_mri(128).npy', utrecht_mri)
np.save('utrecht_mask(128).npy', utrecht_mask)

np.save('amsterdam_mri(128).npy', amsterdam_mri)
np.save('amsterdam_mask(128).npy', amsterdam_mask)

np.save('singapore_mri(128).npy', singapore_mri)
np.save('singapore_mask(128).npy', singapore_mask)
