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
path2='Singapore/'
path3='Amsterdam/'


def image_processing(file_path):
    data_path=os.listdir(file_path)
    flair_dataset=[]
    mask_dataset=[]
    for i in data_path:
    #print(i)
        img_path=os.path.join(file_path, i,'pre')
        mask_path=os.path.join(file_path,i)

        for name in glob.glob(img_path+'/FLAIR*'):
            flair_img=image.load_img(name)
            flair_data=flair_img.get_data()
            flair_data=np.transpose(flair_data, (1,0,2))
            flair_resized = cv2.resize(flair_data, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
            flair_dataset.append(flair_resized)

        for name in glob.glob(mask_path+'/wmh*'):
            mask_img=image.load_img(name)
            mask_data=mask_img.get_data()
            mask_data=np.transpose(mask_data, (1,0,2)) #transpose so orientation matches nilearn plot
            mask_resized=cv2.resize(mask_data, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
            ret, mask_binary=cv2.threshold(mask_resized,0.6,1,cv2.THRESH_BINARY)
            mask_dataset.append(mask_binary)

    return flair_dataset, mask_dataset


utrecht_flair, utrecht_mask=image_processing(path1)
singapore_flair, singapore_mask=image_processing(path2)
amsterdam_flair, amsterdam_mask=image_processing(path3)

utrecht_flair=np.array(utrecht_flair)
utrecht_mask=np.array(utrecht_mask)
singapore_flair=np.array(singapore_flair)
singapore_mask=np.array(singapore_mask)
amsterdam_flair=np.array(amsterdam_flair)
amsterdam_mask=np.array(amsterdam_mask)

print(utrecht_flair.shape)
print(amsterdam_flair.shape)
print(amsterdam_mask.shape)


np.save('utrecht_flair(128).npy', utrecht_flair)
np.save('utrecht_mask(128).npy', utrecht_mask)

np.save('singapore_flair(128).npy', singapore_flair)
np.save('singapore_mask(128).npy', singapore_mask)

np.save('amsterdam_flair(128).npy', amsterdam_flair)
np.save('amsterdam_mask(128).npy', amsterdam_mask)
