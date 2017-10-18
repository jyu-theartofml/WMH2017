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

img_row=128
img_col=128
def image_processing(file_path):
    data_path=os.listdir(file_path)
    flair_dataset=[]
    mask_dataset=[]
    for i in data_path:
        img_path=os.path.join(file_path, i,'pre')
        mask_path=os.path.join(file_path,i)

        for name in glob.glob(img_path+'/FLAIR*'):
            flair_img=image.load_img(name)
            flair_data=flair_img.get_data()
            flair_data=np.transpose(flair_data, (1,0,2))
            flair_resized = cv2.resize(flair_data, dsize=(img_row,img_col), interpolation=cv2.INTER_CUBIC)
            flair_dataset.append(flair_resized)
        #perform some image augmentation (use same parameters for mask for deterministic augmentation)
        #rotate
            M1 = cv2.getRotationMatrix2D((img_row/2,img_col/2),90,1)
            flair_rotate= cv2.warpAffine(flair_resized,M1,(img_row,img_col))
            flair_dataset.append(flair_rotate)
        #shearing
            pts1 = np.float32([[50,50],[100,100],[50,200]])
            pts2 = np.float32([[10,100],[100,100],[100,210]])
            M2 = cv2.getAffineTransform(pts1,pts2)
            flair_shear= cv2.warpAffine(flair_resized,M2,(img_row,img_col))
            flair_dataset.append(flair_shear)
        #zoom
            pts3 = np.float32([[45,48],[124,30],[50,120],[126,126]])
            pts4= np.float32([[0,0],[128,0],[0,128],[128,128]])
            M3 = cv2.getPerspectiveTransform(pts3,pts4)
            flair_zoom = cv2.warpPerspective(flair_resized,M3,(img_row,img_col))
            flair_dataset.append(flair_zoom)

        # perform same transformation on mask files
        for name in glob.glob(mask_path+'/wmh*'):
            mask_img=image.load_img(name)
            mask_data=mask_img.get_data()
            mask_data=np.transpose(mask_data, (1,0,2)) #transpose so orientation matches nilearn plot
            mask_resized=cv2.resize(mask_data, dsize=(img_row,img_col), interpolation=cv2.INTER_CUBIC)
            ret, mask_binary=cv2.threshold(mask_resized,0.6,1,cv2.THRESH_BINARY)
            mask_dataset.append(mask_binary)
            #need to run binary threshold again after augmentation
            mask_rotate= cv2.warpAffine(mask_binary,M1,(img_row,img_col))
            ret, mask_rotate=cv2.threshold(mask_rotate,0.6,1,cv2.THRESH_BINARY)
            mask_dataset.append(mask_rotate)
                
            mask_shear= cv2.warpAffine(mask_binary,M2,(img_row,img_col))
            ret, mask_shear=cv2.threshold(mask_shear,0.6,1,cv2.THRESH_BINARY)
            mask_dataset.append(mask_shear)
                
            mask_zoom= cv2.warpPerspective(mask_binary,M3,(img_row,img_col))
            ret, mask_zoom=cv2.threshold(mask_zoom,0.6,1,cv2.THRESH_BINARY)
            mask_dataset.append(mask_zoom)
                

    flair_array=np.array(flair_dataset)
    mask_array=np.array(mask_dataset)
    return flair_array, mask_array


utrecht_flair, utrecht_mask=image_processing(path1)
amsterdam_flair, amsterdam_mask=image_processing(path2)
singapore_flair, singapore_mask=image_processing(path3)

np.save('utrecht_flair(128)aug.npy', utrecht_flair)
np.save('utrecht_mask(128)aug.npy', utrecht_mask)

np.save('amsterdam_flair(128)aug.npy', amsterdam_flair)
np.save('amsterdam_mask(128)aug.npy', amsterdam_mask)

np.save('singapore_flair(128)aug.npy', singapore_flair)
np.save('singapore_mask(128)aug.npy', singapore_mask)
