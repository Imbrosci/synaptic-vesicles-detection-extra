# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:05:51 2021

Transform density images, produced by the Cell Density 
Counting workflow of ilastik, into probability maps and save them 

@author: imbroscb
"""

#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from os import listdir
from scipy import ndimage
import copy

#%% 

root_density= '...' # root where the density images are stored

for i in range(len(listdir(root_density))):
   
    filename=listdir(root_density)[i]
 
    with h5py.File(root_density+filename, "r") as f:

        # List all groups
        a_group_key = list(f.keys())[0]
    
        # Get the data
        data = list(f[a_group_key])
        
        image=np.zeros((len(data),len(data[0])))

    for j in range(len(data)):
        image[j,:]=data[j].reshape(len(data[j]),)
    
    print('Image_name',filename)
    print('Max: ', np.max(image))

    image=(image-np.min(image))/(np.max(image)-np.min(image))
    plt.imshow(image)
    print('Max: ', np.max(image))
    print('---------------')


    name= filename[:-16] + 'real_probability.h5' # the selected indeces [:-16] serve 
                                                 # to change the density image name 
                                                 # (removing Probabilities.h5' and
                                                 # changing it to 'real_probabilities.h5'                                                  

    root_probability= '...' # root where the produced probability maps should be stored
    hf = h5py.File(root_probability + name, 'w')
    hf.create_dataset('dataset_1', data=image)
    hf.close() 
    
    
