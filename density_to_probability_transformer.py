# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:05:51 2021

Transform density images, produced by the Cell Density Counting workflow of
ilastik, into probability maps and save them.

@author: imbroscb
"""

import h5py  # to install
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

# %% define the paths

# enter the path where the density images are stored
path_density = '...'

# enter the path where the produced probability maps should be saved
path_probability = '...'

# %% open, transform and save the images

# iterate over the files in the path indicated by path_density
for i in range(len(listdir(path_density))):

    filename = listdir(path_density)[i]
    with h5py.File(path_density + filename, "r") as f:

        # list all groups
        groups = list(f.keys())[0]

        # get the data
        data = list(f[groups])

    # reshape the data
    image = np.zeros((len(data), len(data[0])))
    for j in range(len(data)):
        image[j, :] = data[j].reshape(len(data[j]),)

    # print image name, minimum and and maximum density value
    print('Image_name: ', filename)
    print('Min from density: ', np.min(image))
    print('Max from density: ', np.max(image))

    # transform the density in probability (0-1)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # print image name, minimum and and maximum probability values
    print('Min from probability: ', np.min(image))
    print('Max from probability: ', np.max(image))
    print('---------------')

    # plot the image (comment the next two lines if plotting is not wished)
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # update the image name (filename[:-16] was chosen for changing the names
    # of the density images produced by ilastik, by removing from the name:
    # 'Probability.h5' and adding 'real_probability.h5', instead)
    name = filename[:-16] + 'real_probability.h5'

    # save and close the image
    hf = h5py.File(path_probability + name, 'w')
    hf.create_dataset('dataset_1', data=image)
    hf.close()
