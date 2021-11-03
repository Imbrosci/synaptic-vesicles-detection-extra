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

# %% define the directories

# enter the directory where the density images are stored
directory_density = '...'

# enter the directory where the produced probability maps should be saved
directory_probability = '...'

# %% open, transform and save the images

# iterate over the files in the directory indicated by directory_density
for i in range(len(listdir(directory_density))):

    filename = listdir(directory_density)[i]
    with h5py.File(directory_density + filename, "r") as f:

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
    hf = h5py.File(directory_probability + name, 'w')
    hf.create_dataset('dataset_1', data=image)
    hf.close()
