# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 09:43:44 2021

Count the vesicles (objects) in the segmentation map produced by
the Object Classification [Inputs: Raw Data, Pixel Prediction Map] workflow
of ilastik.

Compare the vesicles predicted by the algorithm ilastik with the manually
detected vesicles, calculating the number of true positive (tp),
false positive (fp) and false negative (fn).

Note that the images given to ilastik were already rescaled, therefore
the segmenation maps will have a pixel size of 2.27 nm.

@author: imbroscb
"""

import h5py
from sklearn.cluster import KMeans
from scipy import ndimage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from sklearn.metrics.pairwise import euclidean_distances
import cv2
import copy
from pathlib import Path

# %% define the path with the images and the pixel size

# enter the path where the images to test are stored
path_images = '...'

# change each time according to the pixel size of the images to test
pixel_size = 2.27

# iterate over the files in the path indicated by path_images
for e, img_name in enumerate(listdir(path_images)):
    print(e, img_name)

del e, img_name

# %% get the filename and open and rescale the image

# enter the filename of the image directly (option 1)
# filename = '...'

# get the filename by specifying the desired index (option 2)
idx = 0
filename = listdir(path_images)[idx]

# open the desired image, and rescale it
image_name = Path(filename).stem
img = Image.open(path_images + filename)
np_img = np.array(img)
new_shape0 = round(np_img.shape[0] / 2.27 * pixel_size)
new_shape1 = round(np_img.shape[1] / 2.27 * pixel_size)
np_img = cv2.resize(np_img, (new_shape1, new_shape0))

del filename, img, new_shape0, new_shape1

# %% define the path with the segmenation maps

# enter the path where the segmentation maps are stored
path_segment = '...'

# iterate over the files in the path indicated by path_segment
for e, segment_name in enumerate(listdir(path_segment)):
    print(e, segment_name)

# %% get the filename and open the segmentiation map

# enter the filename of the segmentation map directly (option 1)
# filename_segment = '...'

# get the filename by specifying the desired index (option 2)
idx = 0
filename_segment = listdir(path_segment)[idx]

# open the segmentation map
with h5py.File(path_segment + filename_segment, "r") as f:

    # list all groups
    groups = list(f.keys())[0]

    # get the data
    data = list(f[groups])

    # reshape the data
    image_segment = np.zeros((len(data), len(data[0])))
    for j in range(len(data)):
        image_segment[j, :] = data[j].reshape(len(data[j]),)

# in the segmentation image:
# pixel values == 0 represent the background
# pixel values == 1 are part of the predicted objects
# pixel_values == 2 represent false positive
# therefore, to isolate the predicted objects pixels values == 2 are set to 0

for i in range(image_segment.shape[0]):
    for j in range(image_segment.shape[1]):
        if image_segment[i, j] == 2:
            image_segment[i, j] = 0

# plot the image and the segmentation map
plt.subplot(1, 2, 1)
plt.imshow(np_img, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image_segment, cmap='gray')
plt.axis('off')

del f, groups, data, j, i

# %% define the path with the annotations

# enter the path where the human annotations (labels) are stored
path_labels = '...'

# iterate over the files in the path indicated by path_labels
for e, labels in enumerate(listdir(path_labels)):
    print(e, labels)

del e, labels

# %% get the filename(s) and open the human annotations

# enter the filenames of the human annotations for the image to test directly
# (option 1), the number filenames depends on how many people did the
# annotations
# filename1 = '...'
# filename2 = '...'

# get the filenames by specifying the desired index (option 2)
idx1 = 15
idx2 = 16
filename1 = listdir(path_labels)[idx1]
filename2 = listdir(path_labels)[idx2]

print('Image to test: ', image_name)
print('Label1: ', Path(filename1).stem)
print('Label2: ', Path(filename2).stem)

# load annotations corresponding to filename1
# this cell may be different depending on how the annotations are saved
labels_1 = pd.read_csv(path_labels + filename1, index_col=1)

# alternative
# labels_1 = pd.read_csv(path_labels + filename1, index_col=1,
# delimiter='\t')

x_coord_manual_1 = []
y_coord_manual_1 = []
for idx, row in labels_1.iterrows():
    # the two following lines assume that the annotations are saved in pixels
    x_coord_manual_1.append(int(round(row['X'] / 2.27 * pixel_size)))
    y_coord_manual_1.append(int(round(row['Y'] / 2.27 * pixel_size)))

# eliminate double or multiple sets of coordinates
# this can happen if, by mistake, the same data point was clicked twice or
# multiple times (otherwise the same vesicle would be detected twice or
# multiple times)
coord = zip(x_coord_manual_1, y_coord_manual_1)
coords_unique = []
for i in coord:
    if i not in coords_unique:
        coords_unique.append(i)

x_coord_manual_1, y_coord_manual_1 = zip(*coords_unique)

# load annotations corresponding to filename2
# this cell may be different depending on how the annotations are saved
labels_2 = pd.read_csv(path_labels + filename2, index_col=1)

# alternative
# labels_2 = pd.read_csv(path_labels + filename1, index_col=1,
# delimiter='\t')

x_coord_manual_2 = []
y_coord_manual_2 = []
for idx, row in labels_2.iterrows():
    # the two following lines assume that the annotations are saved in pixels
    x_coord_manual_2.append(int(round(row['X'] / 2.27 * pixel_size)))
    y_coord_manual_2.append(int(round(row['Y'] / 2.27 * pixel_size)))

# eliminate double or multiple sets of coordinates
# this can happen if, by mistake, the same data point was clicked twice or
# multiple times (otherwise the same vesicle would be detected twice or
# multiple times)
coord = zip(x_coord_manual_2, y_coord_manual_2)
coords_unique = []
for i in coord:
    if i not in coords_unique:
        coords_unique.append(i)

x_coord_manual_2, y_coord_manual_2 = zip(*coords_unique)

del filename1, filename2, i, idx, row, coord, coords_unique

# %% estimate the detected vesicles from the segmentation map (from ilastik)
# and get their coordinates

labelarray, count = ndimage.measurements.label(image_segment)
x_coord_ilastik = np.zeros((count))
y_coord_ilastik = np.zeros((count))

# iterate over the found objects (labelarray)
for i in range(count):
    x, y = np.where(labelarray == i + 1)
    temp = []
    if type(x) == int:
        temp.append(x, y)
    else:
        for j in range(len(x)):
            temp.append((x[j], y[j]))
    kmeans = KMeans(n_clusters=1).fit(temp)
    x_coord_ilastik[i] = int(round(kmeans.cluster_centers_[0][1]))
    y_coord_ilastik[i] = int(round(kmeans.cluster_centers_[0][0]))

del labelarray, count, i, x, y, temp, j, kmeans

# plot image + annotations
plt.figure()
plt.imshow(np_img[:, :], cmap='gray')
plt.scatter(x_coord_manual_1, y_coord_manual_1, s=8, color='red',
            label='manual 1')
plt.scatter(x_coord_manual_2, y_coord_manual_2, s=8, color='blue',
            label='manual 2')
plt.scatter(x_coord_ilastik, y_coord_ilastik, s=30, color='white', marker='+',
            label='ilastik')
plt.legend()
plt.axis('off')

# %% check the performance of the algorithm using the coordinates from the
# manual annotations as ground truth

algorithm = list(zip(x_coord_ilastik, y_coord_ilastik))
# run first using option 1 only and then using opion 2 only
# option 1
manual = list(zip(x_coord_manual_1, y_coord_manual_1))
# option 2
# manual = list(zip(x_coord_manual_2, y_coord_manual_2))

# creating pairs
# single_auto will contain false positive (after the loop)
single_auto = copy.deepcopy(algorithm)
single_auto_iter = copy.deepcopy(algorithm)
pairs_manual = []
already_assigned = []

# iterate over the annotations
for m in manual:
    min_distance = 999999
    closest_prediction = 0

    # iterate over the results from algorithm
    for s in single_auto_iter:
        xm = m[0]
        ym = m[1]
        xs = s[0]
        ys = s[1]
        M = np.array([xm, ym]).T
        S = np.array([xs, ys]).T
        distance = euclidean_distances(M.reshape(1, -1), S.reshape(1, -1))

        if distance < min_distance:
            if s not in already_assigned:
                min_distance = distance
                closest_prediction = s

    if min_distance <= np.sqrt(9**2 + 9**2):
        single_auto.remove(closest_prediction)
        pairs_manual.append([m, closest_prediction])
        already_assigned.append(closest_prediction)

# single_manual will contain false negative (after the loop)
single_manual = copy.deepcopy(manual)
single_manual_iter = copy.deepcopy(manual)
pairs_auto = []
already_assigned = []

# iterate over the results from algorithm
for a in algorithm:
    min_distance = 999999
    closest_prediction = 0

    # iterate over the annotations
    for s in single_manual_iter:
        xa = a[0]
        ya = a[1]
        xs = s[0]
        ys = s[1]
        A = np.array([xa, ya]).T
        S = np.array([xs, ys]).T
        distance = euclidean_distances(A.reshape(1, -1), S.reshape(1, -1))

        if distance < min_distance:
            if s not in already_assigned:
                min_distance = distance
                closest_prediction = s

    if min_distance <= np.sqrt(9**2 + 9**2):
        single_manual.remove(closest_prediction)
        pairs_auto.append([a, closest_prediction])
        already_assigned.append(closest_prediction)

# print the outcome
print('tot manual: ', len(manual))
print('tot algorithm: ', len(x_coord_ilastik))
print('tot pairs auto and manual (tp): ', len(pairs_auto), len(pairs_manual))
print('tot single auto (fp): ', len(single_auto))
print('tot single manual (fn): ', len(single_manual))

del m, a, s, A, S, xm, ym, xa, ya, xs, ys, distance, min_distance
del closest_prediction
