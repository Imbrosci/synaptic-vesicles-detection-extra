# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 09:43:44 2021

Count the vesicles (objects) in the segmentation map produced by 
the Object Classification [Inputs: Raw Data, Pixel Prediction Map] workflow 
of ilastik and compare the vesicles predicted by ilastik with the manually 
detected vesicles, calculating the number of True Positives (TP), 
False Positives (FP) and False Negatives (FN).
Note that the images given to ilastik were already rescaled, therefore
the segmenation maps will have a pixel size==2.27 nm
@author: imbroscb
"""

#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy import ndimage
import copy
from PIL import Image
import pandas as pd
from random import randrange
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

#%% get the root where the images are stored

root_img= '...'
pixel_size=2.27 # change each time 

#%% get the filename_img, option 1

filename_img = '...' # enter the filename directly

#%% get the filename, option 2

for i in range(len(listdir(root_img))):
    print(i,listdir(root_img)[i])
del i

#%% choose the desired index (from option 2)

filename_img=listdir(root_img)[0] 

#%% open the desired image

image_name=filename_img[:-4] # -5 if the number of characters of the extension is 4
img=Image.open(root_img+filename_img)
np_img=np.array(img)
print(filename_img)

#%% if needed, resize the image so that each pixel is 2.27 nm (you have to know the resolution)

#example
img=cv2.resize(np_img,(1542,1542)) # the number depends on the resolution of the image
np_img=np.array(img)

#%% get the root where the segmenation maps are stored

root_seg= '...'

#%% get the filename_seg, option 1

filename_seg = '...' # enter the filename directly

#%% get the filename, option 2

for i in range(len(listdir(root_seg))):
    print(i,listdir(root_seg)[i])
del i

#%% choose the desired index (from option 2)

filename_seg=listdir(root_seg)[0]  

#%% get the root where human annotations (with the coordinates of the detected vesicles) are stored

root_label= '...' 


#%% get the filename_labels, option 1

# in case two or more people did the annotations there may be two or more filename_labels
filename_label_1 = '...' # enter the filename directly
filename_label_2= '...' # enter the filename directly

#%% get the filename_label, option 2

for i in range(len(listdir(root_label))):
    print(i,listdir(root_label)[i])
del i

#%% choose the desired indeces (from option 2)

filename_label_1=listdir(root_label)[0] 
filename_label_2=listdir(root_label)[1] 

#%% load annotation from person 1
# this cell may be different depending on how the annotations are done/saved!

#option 1
labels_1=pd.read_csv(root_label+filename_label_1,index_col=1)#,delimiter='\t')

#option 2
#labels = pd.read_csv(root_label+filename_label,index_col=1,delimiter='\t')

x_coord_manual_1=[]
y_coord_manual_1=[]
for idx,row in labels_1.iterrows():
    x_coord_manual_1.append(int(round(row['X']/2.27*pixel_size))) 
    y_coord_manual_1.append(int(round(row['Y']/2.27*pixel_size))) 
    # in case the annotations are not saved in pixel the coordinates need to be rescaled
    # example:
    #x_coord_manual_1.append(int(round(row['X']/4.65*2048/2.27*pixel_size))) 
    #y_coord_manual_1.append(int(round(row['Y']/4.74*2087/2.27*pixel_size))) 

# in case by mistake the same point (vesicle) was clicked twice (or more)
coord=zip(x_coord_manual_1,y_coord_manual_1)
coord2=[]
for i in coord:
    if i not in coord2:
        coord2.append(i)

x_coord_manual_1,y_coord_manual_1=zip(*coord2)

del i,idx,row,coord,coord2   

#%% load annotation from person 2
# this cell may be different depending on how the annotations are done/saved!

#option 1
labels_2 = pd.read_csv(root_label+filename_label_2,index_col=1)#,delimiter='\t')

#option 2
#labels = pd.read_csv(root_label+filename_label,index_col=1,delimiter='\t')

x_coord_manual_2=[]
y_coord_manual_2=[]
for idx,row in labels_2.iterrows():
    x_coord_manual_2.append(int(round(row['X']/2.27*pixel_size))) 
    y_coord_manual_2.append(int(round(row['Y']/2.27*pixel_size))) 
    # in case the annotations are not saved in pixel the coordinates need to be rescaled
    # example:
    #x_coord_manual_2.append(int(round(row['X']/4.65*2048/2.27*pixel_size))) 
    #y_coord_manual_2.append(int(round(row['Y']/4.74*2087/2.27*pixel_size)))  

# in case by mistake the same point (vesicle) was clicked twice (or more)
coord=zip(x_coord_manual_2,y_coord_manual_2)
coord2=[]
for i in coord:
    if i not in coord2:
        coord2.append(i)

x_coord_manual_2,y_coord_manual_2=zip(*coord2)

del i,idx,row,coord,coord2      

#%% open the segmentation map

with h5py.File(root_seg + filename_seg, "r") as f:
    
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])
    image_seg=np.zeros((len(data),len(data[0])))

    for j in range(len(data)):
        image_seg[j,:]=data[j].reshape(len(data[j]),)
        
# In the segmentation image:
# pixel values == 0 represent the background, 1
# pixel values == 1 represent are part of the predicted objects
# pixel_values ==2 represent false positive
# therefore, to isolate only the predicted objects, pixels values == 2 are set to 0

for i in range(image_seg.shape[0]):
    for j in range(image_seg.shape[1]):
        if image_seg[i,j]==2:
            image_seg[i,j]=0

del f,a_group_key,data,j,i

#%% check image and segmentation map

plt.subplot(1,2,1)
plt.imshow(np_img,cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image_seg)
plt.axis('off')

#%% count the number of objects (vesicles)

labelarray,count=ndimage.measurements.label(image_seg)      
           
x_coord_ilastik=np.zeros((count))
y_coord_ilastik=np.zeros((count))


for i in range(count):
    x,y=np.where(labelarray==i+1)        
    temp=[]       
    if type(x) == int:
        temp.append(x,y)        
    else:
        for j in range(len(x)):
            temp.append((x[j],y[j]))
            
    kmeans=KMeans(n_clusters=1).fit(temp)
    x_coord_ilastik[i]=int(round(kmeans.cluster_centers_[0][1]))
    y_coord_ilastik[i]=int(round(kmeans.cluster_centers_[0][0]))

del i,x,y,temp,j,kmeans,labelarray,count
    
#%% check on image + annotations

plt.figure()
plt.imshow(np_img[:,:],cmap='gray')
plt.scatter(x_coord_manual_1,y_coord_manual_1,s=8,color='red')
plt.scatter(x_coord_manual_2,y_coord_manual_2,s=8,color='blue')
plt.scatter(x_coord_ilastik,y_coord_ilastik,s=30,color='white',marker='+')
plt.axis('off')   

#%% one to one with pairs formed at the beginning

# run first with manual_1 and then with manual_2 

manual=list(zip(x_coord_manual_1,y_coord_manual_1))
#manual=list(zip(x_coord_manual_2,y_coord_manual_2))
algorithm=list(zip(x_coord_ilastik,y_coord_ilastik))

# creating pairs
# single_auto will contain false positives (after the loop)

single_auto=copy.deepcopy(algorithm)
single_auto_iter=copy.deepcopy(algorithm)
pairs_manual=[]
already_assigned=[]

for m in manual:
    min_distance=999999
    closest_prediction=0
    for s in single_auto_iter:
        xm=m[0]
        ym=m[1]
        xs=s[0]
        ys=s[1]
        M=np.array([xm,ym]).T
        S=np.array([xs,ys]).T
        distance=euclidean_distances(M.reshape(1,-1),S.reshape(1,-1))

        if distance<min_distance:
            if s not in already_assigned:
                min_distance=distance
                closest_prediction=s
            
    if min_distance<=np.sqrt(9**2+9**2):
        
        single_auto.remove(closest_prediction)
        pairs_manual.append([m,closest_prediction])
        already_assigned.append(closest_prediction)
        
# single manual will contain false negatives (after the loop)
        
single_manual=copy.deepcopy(manual)
single_manual_iter=copy.deepcopy(manual)
pairs_auto=[]
already_assigned=[]

for a in algorithm:
    min_distance=999999
    closest_prediction=0
    for s in single_manual_iter:
        xa=a[0]
        ya=a[1]
        xs=s[0]
        ys=s[1]
        A=np.array([xa,ya]).T
        S=np.array([xs,ys]).T
        distance=euclidean_distances(A.reshape(1,-1),S.reshape(1,-1))
        
        if distance<min_distance:
            if s not in already_assigned:
                min_distance=distance
                closest_prediction=s

    if min_distance<=np.sqrt(9**2+9**2):
        
        single_manual.remove(closest_prediction)
        pairs_auto.append([a,closest_prediction])
        already_assigned.append(closest_prediction)
       
print(len(pairs_auto))
print(len(pairs_manual))


print('--------------------------------------------')
print('tot_manual_1: ', len(x_coord_manual_1))
print('tot_manual_2: ', len(x_coord_manual_2))
print('tot_ilastik: ',len(x_coord_ilastik))

print('tot pairs auto and manual (TP): ', len(pairs_auto),len(pairs_manual))
print('tot single auto (FP): ', len(single_auto))
print('tot single manual (FN): ', len(single_manual))
    
    
del m,a,s,A,S,xm,ym,xa,ya,xs,ys,distance,min_distance,closest_prediction