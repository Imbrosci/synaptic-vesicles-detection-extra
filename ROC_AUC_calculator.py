#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:01:48 2020

@author: barbara
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
from vesicle_classifier import MultiClass
from scipy.interpolate import interp1d
from random import random

#%%
torch.manual_seed(2809)
torch.backends.cudnn.deterministic = True 
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%% transform and load the data

transform=transforms.Compose([transforms.Resize((40,40)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) 
testing_dataset=datasets.ImageFolder(root=os.path.join('data','test'),transform=transform) #training and validation dataset is already divided
validation_loader=torch.utils.data.DataLoader(dataset=testing_dataset,batch_size=100, shuffle=True)

#%% get predictions (as probabilities) and real labels

model = MultiClass(out=2).to(device)

PATH='.../model.pth' # path of model weights (.pth)

model.load_state_dict(torch.load(PATH,map_location=device))  
model.eval()  

prediction=[]
real=[]

with torch.no_grad():# .no_grad because there is no need to calculate gradients  
    for val_inputs,val_labels in validation_loader:
        val_inputs=val_inputs.to(device)
        val_labels=val_labels.to(device)
        val_inputs=val_inputs[:,0,:,:]
        val_inputs=val_inputs.view(val_inputs.shape[0],1,val_inputs.shape[1],val_inputs.shape[2])

        val_output=model.forward(val_inputs)
        
        val_out_proc=np.exp(val_output.cpu())
        val_preds=np.zeros((len(val_out_proc)))
        for i in range(len(val_out_proc)):
            val_out_proc[i,0]=val_out_proc[i,0]/(val_out_proc[i,0]+val_out_proc[i,1])
            val_out_proc[i,1]=val_out_proc[i,1]/(val_out_proc[i,0]+val_out_proc[i,1])
        
        val_out_proc=val_out_proc.numpy()
        prediction.extend(val_out_proc[:,1])
        real.extend(val_labels.data.cpu().numpy())
        
 
ordered=sorted(zip(prediction,real),key=lambda x:x[0])   

del i  
#%% calculate sensitivity and specificity for 500 cutoff values (from 0 to 1)

cutoff=np.linspace(0,1,500)

sensitivity=[]
specificity=[]

real_pos=0
real_neg=0
for label in ordered:
    if label[1]==0:
        real_neg+=1
    elif label[1]==1:
        real_pos+=1

truePos=[]
trueNeg=[]
falsePos=[]
falseNeg=[]

for c in cutoff:

    true_pos=0
    true_neg=0    
    false_pos=0
    false_neg=0

    for i in range(len(ordered)):

        if (ordered[i][0]>c) and (ordered[i][1]==1):
            true_pos+=1
        if (ordered[i][0]<=c) and (ordered[i][1]==0):
            true_neg+=1        
        if (ordered[i][0]>c) and (ordered[i][1]==0):
            false_pos+=1
        if (ordered[i][0]<=c) and (ordered[i][1]==1):
            false_neg+=1

    truePos.append(true_pos)
    trueNeg.append(true_neg)
    falsePos.append(false_pos)
    falseNeg.append(false_neg)
    

sensitivity=np.array(truePos)/real_pos    
specificity=np.array(trueNeg)/real_neg           

del label,c,i           
   
#%% plot the ROC curve 
     
plt.plot((1-specificity),sensitivity)    
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)   
plt.xlabel('1 - specificity',fontsize=14)
plt.ylabel('Sensitivity',fontsize=14)

#%% calculate the AUC score matematically

AUC_score=0
y=sorted(sensitivity)
x=sorted(1-specificity)
for i in range(len(sensitivity)-1):
    temp= (y[i+1]+y[i])/2  * (x[i+1]-x[i])
    AUC_score=AUC_score+temp
    
AUC_score=AUC_score*100
print('AUC score:', AUC_score)

del temp,y,x

#%% calculate the AUC score with Montecarlo simulation 

x=1-specificity
f=interp1d(x,sensitivity)


above=0
under=0
for i in range(100000):
    x=random()
    y=random()
    if f(x)>y:
        under+=1
    else:
        above+=1
        
AUC_score=under/100000*100
print('AUC score:', AUC_score)

del f,i,x,y,above,under   