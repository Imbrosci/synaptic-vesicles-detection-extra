#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:01:48 2020

Calculate the receiver operating characteristic curve (ROC) and
the area under the curve (AUC)

@author: imbroscb
"""

import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from random import random
from CNNs_GaussianNoiseAdder import MultiClass

# set deterministic = True and look for cuda
torch.manual_seed(2809)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transform and load the data to test
transform = transforms.Compose(
    [transforms.Resize((40, 40)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testing_dataset = datasets.ImageFolder(root=os.path.join('data', 'test'),
                                       transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=testing_dataset,
                                          batch_size=100, shuffle=True)

# %% load and test the model

# enter the path where the model weights (model.pth) are stored
PATH = '.../model.pth'

# define and load the model
model = MultiClass(out=2).to(device)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

# initialized predictions and labels as empty lists
predictions = []
labels = []

# no gradients calculation necessary
with torch.no_grad():

    # iterate over the test_loader
    for test_inputs, test_labels in test_loader:

        # set the inputs in the right shape and feed the classifier
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)
        test_inputs = test_inputs[:, 0, :, :]
        test_inputs = test_inputs.view(test_inputs.shape[0],
                                       1, test_inputs.shape[1],
                                       test_inputs.shape[2])
        test_output = model.forward(test_inputs)

        # transform the output into probabilities
        if device == 'cpu':
            test_output = np.exp(test_output)
        else:
            test_output = np.exp(test_output.cpu())
        for i in range(len(test_output)):
            test_output[i, 0] = test_output[i, 0] / (
                test_output[i, 0] + test_output[i, 1])
            test_output[i, 1] = test_output[i, 1] / (
                test_output[i, 0] + test_output[i, 1])

        test_output = test_output.numpy()

        # fill prediction and labels
        predictions.extend(test_output[:, 1])
        if device == 'cpu':
            labels.extend(test_labels.data.numpy())
        else:
            labels.extend(test_labels.data.cpu().numpy())

p_l_sorted = sorted(zip(predictions, labels), key=lambda x: x[0])

del i

# %% calculate sensitivity, specificity and plot the ROC curve

# set 500 cutoff values (from 0 to 1)
cutoff_values = np.linspace(0, 1, 500)

# inizialize some variables (true positive: tp, true negative: tn,
# false positive: fp, false negative: fn)
tp_list = []
tn_list = []
fp_list = []
fn_list = []
sensitivity = []
specificity = []
pos_labels = 0
neg_labels = 0

# iterate over the predictions and labels
for prediction, label in p_l_sorted:
    if label == 0:
        neg_labels += 1
    elif label == 1:
        pos_labels += 1

# iterate over the cutoff values
for c in cutoff_values:

    # reset some variables
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    counter = 0

    # iterate over the predictions and labels
    for prediction, label in p_l_sorted:
        if (prediction > c) and (label == 1):
            tp += 1
        if (prediction <= c) and (label == 0):
            tn += 1
        if (prediction > c) and (label == 0):
            fp += 1
        if (prediction <= c) and (label == 1):
            fn += 1
        counter += 1

    # append the number of tp, tn, fp, fn for each cutoff value
    tp_list.append(tp)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)

# calculate sensitivity and specificity for each cutoff value
sensitivity = np.array(tp_list) / pos_labels
specificity = np.array(tn_list) / neg_labels

# plot the ROC curve
plt.figure()
plt.plot((1 - specificity), sensitivity)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('1 - specificity', fontsize=14)
plt.ylabel('Sensitivity', fontsize=14)

del prediction, label, c

# %% calculate the AUC score in two ways

# calculate the AUC score mathematically
AUC_score_math = 0
y = sorted(sensitivity)
x = sorted(1 - specificity)
for i in range(len(sensitivity) - 1):
    temp = (y[i+1] + y[i]) / 2 * (x[i+1] - x[i])
    AUC_score_math = AUC_score_math + temp

AUC_score_math = AUC_score_math * 100
print('AUC score with math: {:.4f} %'.format(AUC_score_math))

# calculate the AUC score with Montecarlo simulation
# the outcome should be the same of very similar to the AUC_score_math

x = 1 - specificity
f = interp1d(x, sensitivity)
above = 0
under = 0
for i in range(100000):
    x = random()
    y = random()
    if f(x) > y:
        under += 1
    else:
        above += 1

AUC_score_Montecarlo = under / 100000 * 100
print('AUC score with Montecarlo simulation: {:.4f} %'.format(
    AUC_score_Montecarlo))

del x, y, temp, f, i, above, under
