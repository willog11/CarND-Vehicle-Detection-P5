# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:23:46 2017

@author: wogrady
"""
import glob
import time
import numpy as np
import cv2
import pickle

from helper_functions import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.utils as utils
import matplotlib.pyplot as plt


prototype = False
# Read in cars and notcars
images = glob.glob('*vehicles/*/*/*')
cars = []
notcars = []
for image in images:
    if 'non' in image:
        notcars.append(image)
    else:
        cars.append(image)

if prototype == True:
    cars = utils.shuffle(cars)
    sample_size = 100
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

# Define parameters for feature extraction
color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 9 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
visualize_feat = False # Visualize HOG
visualize_trans = True

if visualize_feat == False:
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat, visualize=False)
else:
    car_features, feature_image, hog_features, spatial_features, channels_hist = extract_features(cars, color_space=color_space, 
                                                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                                                    orient=orient, pix_per_cell=pix_per_cell, 
                                                                    cell_per_block=cell_per_block, 
                                                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                                                    hist_feat=hist_feat, hog_feat=hog_feat, visualize=True)
    fig = plt.figure(figsize=(10, 10))
    if len(hog_features) == 0:
        plt.imshow(hog_features)
    else:
        image = cv2.imread(cars[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bin_edges = channels_hist[0][1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        rows = 3
        cols = 3
        idx = 1
        plt.subplot(rows, cols, idx)
        plt.imshow(image)
        plt.title('Original Image')
        
        idx += 1
        plt.subplot(rows, cols, idx)
        plt.imshow(feature_image)
        plt.title(color_space+' Image')
        
        idx += 1
        plt.subplot(rows, cols, idx)
        plt.plot(spatial_features)
        plt.title('Color Spatial Binning')
        
        idx += 1
        plt.subplot(rows, cols, idx)
        plt.bar(bin_centers, channels_hist[0][0])
        plt.xlim(0, 256)
        plt.title('Color Hist 1st Channel')
        
        idx += 1
        plt.subplot(rows, cols, idx)
        plt.bar(bin_centers, channels_hist[1][0])
        plt.xlim(0, 256)
        plt.title('Color Hist 2nd Channel')
        
        idx += 1
        plt.subplot(rows, cols, idx)
        plt.bar(bin_centers, channels_hist[2][0])
        plt.xlim(0, 256)
        plt.title('Color Hist 3rd Channel')
        
        idx += 1
        plt.subplot(rows, cols, idx)
        plt.imshow(hog_features[0],cmap='gray')
        plt.title('HOG 1st Channel')
        
        idx += 1
        plt.subplot(rows, cols, idx)
        plt.imshow(hog_features[1],cmap='gray')
        plt.title('HOG 2nd Channel')
        
        idx += 1
        plt.subplot(rows, cols, idx)
        plt.imshow(hog_features[2],cmap='gray')
        plt.title('HOG 3rd Channel')
        
print('Car samples: ', len(car_features))
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat, visualize=False)
print('Notcar samples: ', len(notcar_features))


X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

# Standardize the features to prevent biasiing
X_scaler = StandardScaler().fit(X) 
scaled_X = X_scaler.transform(X)

if visualize_trans == True:
    fig = plt.figure(figsize=(12, 4))
    rows = 1
    cols = 3
    
    idx = 1
    plt.subplot(rows, cols, idx)
    image = cv2.imread(cars[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title('Original Image')
    
    idx += 1
    plt.subplot(rows, cols, idx)
    plt.plot(X[0])
    plt.title('Raw Features')
    
    idx += 1
    plt.subplot(rows, cols, idx)
    plt.plot(scaled_X[0])
    plt.title('Normalized Features')
    fig.tight_layout()
    
    
if sample_size > 1: 
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)))) 
    
    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=22)
    
    print('Using:',orient,'orientations', pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    
    # Create linear SVC
    svc = LinearSVC()  
    t=time.time() # Check the training time for the SVC
    
    # Train the classifier
    svc.fit(X_train, y_train) 
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4)) # Check the score of the SVC
    
    # Save the classifier for use later
    classifier_pickle = {}
    classifier_pickle["svc"] = svc
    classifier_pickle["X_scaler"] = X_scaler
    classifier_pickle["color_space"] = color_space
    classifier_pickle["orient"] = orient
    classifier_pickle["pix_per_cell"] = pix_per_cell
    classifier_pickle["cell_per_block"] = cell_per_block
    classifier_pickle["hog_channel"] = hog_channel
    classifier_pickle["spatial_size"] = spatial_size
    classifier_pickle["hist_bins"] = hist_bins
    classifier_pickle["spatial_feat"] = spatial_feat 
    classifier_pickle["hist_feat"] = hist_feat
    classifier_pickle["hog_feat"] = hog_feat                         
    
    pickle.dump(classifier_pickle, open( "svm_classifier.p", "wb" ))