# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 17:01:34 2017

@author: wogrady
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
from helper_functions import *
from scipy.ndimage.measurements import label

classifier_pickle = pickle.load( open("svm_classifier_LUV.p", "rb" ) )
svc = classifier_pickle["svc"]
X_scaler = classifier_pickle["X_scaler"]
color_space = classifier_pickle["color_space"]
orient = classifier_pickle["orient"]
pix_per_cell = classifier_pickle["pix_per_cell"]
cell_per_block = classifier_pickle["cell_per_block"]
hog_channel = classifier_pickle["hog_channel"]
spatial_size = classifier_pickle["spatial_size"]
hist_bins = classifier_pickle["hist_bins"]
spatial_feat = classifier_pickle["spatial_feat"]
hist_feat = classifier_pickle["hist_feat"]
hog_feat = classifier_pickle["hog_feat"]
 
cam_pickle = pickle.load( open( "cam_pickle.p", "rb" ) )
mtx = cam_pickle["mtx"]
dist = cam_pickle["dist"]
    
def undistort_image(img, visualize=False):
    # Read in the saved camera matrix and distortion coefficient
    undistort_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undistort_img
    
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, xstart, xstop,
              scale, svc, X_scaler, 
              color_space, orient, pix_per_cell, cell_per_block, 
              spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,
              box_list, visualize=False):
    
    draw_img = np.copy(img)
    img_tosearch = img[ystart:ystop,xstart:xstop,:]

    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch) 
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    #print(pix_per_cell)
    #print(cell_per_block)
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
               
#    if visualize==True:
#        window_list = slide_window(draw_img, x_start_stop=[None, None], y_start_stop=[ystart, ystop], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
#        for window in window_list:
#            cv2.rectangle(draw_img,window[0],window[1],(255,0,0),2) 
        
    #nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if hog_feat == True and hog_channel == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            if hog_feat == True and hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = np.array([])

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
            else:
                spatial_feat = np.array([])
            
            if hist_feat == True:
                hist_features = color_hist(subimg, nbins=hist_bins)
            else:
                hist_feat = np.array([]) 

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if visualize==True:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xstart+xbox_left, ytop_draw+ystart),(xstart+xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),2) 
            else:
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    box_list.append([[xstart+xbox_left, ytop_draw+ystart], [xstart+xbox_left+win_draw,ytop_draw+win_draw+ystart]])
                    cv2.rectangle(draw_img,(xstart+xbox_left, ytop_draw+ystart),(xstart+xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),2) 
                
    return draw_img, box_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 2)
    # Return the image
    return img, bboxes

def process_frame(img, bboxes, tracking):
    
    undistort_img = undistort_image(img, False)
    result_img, bboxes_cars = detect_cars(undistort_img, bboxes)
    
    boxes_tracked = []
    if tracking==True:
        xpadding = 20
        ypadding = 20
        for box in bboxes_cars:
            xstart, ystart = box[0][0]-xpadding, box[0][1]-ypadding
            xend, yend = box[1][0]+xpadding, box[1][1]+ypadding
            bbox = ((xstart, ystart), (xend, yend))
            boxes_tracked.append(bbox)
    
    return result_img, boxes_tracked
    
def detect_cars(img, bboxes):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    box_list = []
    
    for box in bboxes:   
        xstart = box[0][0]
        xstop = box[1][0]
        ystart = box[0][1]
        ystop = box[1][1]
        scale = 0.8
        out_img_inital_track, box_list = find_cars(img, ystart, ystop, xstart, xstop,
                                             scale, svc, X_scaler, 
                                             color_space, orient, pix_per_cell, cell_per_block, 
                                             spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,
                                             box_list, visualize=False)
        
#        out_img_inital, box_list = find_cars(img, ystart, ystop, xstart, xstop, 
#                                         scale, svc, X_scaler, 
#                                         color_space, orient, pix_per_cell, cell_per_block, 
#                                         spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,
#                                         box_list, visualize=False)

#        result = cv2.cvtColor(out_img_inital_track, cv2.COLOR_RGB2BGR)
#        cv2.imshow('Stage 1',result)
#        cv2.waitKey(-1)
        
    
    xstart = 0
    xstop = img.shape[1]
    ystart = 400
    ystop = 685
    scale = 1.8
    out_img_inital_1, box_list = find_cars(img, ystart, ystop, xstart, xstop, 
                                         scale, svc, X_scaler, 
                                         color_space, orient, pix_per_cell, cell_per_block, 
                                         spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,
                                         box_list, visualize=False)
#    result = cv2.cvtColor(out_img_inital_1, cv2.COLOR_RGB2BGR)
#    cv2.imshow('Stage 1',result)
#    cv2.waitKey(-1)
    
    xstart = 400
    xstop = img.shape[1]
    ystart = 350
    ystop = 550
    scale = 1.4
    out_img_inital_2, box_list = find_cars(img, ystart, ystop, xstart, xstop, 
                                         scale, svc, X_scaler, 
                                         color_space, orient, pix_per_cell, cell_per_block, 
                                         spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,
                                         box_list, visualize=False)
    
#    result = cv2.cvtColor(out_img_inital_2, cv2.COLOR_RGB2BGR)
#    cv2.imshow('Stage 2',result)
#    cv2.waitKey(-1)
    
    xstart = 400
    xstop = img.shape[1]
    ystart = 400
    ystop = 500
    scale = 0.8
    out_img_inital_3, box_list = find_cars(img, ystart, ystop, xstart, xstop,
                                         scale, svc, X_scaler, 
                                         color_space, orient, pix_per_cell, cell_per_block, 
                                         spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,
                                         box_list, visualize=False)
    
#    result = cv2.cvtColor(out_img_inital_3, cv2.COLOR_RGB2BGR)
#    cv2.imshow('Stage 3',result)
#    cv2.waitKey(-1)
    
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, bboxes = draw_labeled_bboxes(np.copy(img), labels)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    return draw_img, bboxes
    

test_video = True
bboxes = []

if test_video == True:
    vid_name = 'test_video'
    cap = cv2.VideoCapture(vid_name+'.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img, bboxes = process_frame(frame, bboxes, True)
            #out.write(result)
            cv2.imshow('Result',result_img)
            first_frame = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Video not found")
            break;
    
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    
                        
    
else:
    image_paths = glob.glob('*test_images/*.jpg')
    images = []
    for image_path in image_paths:
            img = mpimg.imread(image_path)
            result_img, bboxes =process_frame(img, bboxes, False)
            cv2.imshow('Post Heat Map',result_img)
            cv2.waitKey(-1)
            cv2.destroyAllWindows()
        
        
