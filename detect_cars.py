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

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
                   
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

# Malisiewicz et al.
def non_max_suppression_fast(boxes_list, overlapThresh):
    # if there are no boxes, return an empty list
    boxes = []
    if len(boxes) == 0:
    	return []
    
    for box in boxes_list:
       boxes.append(box[0][0],box[0][1], box[1][0], box[1][1])
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
     
    # initialize the list of picked indexes	
    pick = []
     
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
     
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
    	# grab the last index in the indexes list and add the
    	# index value to the list of picked indexes
    	last = len(idxs) - 1
    	i = idxs[last]
    	pick.append(i)
     
    	# find the largest (x, y) coordinates for the start of
    	# the bounding box and the smallest (x, y) coordinates
    	# for the end of the bounding box
    	xx1 = np.maximum(x1[i], x1[idxs[:last]])
    	yy1 = np.maximum(y1[i], y1[idxs[:last]])
    	xx2 = np.minimum(x2[i], x2[idxs[:last]])
    	yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
    	# compute the width and height of the bounding box
    	w = np.maximum(0, xx2 - xx1 + 1)
    	h = np.maximum(0, yy2 - yy1 + 1)
    
    	# compute the ratio of overlap
    	overlap = (w * h) / area[idxs[:last]]
    
    	# delete all indexes from the index list that have
    	idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
    
      # return only the bounding boxes that were picked using the
    	# integer data type
    return boxes[pick].astype("int")
    	

def process_frame(img, bboxes, tracking):
    
    #undistort_img = undistort_image(img, False)
    result_img, bboxes_cars = detect_cars(img, bboxes)
    
    boxes_tracked = []
    if tracking==True:
        xpadding = 20
        ypadding = 20
        for box in bboxes_cars:
            xstart, ystart = box[0][0]-xpadding, box[0][1]-ypadding
            xend, yend = box[1][0]+xpadding, box[1][1]+ypadding
            
            xstart = xstart if xstart >= 0 else 0
            xend = xend if xend <= img.shape[1] else img.shape[1]
            ystart = ystart if ystart >= 0 else 0
            yend = yend if yend <= img.shape[0] else img.shape[0]
            bbox = ((xstart, ystart), (xend, yend))
            boxes_tracked.append(bbox)
    
    return result_img, boxes_tracked

def heatmap(img, box_list, thresh, visualize=False):
    # Add heat to each box in box list
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat,box_list)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,thresh)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    if visualize==True:
        fig = plt.figure()
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    draw_img, bboxes = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img, bboxes
    
def detect_cars(img, bboxes):
    
    box_list = []
    tracked_boxes = []

    for box in bboxes:   
        xstart = box[0][0]
        xstop = box[1][0]
        ystart = box[0][1]
        ystop = box[1][1]
        scale = 0.8
        out_img_inital_track, tracked_box_list = find_cars(img, ystart, ystop, xstart, xstop,
                                             scale, svc, X_scaler, 
                                             color_space, orient, pix_per_cell, cell_per_block, 
                                             spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,
                                             box_list, visualize=False)
        draw_img, tracked_boxes = heatmap(img, tracked_box_list, 1, False)
        
#        result = cv2.cvtColor(out_img_inital_track, cv2.COLOR_RGB2BGR)
#        cv2.imshow('Stage 1',result)
#        cv2.waitKey(-1)
        
    
    xstart = 0
    xstop = img.shape[1]
    ystart = 400
    ystop = 665
    scale = 1.8
    out_img_inital_1, box_list = find_cars(img, ystart, ystop, xstart, xstop, 
                                         scale, svc, X_scaler, 
                                         color_space, orient, pix_per_cell, cell_per_block, 
                                         spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,
                                         box_list, visualize=False)
    
    draw_img, box_list_1 = heatmap(img, box_list, 3, False)
    
#    result = cv2.cvtColor(out_img_inital_1, cv2.COLOR_RGB2BGR)
#    cv2.imshow('Stage 1',result)
#    cv2.waitKey(-1)
    
    xstart = 400
    xstop = img.shape[1]
    ystart = 400
    ystop = 600
    scale = 1.3
    out_img_inital_2, box_list = find_cars(img, ystart, ystop, xstart, xstop, 
                                         scale, svc, X_scaler, 
                                         color_space, orient, pix_per_cell, cell_per_block, 
                                         spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,
                                         box_list, visualize=False)
    
    draw_img, box_list_2 = heatmap(img, box_list, 3, False)
    
#    result = cv2.cvtColor(out_img_inital_2, cv2.COLOR_RGB2BGR)
#    cv2.imshow('Stage 2',result)
#    cv2.waitKey(-1)
    
    xstart = 500
    xstop = img.shape[1]
    ystart = 400
    ystop = 500
    scale = 0.8
    out_img_inital_3, box_list = find_cars(img, ystart, ystop, xstart, xstop,
                                         scale, svc, X_scaler, 
                                         color_space, orient, pix_per_cell, cell_per_block, 
                                         spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,
                                         box_list, visualize=False)
    
    draw_img, box_list_3 = heatmap(img, box_list, 2, False)
    
#    result = cv2.cvtColor(out_img_inital_3, cv2.COLOR_RGB2BGR)
#    cv2.imshow('Stage 3',result)
#    cv2.waitKey(-1)
    
    result_boxes = tracked_boxes + box_list_1 + box_list_2 + box_list_3
    draw_img, bboxes = heatmap(img, result_boxes, 0)
    

    #draw_img, bboxes = heatmap(img, box_list, 2)
    #bboxes = non_max_suppression_fast(bboxes, 0.5)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    return draw_img, bboxes
    

test_video = True
bboxes = []

if test_video == True:
    vid_name = 'project_video'
    cap = cv2.VideoCapture(vid_name+'.mp4')
    out = cv2.VideoWriter(vid_name+"_"+color_space+'_result.avi',-1, 20.0, (1280,720))    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img, bboxes = process_frame(frame, bboxes, True)
            out.write(result_img)
            cv2.imshow('Result',result_img)
            first_frame = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Video not found")
            break;
    
    cap.release()
    out.release()
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
        
        
