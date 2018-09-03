import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from moviepy.editor import VideoFileClip

import cv2

from lesson_functions import *

# load a pe-trained svc model from a serialized (pickle) file
svc_pickle = pickle.load(open("trained_svc_pickle.p", "rb"))

# get attributes of our svc object
svc = svc_pickle["svc"]
X_scaler = svc_pickle["x_scaler"]
scale = svc_pickle["scale"]
orient = svc_pickle["orient"]
pix_per_cell = svc_pickle["pix_per_cell"]
cell_per_block = svc_pickle["cell_per_block"]
spatial_size = svc_pickle["spatial_size"]
hist_bins = svc_pickle["hist_bins"]
color_space = svc_pickle["color_space"]


def process_veh_detect_video_img(image):
    #img = img.astype(np.float32)/255
    draw_img = car_heatmap_bbox_img(image)
    return draw_img


def find_cars_with_scale_variance(img, scale_var, y_start_stop):
    scales = [scale-scale_var, scale, scale_var+scale_var]
    found_imgs = []
    bboxes = []
    for a_scale in scales:
        found_img, bbox = find_cars(img, color_space=color_space,
                                    ystart=y_start_stop[0], ystop=y_start_stop[1], scale=a_scale,
                                    svc=svc, X_scaler=X_scaler, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    spatial_size=spatial_size, hist_bins=hist_bins)
        found_imgs.append(found_img)
        bboxes.append(bbox)

    return found_imgs, bboxes


def car_heatmap_bbox_img(image):
    h = image.shape[0]
    y_start_stop = [h//2, h]
    heat_img = np.zeros_like(image[:, :, 0]).astype(np.float)

    found_imgs, bboxes = find_cars_with_scale_variance(image, scale_var=0.5, y_start_stop=y_start_stop)
    heat_imgs = np.zeros_like(image[:, :, 0]).astype(np.float)
    for bbox in bboxes:
        heat_imgs += add_heat(heatmap=heat_img, bbox_list=bbox)

    heat_img = apply_threshold(heatmap=heat_imgs/len(bboxes), threshold=1)

    heat_img = np.clip(heat_img, 0, 255)

    labels = label(heat_img)

    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img


#Output_video = 'output_project_video_scaled.mp4'
#Input_video = 'project_video.mp4'

#clip1 = VideoFileClip(Input_video)
#video_clip = clip1.fl_image(process_veh_detect_video_img)
#video_clip.write_videofile(Output_video, audio=False)
