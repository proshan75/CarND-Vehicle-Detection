import glob
import pickle
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label

from lesson_functions import (add_heat, apply_threshold, draw_labeled_bboxes,
                              find_cars)
from test_feature_exrtact import visualize

#from test_training_classifier import *

test_win_images = glob.glob('test_images/*')

scale = 1.5
temp_images = []
titles = []


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

for img_file in test_win_images:
    t = time.time()
    img = mpimg.imread(img_file)
    win_img = np.copy(img)
    heat_img = np.zeros_like(img[:, :, 0]).astype(np.float)
    img = img.astype(np.float32)/255
    #print(np.min(img), np.max(img))
    h = img.shape[0]
    y_start_stop = [h//2, h]
    #print('y_start_stop', y_start_stop)

    found_car_img, bbox_list_1 = find_cars(img, color_space=color_space,
                                         ystart=y_start_stop[0], ystop=y_start_stop[1], scale=scale-0.5,
                                         svc=svc, X_scaler=X_scaler, orient=orient,
                                         pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                         spatial_size=spatial_size, hist_bins=hist_bins)

    found_car_img, bbox_list_1_5 = find_cars(img, color_space=color_space,
                                         ystart=y_start_stop[0], ystop=y_start_stop[1], scale=scale,
                                         svc=svc, X_scaler=X_scaler, orient=orient,
                                         pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                         spatial_size=spatial_size, hist_bins=hist_bins)

    found_car_img, bbox_list_2 = find_cars(img, color_space=color_space,
                                         ystart=y_start_stop[0], ystop=y_start_stop[1], scale=scale+0.5,
                                         svc=svc, X_scaler=X_scaler, orient=orient,
                                         pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                         spatial_size=spatial_size, hist_bins=hist_bins)

    heat_img = add_heat(heatmap=heat_img, bbox_list=(bbox_list_1+bbox_list_1_5+bbox_list_2))

    heat_img = apply_threshold(heatmap=heat_img, threshold=1)

    heat_img = np.clip(heat_img, 0, 255)

    labels = label(heat_img)

    print(labels[1])

    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    titles.append(img_file[-9:])
    titles.append(img_file[-9:])
    temp_images.append(draw_img)
    temp_images.append(heat_img)

fig = plt.figure(figsize=(12, 24))
visualize(fig, 6, 2, temp_images, titles)
write_name = 'output_images/hog_subsample_heat/labeled_car_heat'+'.jpg'
print('labeled car heat image: ', write_name)

fig.savefig(write_name)
