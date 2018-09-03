import glob

import matplotlib.image as mpimg
import numpy as np

from lesson_functions import draw_boxes, search_windows, slide_window
from test_training_classifier import *

test_win_images = glob.glob('test_images/*')

overlap = 0.5
temp_images = []
titles = []

for img_file in test_win_images:
    t = time.time()
    img = mpimg.imread(img_file)
    win_img = np.copy(img)
    img = img.astype(np.float32)/255
    print(np.min(img), np.max(img))
    h = img.shape[0]
    y_start_stop = [h//2, h]
    print('y_start_stop', y_start_stop)

    windows = slide_window(img, x_start_stop=[None, None],
                           y_start_stop=y_start_stop, xy_window=(96, 96), xy_overlap=(overlap, overlap))

    search_wins = search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                 spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(win_img, search_wins, color=(0, 0, 255), thick=5)

    temp_images.append(window_img)
    titles.append(img_file)
    print(time.time()-t, 'seconds to process one image searching', len(windows), 'windows')
    
fig = plt.figure(figsize=(12,18), dpi=300)
visualize(fig, 5,2, temp_images, titles)
write_name= 'output_images/search_draw_win/search_draw_win'+'.jpg'
print('car_plot image: ', write_name)

fig.savefig(write_name)
