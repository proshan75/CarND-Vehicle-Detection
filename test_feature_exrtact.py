import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np


from lesson_functions import single_img_features


def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])

car_images = glob.glob('Data_Exploration/vehicles/**/*')
noncar_images = glob.glob('Data_Exploration/non-vehicles/**/*')

print('cars count: ', len(car_images))
print('non-cars count: ', len(noncar_images))

car_ind = np.random.randint(0, len(car_images))
noncar_ind = np.random.randint(0, len(noncar_images))

car_image = mpimg.imread(car_images[car_ind])
noncar_image = mpimg.imread(noncar_images[noncar_ind])

color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 1
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
vis = True

car_features, car_hog_image = single_img_features(car_image,
                                                  color_space=color_space,
                                                  spatial_size=spatial_size,
                                                  hist_bins=hist_bins,
                                                  orient=orient,
                                                  pix_per_cell=pix_per_cell,
                                                  cell_per_block=cell_per_block,
                                                  hog_channel=hog_channel,
                                                  spatial_feat=spatial_feat,
                                                  hist_feat=hist_feat,
                                                  hog_feat=hog_feat, 
                                                  vis=vis)

noncar_features, noncar_hog_image = single_img_features(noncar_image,
                                                        color_space=color_space,
                                                        spatial_size=spatial_size,
                                                        hist_bins=hist_bins,
                                                        orient=orient,
                                                        pix_per_cell=pix_per_cell,
                                                        cell_per_block=cell_per_block,
                                                        hog_channel=hog_channel,
                                                        spatial_feat=spatial_feat,
                                                        hist_feat=hist_feat,
                                                        hog_feat=hog_feat,
                                                        vis=vis)

images = [car_image, car_hog_image, noncar_image, noncar_hog_image]
titles = ['car image', 'car HOG image', 'noncar image', 'noncar HOG image']
fig = plt.figure(figsize=(12,3))
visualize(fig, 1, 4, images, titles)

write_name= 'output_images/image_features_plot/car_plot'+str(car_ind)+'_RGB.jpg'
print('car_plot image: ', write_name)
fig.savefig(write_name)

print('done')
