
import matplotlib.image as mpimg

import numpy as np
import pickle

from test_feature_exrtact import *
from lesson_functions import *

color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
scale = 1.5

t = time.time()
n_samples = 1000
random_ids = np.random.randint(0, len(car_images), n_samples)
test_cars = np.array(car_images)[random_ids]
test_noncars = np.array(noncar_images)[random_ids]

car_features = extract_features(car_images,
                                color_space=color_space, 
                                spatial_size=spatial_size, 
                                hist_bins=hist_bins,
                                orient=orient, 
                                pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, 
                                spatial_feat=spatial_feat, 
                                hist_feat=hist_feat,
                                hog_feat=hog_feat)

noncar_features = extract_features(noncar_images,
                                color_space=color_space, 
                                spatial_size=spatial_size, 
                                hist_bins=hist_bins,
                                orient=orient, 
                                pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, 
                                spatial_feat=spatial_feat, 
                                hist_feat=hist_feat,
                                hog_feat=hog_feat)

print (time.time()-t, 'Seconds to compute features..')

X = np.vstack((car_features, noncar_features)).astype(np.float64)

X_scaler = StandardScaler().fit(X)

scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

rand_state = np.random.randint(0,100)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

print('Using: ', orient, 'orientations, ', pix_per_cell, 'pixels per cell,', 
cell_per_block, 'cells per block,', hist_bins, 'histogram bins, and',
spatial_size, 'spatial sampling')
print('Feature vector length: ', len(X_train[0]))

svc = LinearSVC(C=0.01)
t = time.time()
svc.fit(X_train, y_train)
print(round(time.time()-t, 2), 'Seconds to train SVC...')
print('accuracy: ', round(svc.score(X_test, y_test), 4))


#trained_svc_pickle = {}
#trained_svc_pickle["svc"] = svc
#trained_svc_pickle["x_scaler"] = X_scaler
#trained_svc_pickle["scale"] = scale
#trained_svc_pickle["orient"] = orient
#trained_svc_pickle["pix_per_cell"] = pix_per_cell
#trained_svc_pickle["cell_per_block"] = cell_per_block
#trained_svc_pickle["spatial_size"] = spatial_size
#trained_svc_pickle["hist_bins"] = hist_bins
#trained_svc_pickle["color_space"] = color_space

#pickle.dump(trained_svc_pickle, open('./trained_svc_pickle.p', 'wb'))

