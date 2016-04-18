#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from ibeis_cnn.ingest_ibeis import get_cnn_detector_training_images
from saliency import transform_image_to_prior_space
from os.path import join, expanduser
from sklearn.cluster import KMeans
from config import PROPOSALS, CHIP_SIZE
import numpy as np
import ibeis_cnn  # NOQA
import ibeis
import cv2

dbname_list = [
    'ELPH_Master',
    'GIR_Master',
    'GZ_Master',
    'NNP_MasterGIRM',
    'PZ_Master1',
]

# Process all databases
global_bbox_list = []
for dbname in dbname_list:
    print(dbname)
    ibs = ibeis.opendb(join('/', 'Datasets', 'BACKGROUND', dbname))
    global_bbox_list_ = get_cnn_detector_training_images(
        ibs,
        image_size=CHIP_SIZE[0]
    )
    global_bbox_list.extend(global_bbox_list_)

global_bbox_list = np.array(global_bbox_list)

################################################################################
# Residual clustering

global_bbox_list_ = transform_image_to_prior_space(global_bbox_list)

# Cluster the bboxes into priors
model = KMeans(n_clusters=PROPOSALS)
model.fit(global_bbox_list_)
cluster_priors = model.cluster_priors_

print(cluster_priors.shape)
print(cluster_priors.dtype)
print(cluster_priors)

# Save the clusters
prior_filename = 'priors.transformed.npy'
prior_filepath = expanduser(join('~', 'Desktop', 'extracted', prior_filename))
np.save(prior_filepath, cluster_priors)

################################################################################
# Normal clustering

# Cluster the bboxes into priors
model = KMeans(n_clusters=PROPOSALS)
model.fit(global_bbox_list)
cluster_priors = model.cluster_priors_

print(cluster_priors.shape)
print(cluster_priors.dtype)
print(cluster_priors)

# Save the clusters
prior_filename = 'priors.npy'
prior_filepath = expanduser(join('~', 'Desktop', 'extracted', prior_filename))
np.save(prior_filepath, cluster_priors)

# Reload the clusters
cluster_priors = np.load(prior_filepath)

# Assert clustering priors are reconstructable
model = KMeans(n_clusters=PROPOSALS)
model.cluster_priors_ = cluster_priors
for index, prior in enumerate(cluster_priors):
    assert index == model.predict(prior)

# Show centroids
image_size = 512
canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
for (xc, yc, xr, yr) in cluster_priors:
    xtl_ = int((xc - xr) * image_size)
    ytl_ = int((yc - yr) * image_size)
    xbr_ = int((xc + xr) * image_size)
    ybr_ = int((yc + yr) * image_size)

    xtl_ = min(image_size, max(0, xtl_))
    ytl_ = min(image_size, max(0, ytl_))
    xbr_ = min(image_size, max(0, xbr_))
    ybr_ = min(image_size, max(0, ybr_))

    cv2.rectangle(canvas, (xtl_, ytl_), (xbr_, ybr_), (0, 255, 0))

# Show the centoids found by k-means
cv2.imshow('prior centroids', canvas)
cv2.waitKey(0)
