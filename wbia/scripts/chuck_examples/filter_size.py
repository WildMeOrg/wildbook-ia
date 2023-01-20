'''
SCRIPTS TO FORM CENSUS ANNOTATION REGIONS AND TO FILTER
ANNOTATIONS BY ASPECTG RATIO, SIZE, CONTRAST
'''

''' ***************************************************
*
*  CENSUS ANNOTATION REGION AND REGRESSION:
*
*  In order to avoid distractions from other zebras
*  overlapping the bounding box and from background and 
*  vegetation, we formed the notion of a "census annotation
*  region" that narrows down the area of the annotation 
*  to only the most important. This has proven to be 
*  effective, but could be replaced as we move toward use
*  of instance segmentation. It is included here for
*  completeness.
'''

import numpy as np

def convert_box(bbox, aid):
    '''
    Convert from the prediction, which is in normalized 0,1
    coordinates relative to the original bounding box, to
    image image coordinates.
    '''
    xtl, ytl, w, h = ibs.get_annot_bboxes(aid)
    x0, y0, x1, y1 = bbox
    x0 = int(np.around(x0 * w))
    y0 = int(np.around(y0 * h))
    x1 = int(np.around(x1 * w))
    y1 = int(np.around(y1 * h))
    xtl += x0
    ytl += y0
    w -= x0 + x1
    h -= y0 + y1
    bbox = (xtl, ytl, w, h)
    return bbox

# Compute the CA Regions (smaller bounding box) for each Census Annotation
config = {
    'canonical_weight_filepath': 'canonical_zebra_grevys_v4',
}
prediction_list = ibs.depc_annot.get_property(
    'canonical', 
    ca_aids, 
    None, 
    config=config
)

# The following is needed to ensure that the function convert_box is
# visible to the list comprehension that follows. Seems to be an
# ipython thing.
globals().update(locals())

# Convert the bounding boxes from normalized coordinates to image
# coordinates.
bboxes = [
    convert_box(prediction, aid) 
    for prediction, aid in zip(prediction_list, ca_aids)
]

# Create new annotations from the CA Regions
gids = ibs.get_annot_gids(ca_aids)
car_aids = ibs.add_annots(
    gids,
    bboxes,
)

#  Set the species to be a tag that includes the actual species
#  plus the canonical label.
#  NOTE: this seems to be an unnecesary hack considering that
#  there is actually a canonical boolean field in the database.
species_ = ['zebra_grevys+_canonical_'] * len(ca_aids)
viewpoints_ = ibs.get_annot_viewpoints(ca_aids)

ibs.set_annot_species(car_aids, species_)
ibs.set_annot_viewpoints(car_aids, viewpoints_)
ibs.set_annot_detect_confidence(car_aids, confs)



''' ***************************************************
*
*  NON-MAXIMUM SUPPRESSION:
*
*  Non-maximum thresholding is applied to the annotations
*  within each image.
'''

import tqdm

NMS_THRESH = 0.9

# aids = ibs.get_valid_aids()
aids = car_aids

# The ids for the images that the aids came from
gids = ibs.get_annot_gids(aids)

# Gather the aids for each gid
gid_map = {}
for gid, aid in zip(gids, aids):
    if gid not in gid_map:
        gid_map[gid] = []
    gid_map[gid].append(aid)

# Apply NMS to the aids in each image separately.
keep_aids = []
for gid in tqdm.tqdm(gid_map):
    img_aids = sorted(set(gid_map[gid]))
    keep_aids += ibs.nms_aids(img_aids, nms_thresh=NMS_THRESH)

# Find the annotations to delete
to_delete_aids = sorted(list(set(aids) - set(keep_aids)))

print(f"Non-maximum suppression would eliminate {len(to_delete_aids)} out of")
print(f"{len(aids)} based on NMS threshold of {NMS_THRESH}")

# Only uncomment this if you want to actually do the deletion.
# ibs.delete_annots(to_delete_aids)


''' ***************************************************
*
*  CHECK ASPECT RATIO AND REMOVE ECCENTRIC ANNOTATIONS
*
*  This really should be applied to the log of the 
*  apect ratio in order to treat width and height
*  symmetrically.
'''
import numpy as numpy

aids = ca_aids

bboxes = ibs.get_annot_bboxes(aids)
aspects = [h / w for xtl, ytl, w, h in bboxes]
aspect_ratio_mean = np.mean(aspects)
aspect_ratio_std = np.std(aspects)
aspect_thresh_min = aspect_ratio_mean - 2.0 * aspect_ratio_std
aspect_thresh_max = aspect_ratio_mean + 2.0 * aspect_ratio_std

print(f'Aspect ratio mean {aspect_ratio_mean:0.2f}')
print(f'   ratio std dev {aspect_ratio_std:0.2f}')
print(f'   threshold min {aspect_thresh_min:0.2f}')
print(f'   threshold max {aspect_thresh_max:0.2f}')

globals().update(locals())

flags = [
    aspect_thresh_min <= aspect and aspect <= aspect_thresh_max
    for aspect in aspects
]
keep_aids = ut.compress(aids, flags)

delete_aids = sorted(list(set(aids) - set(keep_aids)))
print(f'Aspect ratio check would remove {len(delete_aids)} out of {len(aids)} annotations')

# Only uncomment this if you really mean it.
# ibs.delete_annots(delete_aids)


''' ***************************************************
*
*  ELIMINATE TOO-SMALL ANNOTATIONS
*
*  NOTE: This probably should be based on absolute measures.
'''
# aids = ibs.get_valid_aids()
aids = ca_aids

bboxes = ibs.get_annot_bboxes(aids)

globals().update(locals())

ws = [w for xtl, ytl, w, h in bboxes]
hs = [h for xtl, ytl, w, h in bboxes]
w_thresh = np.mean(ws) - 1.5 * np.std(ws)
h_thresh = np.mean(hs) - 1.5 * np.std(hs)

print(f'Width mean {np.mean(ws):0.1f}, std dev {np.std(ws):0.1f}'
      f', lower threshold {w_thresh:0.1f}')
print(f'Height mean {np.mean(hs):0.1f}, std dev {np.std(hs):0.1f}'
      f', lower threshold {h_thresh:0.1f}')

globals().update(locals())

flags = [
    w_thresh <= w and h_thresh <= h
    for w, h in zip(ws, hs)
]
keep_aids = ut.compress(aids, flags)
delete_aids = list(set(aids) - set(keep_aids))

print(f'Aspect ratio check would remove {len(delete_aids)} out of {len(aids)} annotations')

# Only uncomment this if you really mean it.
# ibs.delete_annots(delete_aids)


''' ***********************************************
*
*  GRADIENT CHECK --- ELIMINATE LOW-CONTRAST ANNOTS
*
* '''

import cv2 

def gradient_magnitude(image_filepath):
    try:
        image = cv2.imread(image_filepath)
        image = image.astype(np.float32)

        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
    except Exception:
        magnitude = [-1.0]

    result = {
        'sum': np.sum(magnitude),
        'mean': np.mean(magnitude),
        'min': np.min(magnitude),
    }
    return result

# aids = ibs.get_valid_aids()
aids = ca_aids

#  Access the cached chip files.
chips = ibs.get_annot_chip_fpath(aids)

globals().update(locals())

#  Provide the chips as an argument to the parallel computation
#  of gradient magnitudes.
args = list(zip(chips))
gradient_dicts = list(ut.util_parallel.generate2(
    gradient_magnitude, args, ordered=True
))

# Compute the mean and threshold
gradient_means = ut.take_column(gradient_dicts, 'mean')
gradient_thresh = np.mean(gradient_means) - 2.0 * np.std(gradient_means)  # 2.0 was 1.5

globals().update(locals())

flags = [
    gradient_mean >= gradient_thresh 
    for gradient_mean in gradient_means
]
keep_aids = ut.compress(aids, flags)

delete_aids = list(set(aids) - set(keep_aids))
print(f'Gradient threshold check would remove {len(delete_aids)} out of {len(aids)} annotations')

# Only uncomment this if you really mean it.
# ibs.delete_annots(delete_aids)