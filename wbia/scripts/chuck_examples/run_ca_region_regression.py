# Need the following:

# ca_aids = ...  # list of aids to be designated and ca's

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
