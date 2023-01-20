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
