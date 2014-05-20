from __future__ import absolute_import, division, print_function
# Python
#from itertools import izip  # UNUSED
#from os.path import exists, join, split  # UNUSED
import os
# UTool
import utool
from vtool import image
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[detect_randomforest]', DEBUG=False)

from . import grabmodels
from pyrf import Random_Forest_Detector


__LOCATION__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


@utool.indent_func
def detect_rois(ibs, gid_list, gpath_list, species, quick=True, **kwargs):
    # Ensure all models downloaded and accounted for
    grabmodels.ensure_models()

    # Create detector
    if quick:
        config = {}
    else:
        config = {
            'scales': '10 2.0 1.5 1.33 1.15 1.0 0.75 0.55 0.40 0.30 0.20'
        }

    detector = Random_Forest_Detector(rebuild=False, **config)

    detect_path = ''
    trees_path     = os.path.join(__LOCATION__, 'rf', species)
    tree_prefix = species + '-'

    detect_config = {
        'percentage_top':    0.40,
    }

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, tree_prefix)

    # Get scales from the database
    old_sizes_list = ibs.get_image_sizes(gid_list)

    for ix in xrange(len(gpath_list)):
        src_fpath = gpath_list[ix]
        dst_fpath = os.path.join(detect_path, gpath_list[ix].split('/')[-1])
        scale = old_sizes_list[ix][0] / image.open_image_size(src_fpath)[0]

        print('Processing [at scale %.2f]: ' %(scale) + gpath_list[ix])
        results, timing = detector.detect(forest, src_fpath, dst_fpath, **detect_config)

        for res in results:
            # 0 centerx
            # 1 centery
            # 2 minx
            # 3 miny
            # 4 maxx
            # 5 maxy
            # 6 0.0 [UNUSED]
            # 7 supressed flag

            if res[7] == 0:
                gid = [gid_list[ix]]
                roi = [(int(res[2] * scale), 
                        int(res[3] * scale), 
                        int((res[4] - res[2]) * scale), 
                        int((res[5] - res[3]) * scale)
                        )]
                ibs.add_rois(gid, roi)
