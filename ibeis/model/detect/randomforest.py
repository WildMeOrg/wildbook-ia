from __future__ import absolute_import, division, print_function
# Python
#from itertools import izip  # UNUSED
#from os.path import exists, join, split  # UNUSED
import os
# UTool
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[detect_randomforest]', DEBUG=False)

from . import grabmodels
from pyrf import Random_Forest_Detector


__LOCATION__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


@utool.indent_func
def detect_rois(ibs, gid_list, gpath_list, species, **kwargs):
    # Ensure all models downloaded and accounted for
    grabmodels.ensure_models()

    # Create detector
    detector = Random_Forest_Detector(rebuild=False)

    detect_path = ''
    trees_path     = os.path.join(__LOCATION__, 'rf', species)
    tree_prefix = species + '-'

    detect_config = {
        'percentage_top':    0.40,
    }

    # Load forest, so we don't have to reload every time
    forest = detector.load(trees_path, tree_prefix)

    gids = []
    rois = []
    for ix in xrange(len(gpath_list)):
        print('Processing: ' + gpath_list[ix])
        src_fpath = gpath_list[ix]
        dst_fpath = os.path.join(detect_path, gpath_list[ix].split('/')[-1])

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
                gids.append(gid_list[ix])
                rois.append((int(res[2]), int(res[3]), int(res[4] - res[2]), int(res[5] - res[3])))

    return gids, rois
