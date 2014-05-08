from __future__ import absolute_import, division, print_function
# Python
from itertools import izip
from os.path import exists, join, split
import os
# UTool
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[detect_randomforest]', DEBUG=False)

from pyrf import Random_Forest_Detector


__LOCATION__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


@utool.indent_func
def detect_rois(ibs, gid_list, gpath_list, species, **kwargs):
	# Create detector
	detector = Random_Forest_Detector(rebuild=False)

	detect_path = ''
	trees_path 	= os.path.join(__LOCATION__, 'rf', species)
	tree_prefix = species + '-'

	detect_config = {
		'percentage_top':	0.40,
	}
		
	# Load forest, so we don't have to reload every time
	forest = detector.load(trees_path, tree_prefix)

	gids = []
	rois = []
	for i in range(len(gpath_list)):
		print('Processing: ' + gpath_list[i])
		src_fpath = gpath_list[i]
		dst_fpath = os.path.join(detect_path, gpath_list[i].split('/')[-1])

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
				gids.append(gid_list[i])
				rois.append((int(res[2]), int(res[3]), int(res[4] - res[2]), int(res[5] - res[3])))

	return gids, rois


# @utool.indent_func
# def detect_rois(ibs, gid_list, gpath_list, species, **kwargs):

#     def _gen_detect_rf(tup):
#         gid, src_fpath, dst_fpath = tup
#         results, timing = detector.detect(forest, src_fpath, dst_fpath, **detect_config)
#         return gid, src_fpath, results

#     # Create detector
#     detector = Random_Forest_Detector(rebuild=False)

#     # Load forest, so we don't have to reload every time
#     trees_path     = os.path.join(__LOCATION__, 'rf', species)
#     tree_prefix = species + '-'
#     forest = detector.load(trees_path, tree_prefix)

#     # Algorithm Settings
#     detect_path = ''
#     detect_config = {
#         'percentage_top':    0.40,
#     }
        
#     # Run Asynchronously
#     dpath_list = [os.path.join(detect_path, gpath.split('/')[-1]) for gpath in gpath_list]
#     arg_list = list(izip(gid_list, gpath_list, dpath_list))
#     detect_async_iter = utool.util_parallel.generate(_gen_detect_rf, arg_list)

#     print('Computing %d RF detections asynchronously' % (len(gpath_list)))
#     gids = []
#     rois = []
#     for gid, gpath, results in detect_async_iter:
#         print('Detected: %r' % gpath)
#         for res in results:
#             # 0 centerx
#             # 1 centery
#             # 2 minx
#             # 3 miny
#             # 4 maxx
#             # 5 maxy
#             # 6 0.0 [UNUSED]
#             # 7 supressed flag

#             if res[7] == 0:
#                 gids.append(gid_list[i])
#                 rois.append((int(res[2]), int(res[3]), int(res[4] - res[2]), int(res[5] - res[3])))
#         pass

#     print('Done detecting ROIS')

#     return gids, rois
