from __future__ import print_function, division
# Standard
import os
from os.path import realpath, dirname
from pyrf import Random_Forest_Detector


__LOCATION__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def detect_rois(ibs, gid_list, species, **kwargs):
	# Create detector
	detector = Random_Forest_Detector(rebuild=False)

	print(__LOCATION__)
	detect_path = ''
	trees_path 	= os.path.join(__LOCATION__, 'rf', species)
	tree_prefix = species + '-'

	detect_config = {
		'percentage_top':	0.40,
	}
		
	# Load forest, so we don't have to reload every time
	forest = detector.load(trees_path, tree_prefix)

	for i in range(len(gid_list)):
		src_fpath = gid_list[i]
		dst_fpath = os.path.join(detect_path, gid_list[i].split('/')[-1])

		results, timing = detector.detect(forest, src_fpath, dst_fpath, **detect_config)

		print('[rf] %s | Time: %.3f' %(src_fpath, timing))
		print(results)