import pyrf

def detect_rois(ibs, gid_list, **kwargs):
	# Create detector
	detector = Random_Forest_Detector()
	category = 'zebra_plains'

	trees_path 	= os.path.join('results', category, 'trees')
	tree_prefix = category + '-'

	detect_config = {
		'percentage_top':	0.40,
	}
		
	# Load forest, so we don't have to reload every time
	forest = detector.load(trees_path, tree_prefix)

	for i in range(len(gid_list)):
		src_fpath = gid_list[i]
		dst_fpath = os.path.join(detect_path, files[i].split('/')[-1])

		results, timing = detector.detect(forest, src_fpath, dst_fpath, **detect_config)

		print('[rf] %s | Time: %.3f' %(src_fpath, timing))
		print(results)