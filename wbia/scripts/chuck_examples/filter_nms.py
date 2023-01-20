# NEED A LIST OF AIDS

# aids = ibs.get_valid_aids()
# aids = car_aids


''' ***************************************************
*
*  NON-MAXIMUM SUPPRESSION:
*
*  Non-maximum thresholding is applied to the annotations
*  within each image.
'''

import tqdm

NMS_THRESH = 0.9

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
