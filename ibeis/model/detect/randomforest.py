"""
Interface to pyrf random forest object detection.
"""
from __future__ import absolute_import, division, print_function
from os.path import splitext, exists, join
from six.moves import zip, map, range
from ibeis.model.detect import grabmodels
from vtool import image as gtool
import utool as ut
import cv2
import pyrf
import multiprocessing
import random
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[randomforest]')


"""
from ibeis.model.detect import randomforest
dir(randomforest)
func = randomforest.ibeis_generate_image_detections
print(ut.make_default_docstr(func))
"""


VERBOSE_RF = ut.get_argflag('--verbrf') or ut.VERBOSE


#=================
# IBEIS INTERFACE
#=================


def train(ibs, gid_list, trees_path=None, species=None, verbose=VERBOSE_RF, **kwargs):
    def _overlap_percentage((xmin1, xmax1, ymin1, ymax1), (xmin2, xmax2, ymin2, ymax2)):
        width1, height1 = xmax1 - xmin1, ymax1 - ymin1
        width2, height2 = xmax2 - xmin2, ymax2 - ymin2
        x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
        y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
        area_overlap = float(x_overlap * y_overlap)
        area_total = min(width1 * height1, width2 * height2)
        percentage = area_overlap / area_total
        return percentage

    def valid_candidate(candidate, annot_bbox_list, overlap=0.0, tries=10):
        for i in range(tries):
            valid = True
            for annot_bbox in annot_bbox_list:
                xtl, ytl, width, height = annot_bbox
                xmin, xmax, ymin, ymax = xtl, xtl + width, ytl, ytl + height
                if _overlap_percentage(candidate, (xmin, xmax, ymin, ymax)) > overlap:
                    valid = False
                    break  # break inner loop
            if valid:
                return True
        return False
    
    # Ensure directories for negatives
    if trees_path is None:
        trees_path = join(ibs.get_ibsdir(), 'trees')
    negatives_cache = join(ibs.get_cachedir(), 'pyrf_train_negatives')
    if exists(negatives_cache):
        ut.remove_dirs(negatives_cache)
    ut.ensuredir(negatives_cache)
    # Get positive chip paths
    if species is None:
        aids_list = ibs.get_image_aids(gid_list)
    else:
        aids_list = ibs.get_image_aids_of_species(gid_list, species)
    aid_list = ut.flatten(aids_list)
    train_pos_cpath_list = ibs.get_annot_chip_fpaths(aid_list)
    # Get negative chip paths
    train_neg_cpath_list = []
    while len(train_neg_cpath_list) < len(train_pos_cpath_list):
        sample = random.randint(0, len(gid_list) - 1)
        gid = gid_list[sample]
        img_width, img_height = ibs.get_image_sizes(gid)
        size = min(img_width, img_height)
        if species is None:
            aid_list = ibs.get_image_aids(gid)
        else:
            aid_list = ibs.get_image_aids(gid, species)
        annot_bbox_list = ibs.get_annot_bboxes(aid_list)

        square = random.randint(int(size / 4), int(size / 2))
        xmin = random.randint(0, img_width - square)
        xmax = xmin + square
        ymin = random.randint(0, img_height - square)
        ymax = ymin + square
        if valid_candidate((xmin, xmax, ymin, ymax), annot_bbox_list):
            print("CREATING", xmin, xmax, ymin, ymax)
            img = ibs.get_images(gid)
            img_path = join(negatives_cache, "neg_%07d.JPEG" % (len(train_neg_cpath_list), ))
            img = img[ymin:ymax, xmin:xmax]
            cv2.imwrite(img_path, img)
            train_neg_cpath_list.append(img_path)
    # Train trees
    detector = pyrf.Random_Forest_Detector()
    detector.train(train_pos_cpath_list, train_neg_cpath_list, trees_path, **kwargs)
    # Remove cached negatives directory
    ut.remove_dirs(negatives_cache)


def test():
    pass
