# LICENCE
from __future__ import absolute_import, division, print_function
from six.moves import range
# Science
import numpy as np
import cv2
# ---------------
# Preprocessing funcs
from utool.util_inject import inject
(print, print_, printDBG, rrr, profile) = inject(
    __name__, '[gfilt]', DEBUG=False)


def adapteq_fn(chipBGR):
    """ create a CLAHE object (Arguments are optional). """
    chipLAB = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2LAB)
    tileGridSize = (8, 8)
    clipLimit = 2.0
    clahe_obj = cv2.createCLAHE(clipLimit, tileGridSize)
    chipLAB[:, :, 0] = clahe_obj.apply(chipLAB[:, :, 0])
    chipBGR = cv2.cvtColor(chipLAB, cv2.COLOR_LAB2BGR)
    return chipBGR


def histeq_fn(chipBGR):
    """ Histogram equalization of a grayscale image. from  _tpl/other """
    chipLAB = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2LAB)
    chipLAB[:, :, 0] = cv2.equalizeHist(chipLAB[:, :, 0])
    chipBGR = cv2.cvtColor(chipLAB, cv2.COLOR_LAB2BGR)
    return chipBGR


def clean_mask(mask, num_dilate=3, num_erode=3, window_frac=.025):
    """ Clean the mask
    (num_erode, num_dilate) = (1, 1)
    (w, h) = (10, 10) """
    w = h = int(round(min(mask.shape) * window_frac))
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (w, h))
    _mask = mask
    # compute the closing
    for ix in range(num_dilate):
        _mask = cv2.dilate(_mask, element)
    for ix in range(num_erode):
        _mask = cv2.erode(_mask, element)
    return _mask


def grabcut_fn(chipBGR):
    """ naively segments a chip """
    chipRGB = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2RGB)
    (h, w) = chipRGB.shape[0:2]
    _mask = np.zeros((h, w), dtype=np.uint8)  # Initialize: mask
    # Set inside to cv2.GC_PR_FGD (probably forground)
    _mask[ :, :] = cv2.GC_PR_FGD
    # Set border to cv2.GC_BGD (definitely background)
    _mask[ 0, :] = cv2.GC_BGD
    _mask[-1, :] = cv2.GC_BGD
    _mask[:,  0] = cv2.GC_BGD
    _mask[:, -1] = cv2.GC_BGD
    # Grab Cut Parameters
    rect = (0, 0, w, h)
    num_iters = 5
    mode = cv2.GC_INIT_WITH_MASK
    bgd_model = np.zeros((1, 13 * 5), np.float64)
    fgd_model = np.zeros((1, 13 * 5), np.float64)
    # Grab Cut Execution
    cv2.grabCut(chipRGB, _mask, rect, bgd_model, fgd_model, num_iters, mode=mode)
    is_forground = (_mask == cv2.GC_FGD) + (_mask == cv2.GC_PR_FGD)
    chip_mask = np.where(is_forground, 255, 0).astype('uint8')
    # Crop
    chip_mask = clean_mask(chip_mask)
    chip_mask = np.array(chip_mask, np.float) / 255.0
    # Mask value component of HSV space
    chipHSV = cv2.cvtColor(chipRGB, cv2.COLOR_RGB2HSV)
    chipHSV = np.array(chipHSV, dtype=np.float) / 255.0
    chipHSV[:, :, 2] *= chip_mask
    chipHSV = np.array(np.round(chipHSV * 255.0), dtype=np.uint8)
    seg_chipBGR = cv2.cvtColor(chipHSV, cv2.COLOR_HSV2BGR)
    return seg_chipBGR

"""
#def maxcontr_fn(chip):
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        #chip_ = pil2_float_img(chip)
        ##p2 = np.percentile(chip_, 2)
        ##p98 = np.percentile(chip_, 98)
        #chip_ = skimage.exposure.equalize_hist(chip_)
        #retchip = Image.fromarray(skimage.utool.img_as_ubyte(chip_)).convert('L')
    #return retchip


#def localeq_fn(chip):
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        #chip_ = skimage.utool.img_as_uint(chip)
        #chip_ = skimage.exposure.equalize_adapthist(chip_, clip_limit=0.03)
        #retchip = Image.fromarray(skimage.utool.img_as_ubyte(chip_)).convert('L')
    #return retchip


#def rankeq_fn(chip):
    ##chip_ = skimage.utool.img_as_ubyte(chip)
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        #chip_ = pil2_float_img(chip)
        #selem = skimage.morphology.disk(30)
        #chip_ = skimage.filter.rank.equalize(chip_, selem=selem)
        #retchip = Image.fromarray(skimage.utool.img_as_ubyte(chip_)).convert('L')
        #return retchip


#def skimage_historam_equalize(chip):
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        #chip_ = pil2_float_img(chip)
        #p2 = np.percentile(chip_, 2)
        #p98 = np.percentile(chip_, 98)
        #chip_ = skimage.exposure.rescale_intensity(chip_, in_range=(p2, p98))
        #retchip = Image.fromarray(skimage.utool.img_as_ubyte(chip_)).convert('L')
    #return retchip
"""
