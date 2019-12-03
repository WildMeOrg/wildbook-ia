# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from six.moves import range
import numpy as np
import utool as ut
import ubelt as ub
import cv2


class IntensityPreproc(object):
    """
    Prefered over old methods

    CommandLine:
        python -m vtool.image_filters IntensityPreproc --show

    Doctest:
        >>> from vtool.image_filters import *
        >>> import vtool as vt
        >>> chipBGR = vt.imread(ut.grab_file_url('http://i.imgur.com/qVWQaex.jpg'))
        >>> filter_list = [
        >>>     ('medianblur', {}),
        >>>     ('adapteq', {}),
        >>> ]
        >>> self = IntensityPreproc()
        >>> chipBGR2 = self.preprocess(chipBGR, filter_list)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> pt.imshow(chipBGR, pnum=(1, 2, 1), fnum=1)
        >>> pt.imshow(chipBGR2, pnum=(1, 2, 2), fnum=1)
        >>> ut.show_if_requested()
    """

    def preprocess(self, chipBGR, filter_list):
        """
        filter_list is a list of (name, config) tuples for preforming filter ops
        """

        # Convert into LAB space for grayscale extraction
        chipLAB = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2LAB)
        intensity = chipLAB[:, :, 0]

        # Modify intensity
        for filtname, config in filter_list:
            intensity = getattr(self, filtname)(intensity, **config)

        # Add color back in
        chipLAB[:, :, 0] = intensity
        chipBGR = cv2.cvtColor(chipLAB, cv2.COLOR_LAB2BGR)
        return chipBGR

    def adapteq(self, intensity, tileGridSize=(8, 8), clipLimit=2.0):
        clahe_obj = cv2.createCLAHE(clipLimit, tileGridSize)
        intensity = clahe_obj.apply(intensity)
        return intensity

    def medianblur(self, intensity, noise_thresh=50, ksize1=3, ksize2=5):
        istd = intensity.std()
        ksize = ksize1 if istd < noise_thresh else ksize2
        intensity = cv2.medianBlur(intensity, ksize)
        return intensity

    def histeq(self, intensity):
        """ Histogram equalization of a grayscale image. """
        return cv2.equalizeHist(intensity)


def manta_matcher_filters(chipBGR):
    """
    References:
        http://onlinelibrary.wiley.com/doi/10.1002/ece3.587/full

    Ignore:
        >>> from ibeis.core_annots import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('Mantas')
        >>> chipBGR = vt.imread(ut.grab_file_url('http://i.imgur.com/qVWQaex.jpg'))
    """
    chipLAB = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2LAB)

    intensity = chipLAB[:, :, 0]
    # Median filter
    noise_thresh = 100
    ksize = 5 if intensity.std() > noise_thresh else 3
    intensity = cv2.medianBlur(intensity, ksize)

    tileGridSize = (8, 8)
    clipLimit = 2.0
    clahe_obj = cv2.createCLAHE(clipLimit, tileGridSize)
    intensity = clahe_obj.apply(intensity, dst=intensity)

    chipLAB[:, :, 0] = intensity
    chipBGR = cv2.cvtColor(chipLAB, cv2.COLOR_LAB2BGR)
    return chipBGR


def adapteq_fn(chipBGR):
    """
    adaptive histogram equalization with CLAHE

    Example:
        >>> from vtool.image_filters import *
        >>> import vtool as vt
        >>> import utool as ut
        >>> chipBGR = vt.imread(ut.grab_file_url('http://i.imgur.com/qVWQaex.jpg'))
        >>> chip2 = adapteq_fn(chipBGR)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> pt.imshow(chipBGR, pnum=(1, 2, 1), fnum=1)
        >>> pt.imshow(chip2, pnum=(1, 2, 2), fnum=1)
        >>> ut.show_if_requested()
    """
    chipLAB = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2LAB)
    tileGridSize = (8, 8)
    clipLimit = 2.0
    clahe_obj = cv2.createCLAHE(clipLimit, tileGridSize)
    chipLAB[:, :, 0] = clahe_obj.apply(chipLAB[:, :, 0])
    chipBGR = cv2.cvtColor(chipLAB, cv2.COLOR_LAB2BGR)
    return chipBGR


def medianfilter_fn(chipBGR):
    """
    median filtering

    Example:
        >>> from vtool.image_filters import *
        >>> import vtool as vt
        >>> import utool as ut
        >>> chipBGR = vt.imread(ut.grab_file_url('http://i.imgur.com/qVWQaex.jpg'))
        >>> chip2 = adapteq_fn(chipBGR)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> pt.imshow(chipBGR, pnum=(1, 2, 1), fnum=1)
        >>> pt.imshow(chip2, pnum=(1, 2, 2), fnum=1)
        >>> ut.show_if_requested()
    """
    chipLAB = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2LAB)
    intensity = chipLAB[:, :, 0]
    noise_thresh = 100
    ksize = 5 if intensity.std() > noise_thresh else 3
    intensity = cv2.medianBlur(intensity, ksize)
    chipLAB[:, :, 0] = intensity
    chipBGR = cv2.cvtColor(chipLAB, cv2.COLOR_LAB2BGR)
    return chipBGR


def histeq_fn(chipBGR):
    """ Histogram equalization of a grayscale image. """
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



if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool.image_filters
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
