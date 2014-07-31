from __future__ import absolute_import, division, print_function
from six.moves import range
import numpy as np
import cv2
# Tools
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[seg]', DEBUG=False)

DEBUG_SEGM = False


def printDBG(msg):
    if DEBUG_SEGM:
        print(msg)
    pass


def resize_img_and_bbox(img_fpath, bbox_, new_size=None, sqrt_area=400.0):
    printDBG('[segm] imread(%r) ' % img_fpath)
    full_img = cv2.imread(img_fpath)
    (full_h, full_w) = full_img.shape[:2]                 # Image Shape
    printDBG('[segm] full_img.shape=%r' % (full_img.shape,))
    (rw_, rh_) = bbox_[2:]
    # Ensure that we know the new chip size
    if new_size is None:
        target_area = float(sqrt_area) ** 2

        def _resz(w, h):
            ht = np.sqrt(target_area * h / w)
            wt = w * ht / h
            return (int(round(wt)), int(round(ht)))
        new_size_ = _resz(rw_, rh_)
    else:
        new_size_ = new_size
    # Get Scale Factors
    fx = new_size_[0] / rw_
    fy = new_size_[1] / rh_
    printDBG('[segm] fx=%r fy=%r' % (fx, fy))
    dsize = (int(round(fx * full_w)), int(round(fy * full_h)))
    printDBG('[segm] dsize=%r' % (dsize,))
    # Resize the image
    img_resz = cv2.resize(full_img, dsize, interpolation=cv2.INTER_LANCZOS4)
    # Get new ANNOTATION in resized image
    bbox_resz = np.array(np.round(bbox_ * fx), dtype=np.int64)
    return img_resz, bbox_resz


def clean_mask(mask, num_dilate=3, num_erode=3, window_frac=.025):
    """
    Clean the mask
    (num_erode, num_dilate) = (1, 1)
    (w, h) = (10, 10)
    """
    w = h = int(round(min(mask.shape) * window_frac))
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (w, h))
    _mask = mask
    # compute the closing
    for ix in range(num_dilate):
        _mask = cv2.dilate(_mask, element)
    for ix in range(num_erode):
        _mask = cv2.erode(_mask, element)
    return _mask


def fill_holes(mask):
    mode = cv2.RETR_CCOMP
    method = cv2.CHAIN_APPROX_SIMPLE
    image, contours, hierarchy = cv2.findContours(mask, mode, method)
    out = cv2.drawContours(image, contours, -1, (1, 0, 0))
    return out


# Open CV relevant values:
# grabcut_mode = cv2.GC_EVAL
# grabcut_mode = cv2.GC_INIT_WITH_RECT
# cv2.GC_BGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD, cv2.GC_FGD
@profile
def grabcut(rgb_chip):
    (h, w) = rgb_chip.shape[0:2]
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
    cv2.grabCut(rgb_chip, _mask, rect, bgd_model, fgd_model, num_iters, mode=mode)
    is_forground = (_mask == cv2.GC_FGD) + (_mask == cv2.GC_PR_FGD)
    chip_mask = np.where(is_forground, 255, 0).astype('uint8')
    # Crop
    chip_mask = clean_mask(chip_mask)
    chip_mask = np.array(chip_mask, np.float) / 255.0
    # Mask value component of HSV space
    chip_hsv = cv2.cvtColor(rgb_chip, cv2.COLOR_RGB2HSV)
    chip_hsv = np.array(chip_hsv, dtype=np.float) / 255.0
    chip_hsv[:, :, 2] *= chip_mask
    chip_hsv = np.array(np.round(chip_hsv * 255.0), dtype=np.uint8)
    seg_chip = cv2.cvtColor(chip_hsv, cv2.COLOR_HSV2RGB)
    return seg_chip


def segment(img_fpath, bbox_, new_size=None):
    'Runs grabcut'
    printDBG('[segm] segment(img_fpath=%r, bbox=%r)>' % (img_fpath, bbox_))
    num_iters = 5
    bgd_model = np.zeros((1, 13 * 5), np.float64)
    fgd_model = np.zeros((1, 13 * 5), np.float64)
    mode = cv2.GC_INIT_WITH_MASK
    # Initialize
    # !!! CV2 READS (H,W) !!!
    #  WH Unsafe
    img_resz, bbox_resz = resize_img_and_bbox(img_fpath, bbox_, new_size=new_size)
    # WH Unsafe
    (img_h, img_w) = img_resz.shape[:2]                       # Image Shape
    printDBG(' * img_resz.shape=%r' % ((img_h, img_w),))
    # WH Safe
    tlbr = utool.xywh_to_tlbr(bbox_resz, (img_w, img_h))  # Rectangle ANNOTATION
    (x1, y1, x2, y2) = tlbr
    rect = tuple(bbox_resz)                               # Initialize: rect
    printDBG(' * rect=%r' % (rect,))
    printDBG(' * tlbr=%r' % (tlbr,))
    # WH Unsafe
    _mask = np.zeros((img_h, img_w), dtype=np.uint8)  # Initialize: mask
    _mask[y1:y2, x1:x2] = cv2.GC_PR_FGD             # Set ANNOTATION to cv2.GC_PR_FGD
    # Grab Cut
    tt = utool.Timer(' * cv2.grabCut()', verbose=DEBUG_SEGM)
    cv2.grabCut(img_resz, _mask, rect, bgd_model, fgd_model, num_iters, mode=mode)
    tt.toc()
    img_mask = np.where((_mask == cv2.GC_FGD) + (_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    # Crop
    chip      = img_resz[y1:y2, x1:x2]
    chip_mask = img_mask[y1:y2, x1:x2]
    chip_mask = clean_mask(chip_mask)
    chip_mask = np.array(chip_mask, np.float) / 255.0
    # Mask the value of HSV
    chip_hsv = cv2.cvtColor(chip, cv2.COLOR_RGB2HSV)
    chip_hsv = np.array(chip_hsv, dtype=np.float) / 255.0
    chip_hsv[:, :, 2] *= chip_mask
    chip_hsv = np.array(np.round(chip_hsv * 255.0), dtype=np.uint8)
    seg_chip = cv2.cvtColor(chip_hsv, cv2.COLOR_HSV2RGB)
    return seg_chip, img_mask
