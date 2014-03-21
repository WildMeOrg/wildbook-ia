# LICENCE
from __future__ import print_function, division
# Science
import cv2


CV2_WARP_KWARGS = {'flags': cv2.INTER_LANCZOS4,
                   'borderMode': cv2.BORDER_CONSTANT}


def imread(img_fpath):
    try:
        # opencv always reads in BGR mode (fastest load time)
        imgBGR = cv2.imread(img_fpath, flags=cv2.CV_LOAD_IMAGE_COLOR)
        return imgBGR
    except Exception as ex:
        print('[gtool] Caught Exception: %r' % ex)
        print('[gtool] ERROR reading: %r' % (img_fpath,))
        raise


def cvt_BGR2L(imgBGR):
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgL = imgLAB[:, :, 0]
    return imgL


def warpAffine(img, M, dsize):
    warped_img = cv2.warpAffine(img, M[0:2], tuple(dsize), **CV2_WARP_KWARGS)
    return warped_img
