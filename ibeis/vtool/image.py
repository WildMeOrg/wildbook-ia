# LICENCE
from __future__ import print_function, division
# Science
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import util_path
import util_str
import util_progress


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


def check_exif_keys(pil_image):
    info_ = pil_image._getexif()
    valid_keys = []
    invalid_keys = []
    for key, val in info_.iteritems():
        try:
            exif_keyval = TAGS[key]
            valid_keys.append((key, exif_keyval))
        except KeyError:
            invalid_keys.append(key)
    print('[io] valid_keys = ' + '\n'.join(valid_keys))
    print('-----------')
    #import draw_func2 as df2
    #exec(df2.present())


def read_all_exif_tags(pil_image):
    info_ = pil_image._getexif()
    info_iter = info_.iteritems()
    tag_ = lambda key: TAGS.get(key, key)
    exif = {} if info_ is None else {tag_(k): v for k, v in info_iter}
    return exif


def read_one_exif_tag(pil_image, tag):
    try:
        exif_key = TAGS.keys()[TAGS.values().index(tag)]
    except ValueError:
        return 'Invalid EXIF Tag'
    info_ = pil_image._getexif()
    if info_ is None:
        return None
    else:
        invalid_str = 'Invalid EXIF Key: exif_key=%r, tag=%r' % (exif_key, tag)
        exif_val = info_.get(exif_key, invalid_str)
    return exif_val
    #try:
        #exif_val = info_[exif_key]
    #except KeyError:
        #exif_val = 'Invalid EXIF Key: exif_key=%r, tag=%r' % (exif_key, tag)
        #print('')
        #print(exif_val)
        #check_exif_keys(pil_image)


def read_exif(fpath, tag=None):
    try:
        pil_image = Image.open(fpath)
        if not hasattr(pil_image, '_getexif'):
            return 'No EXIF Data'
    except IOError as ex:
        import argparse2
        print('Caught IOError: %r' % (ex,))
        print_image_checks(fpath)
        if argparse2.ARGS_.strict:
            raise
        return {} if tag is None else None
    if tag is None:
        exif = read_all_exif_tags(pil_image)
    else:
        exif = read_one_exif_tag(pil_image, tag)
    del pil_image
    return exif


def print_image_checks(img_fpath):
    hasimg = util_path.checkpath(img_fpath, verbose=True)
    if hasimg:
        _tup = (img_fpath, util_str.filesize_str(img_fpath))
        print('[io] Image %r (%s) exists. Is it corrupted?' % _tup)
    else:
        print('[io] Image %r does not exists ' (img_fpath,))
    return hasimg


def read_exif_list(fpath_list, **kwargs):
    def _gen(fpath_list):
        # Exif generator
        nGname = len(fpath_list)
        lbl = '[io] Load Image EXIF'
        mark_progress, end_progress = util_progress.progress_func(nGname, lbl, 16)
        for count, fpath in enumerate(fpath_list):
            mark_progress(count)
            yield read_exif(fpath, **kwargs)
        end_progress()
    exif_list = [exif for exif in _gen(fpath_list)]
    return exif_list
