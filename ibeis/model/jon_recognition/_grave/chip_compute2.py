from __future__ import absolute_import, division, print_function
# Python
import warnings
from os.path import join
# Science
import numpy as np
import cv2
from PIL import Image
# Hotspotter
import algos
import utool
from ibeis.dev import params
import vtool.images as gtool
(print, print_,
 printDBG, rrr, profile) = utool.inject(__name__, '[cc2]', DEBUG=False)


def xywh_to_tlbr(roi, img_wh):
    (img_w, img_h) = img_wh
    if img_w == 0 or img_h == 0:
        msg = '[cc2.1] Your csv tables have an invalid ROI.'
        print(msg)
        warnings.warn(msg)
        img_w = 1
        img_h = 1
    # Ensure ROI is within bounds
    (x, y, w, h) = roi
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, img_w - 1)
    y2 = min(y + h, img_h - 1)
    return (x1, y1, x2, y2)


# Parallelizable Work Functions
def build_transform2(roi, chipsz, theta):
    (x, y, w, h) = roi
    (w_, h_) = chipsz


def build_transform(x, y, w, h, w_, h_, theta, homogenous=False):
    sx = (w_ / w)  # ** 2
    sy = (h_ / h)  # ** 2
    cos_ = np.cos(-theta)
    sin_ = np.sin(-theta)
    tx = -(x + (w / 2))
    ty = -(y + (h / 2))
    T1 = np.array([[1, 0, tx],
                   [0, 1, ty],
                   [0, 0, 1]], np.float64)

    S = np.array([[sx, 0,  0],
                  [0, sy,  0],
                  [0,  0,  1]], np.float64)

    R = np.array([[cos_, -sin_, 0],
                  [sin_,  cos_, 0],
                  [   0,     0, 1]], np.float64)

    T2 = np.array([[1, 0, (w_ / 2)],
                   [0, 1, (h_ / 2)],
                   [0, 0, 1]], np.float64)
    M = T2.dot(R.dot(S.dot(T1)))
    #.dot(R)#.dot(S).dot(T2)

    if homogenous:
        transform = M
    else:
        transform = M[0:2, :] / M[2, 2]

    #utool.horiz_print(S, T1, T2)
    #print('T1======')
    #print(T1)
    #print('R------')
    #print(R)
    #print('S------')
    #print(S)
    #print('T2------')
    #print(T2)
    #print('M------')
    #print(M)
    #print('Aff------')
    #print(Aff)
    #print('======')
    return transform


# RCOS TODO: Parametarize interpolation method
INTERPOLATION_TYPES = {
    'nearest': cv2.INTER_NEAREST,
    'linear':  cv2.INTER_LINEAR,
    'area':    cv2.INTER_AREA,
    'cubic':   cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}

cv2_flags = INTERPOLATION_TYPES['lanczos']
cv2_borderMode  = cv2.BORDER_CONSTANT
cv2_warp_kwargs = {'flags': cv2_flags, 'borderMode': cv2_borderMode}


def extract_chip(img_fpath, roi, theta, new_size):
    'Crops chip from image ; Rotates and scales; Converts to grayscale'
    # Read parent image
    #printDBG('[cc2] reading image')
    imgBGR = gtool.imread(img_fpath)
    #printDBG('[cc2] building transform')
    # Build transformation
    (rx, ry, rw, rh) = roi
    (rw_, rh_) = new_size
    Aff = build_transform(rx, ry, rw, rh, rw_, rh_, theta)
    #printDBG('[cc2] rotate and scale')
    # Rotate and scale
    imgBGR = cv2.warpAffine(imgBGR, Aff, (rw_, rh_), **cv2_warp_kwargs)
    #printDBG('[cc2] return extracted')
    return imgBGR


# TODO: Change the force_gray to work a little nicer
def compute_chip(img_fpath, chip_fpath, roi, theta, new_size, filter_list, force_gray=False):
    '''Extracts Chip; Applies Filters; Saves as png'''
    #printDBG('[cc2] extracting chip')
    chipBGR = extract_chip(img_fpath, roi, theta, new_size)
    #printDBG('[cc2] extracted chip')
    for func in filter_list:
        #printDBG('[cc2] computing filter: %r' % func)
        chipBGR = func(chipBGR)
    cv2.imwrite(chip_fpath, chipBGR)
    return True

# ---------------
# Preprocessing funcs


def adapteq_fn(chipBGR):
    # create a CLAHE object (Arguments are optional).
    chipLAB = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2LAB)
    tileGridSize = (8, 8)
    clipLimit = 2.0
    clahe_obj = cv2.createCLAHE(clipLimit, tileGridSize)
    chipLAB[:, :, 0] = clahe_obj.apply(chipLAB[:, :, 0])
    chipBGR = cv2.cvtColor(chipLAB, cv2.COLOR_LAB2BGR)
    return chipBGR


def histeq_fn(chapBGR):
    # Histogram equalization of a grayscale image. from  _tpl/other
    chipLAB = cv2.cvtColor(chapBGR, cv2.COLOR_BGR2LAB)
    chipLAB[:, :, 0] = cv2.equalizeHist(chipLAB[:, :, 0])
    chapBGR = cv2.cvtColor(chipLAB, cv2.COLOR_LAB2BGR)
    return chapBGR


def grabcut_fn(chipBGR):
    import segmentation
    chipRGB = cv2.cvtColor(chipBGR, cv2.COLOR_BGR2RGB)
    chipRGB = segmentation.grabcut(chipRGB)
    chapBGR = cv2.cvtColor(chipRGB, cv2.COLOR_RGB2BGR)
    return chapBGR


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


def region_norm_fn(chip):
    chip_ = np.array(chip, dtype=np.float)
    chipw, chiph = chip_.shape
    half_w = chipw * .1
    half_h = chiph * .1
    x1 = chipw / 2 - half_w
    y1 = chiph / 2 - half_h
    x2 = chipw / 2 + half_w
    y2 = chiph / 2 + half_h
    (x1, y1, x2, y2) = map(int, map(round, (x1, y1, x2, y2)))
    area = chip_[x1:x2, y1:y2]
    intensity = area.flatten()
    freq, _  = np.histogram(intensity, 32)
    #algos.viz_localmax(freq)
    maxpos  = algos.localmax(freq)  # fix with htool
    min_int = intensity.min()
    max_int = intensity.max()
    maxima = min_int + (max_int - min_int) * np.array(maxpos) / float(len(freq))
    if len(maxima) > 2:
        low  = float(maxima[0])
        high = float(maxima[-1])
    else:
        low  = float(min_int)
        high = float(max_int)
    chip_ = (chip_ - low) * 255.0 / (high - low)
    chip_ = chip_.round()
    chip_[chip_ < 0] = 0
    chip_[chip_ > 255] = 255
    return chip_


#def pil2_float_img(chip):
    #return skimage.utool.img_as_float(chip)
    ##chip_ = np.asarray(chip, dtype=np.float)
    ##if chip_.max() > 1:
        ##chip_ /= 255.0
    #return chip_


def compute_uniform_area_chip_sizes(roi_list, sqrt_area=None):
    'Computes a normalized chip size to rescale to'
    if not (sqrt_area is None or sqrt_area <= 0):
        target_area = sqrt_area ** 2

        def _resz(w, h):
            try:
                ht = np.sqrt(target_area * h / w)
                wt = w * ht / h
                return (int(round(wt)), int(round(ht)))
            except Exception:
                msg = '[cc2.2] Your csv tables have an invalid ROI.'
                print(msg)
                warnings.warn(msg)
                return (1, 1)
        chipsz_list = [_resz(float(w), float(h)) for (x, y, w, h) in roi_list]
    else:  # no rescaling
        chipsz_list = [(int(w), int(h)) for (x, y, w, h) in roi_list]
    return chipsz_list


def batch_extract_chips(gfpath_list, cfpath_list, roi_list, theta_list,
                        uniform_size=None, uniform_sqrt_area=None,
                        filter_list=[], num_procs=1, lazy=True, force_gray=False):
    '''
    cfpath_fmt - a string with a %d embedded where the cid will go.
    '''
    try:
        list_size_list = map(len, (gfpath_list, cfpath_list, roi_list, theta_list))
        assert all([list_size_list[0] == list_size for list_size in list_size_list])
    except AssertionError as ex:
        print(ex)
        raise
    # Normalized Chip Sizes: ensure chips have about sqrt_area squared pixels
    if uniform_sqrt_area is not None:
        chipsz_list = compute_uniform_area_chip_sizes(roi_list, uniform_sqrt_area)
    elif uniform_size is not None:
        chipsz_list = [uniform_size] * len(roi_list)
    else:
        chipsz_list = [(int(w), int(h)) for (x, y, w, h) in roi_list]

    arg_list = [gfpath_list, cfpath_list, roi_list, theta_list, chipsz_list]
    pcc_kwargs = {
        'arg_list': arg_list,
        'lazy': lazy,
        'num_procs': num_procs,
        'common_args': [filter_list, force_gray]
    }
    # Compute all chips with paramatarized filters
    utool.parallel_compute(compute_chip, **pcc_kwargs)


# Main Script
@utool.indent_decor('[cc2]')
@profile
def load_chips(ibs, cid_list=None, force_compute=False, **kwargs):
    print('=============================')
    print('[cc2] Precomputing chips and loading chip paths: %r' % ibs.get_db_name())
    #----------------
    # COMPUTE SETUP
    #----------------
    chip_cfg = ibs.prefs.chip_cfg
    chip_uid = chip_cfg.get_uid()
    if ibs.cpaths.chip_uid != '' and ibs.cpaths.chip_uid != chip_uid:
        print('[cc2] Disagreement: OLD_chip_uid = %r' % ibs.cpaths.chip_uid)
        print('[cc2] Disagreement: NEW_chip_uid = %r' % chip_uid)
        print('[cc2] Unloading all chip information')
        ibs.unload_all()
    print('[cc2] chip_uid = %r' % chip_uid)
    # Get the list of chips paths to load
    cid_list = ibs.get_valid_cxs() if cid_list is None else cid_list
    if not np.iterable(cid_list):
        cid_list = [cid_list]
    print('[cc2] len(cid_list) = %r' % len(cid_list))
    if len(cid_list) == 0:
        return  # HACK
    cid_list = np.array(cid_list)  # HACK
    ibs.cpaths.chip_uid = chip_uid
    #print('[cc2] Requested %d chips' % (len(cid_list)))
    #print('[cc2] cid_list = %r' % (cid_list,))
    # Get table information
    try:
        gid_list    = ibs.tables.cid2_gx[cid_list]
        cid_list   = ibs.tables.cid2_cid[cid_list]
        theta_list = ibs.tables.cid2_theta[cid_list]
        roi_list   = ibs.tables.cid2_roi[cid_list]
        #gname_list = ibs.tables.gid2_gname[gid_list]
    except IndexError as ex:
        print(repr(ex))
        print(ibs.tables)
        print('cid_list=%r' % (cid_list,))
        raise
    # Get ChipConfig Parameters
    sqrt_area   = chip_cfg['chip_sqrt_area']

    filter_list = []
    if chip_cfg['adapteq']:
        filter_list.append(adapteq_fn)
    if chip_cfg['histeq']:
        filter_list.append(histeq_fn)
    if chip_cfg['region_norm']:
        filter_list.append(region_norm_fn)
    #if chip_cfg['maxcontrast']:
        #filter_list.append(maxcontr_fn)
    #if chip_cfg['rank_eq']:
        #filter_list.append(rankeq_fn)
    #if chip_cfg['local_eq']:
        #filter_list.append(localeq_fn)
    if chip_cfg['grabcut']:
        filter_list.append(grabcut_fn)

    #---------------------------
    # ___Normalized Chip Args___
    #---------------------------
    # Full Image Paths: where to extract the chips from
    gfpath_list = ibs.gid2_gname(gid_list, full=True)
    #img_dir = ibs.dirs.img_dir
    #gfpath_list = [join(img_dir, gname) for gname in iter(gname_list)]
    # Chip Paths: where to write extracted chips to
    _cfname_fmt = 'cid%d' + chip_uid + chip_cfg['chipfmt']
    _cfpath_fmt = join(ibs.dirs.chip_dir, _cfname_fmt)
    cfpath_list = [_cfpath_fmt  % cid for cid in iter(cid_list)]
    # Normalized Chip Sizes: ensure chips have about sqrt_area squared pixels
    chipsz_list = compute_uniform_area_chip_sizes(roi_list, sqrt_area)

    #--------------------------
    # EXTRACT AND RESIZE CHIPS
    #--------------------------
    # Compute all chips with paramatarized filters
    pcc_kwargs = {
        'func': compute_chip,
        'arg_list': [gfpath_list, cfpath_list, roi_list, theta_list, chipsz_list],
        'lazy': not params.args.nocache_chips and (not force_compute),
        'num_procs': params.args.num_procs,
        'common_args': [filter_list],
    }
    utool.parallel_compute(**pcc_kwargs)

    # Read sizes
    # RCOS TODO: This is slow. We need to cache this data.
    try:
        # Hackish way to read images sizes a little faster.
        # change the directory so the os doesnt have to do as much work
        import os
        cwd = os.getcwd()
        os.chdir(ibs.dirs.chip_dir)
        cfname_list = [_cfname_fmt  % cid for cid in iter(cid_list)]
        rsize_list = [(None, None) if path is None else Image.open(path).size
                      for path in iter(cfname_list)]
        os.chdir(cwd)
    except IOError as ex:
        import gc
        gc.collect()
        print('[cc2] ex=%r' % ex)
        print('path=%r' % path)
        if utool.checkpath(path, verbose=True):
            import time
            time.sleep(1)  # delays for 1 seconds
            print('[cc2] file exists but cause IOError?')
            print('[cc2] probably corrupted. Removing it')
            try:
                utool.remove_file(path)
            except OSError:
                print('Something bad happened')
                raise
        raise
    #----------------------
    # UPDATE API VARIABLES
    #----------------------
    print('[cc2] Done Precomputing chips and loading chip paths')

    # Extend the datastructure if needed
    list_size = max(cid_list) + 1
    #utool.ensure_list_size(ibs.cpaths.cid2_chip_path, list_size)
    utool.ensure_list_size(ibs.cpaths.cid2_rchip_path, list_size)
    utool.ensure_list_size(ibs.cpaths.cid2_rchip_size, list_size)
    # Copy the values into the ChipPaths object
    #for lx, cid in enumerate(cid_list):
        #ibs.cpaths.cid2_chip_path[cid] = cfpath_list[lx]
    for lx, cid in enumerate(cid_list):
        ibs.cpaths.cid2_rchip_path[cid] = cfpath_list[lx]
    for lx, cid in enumerate(cid_list):
        ibs.cpaths.cid2_rchip_size[cid] = rsize_list[lx]
    #ibs.load_cx2_rchip_size()  # TODO: Loading rchip size should be handled more robustly
    print('[cc2]=============================')
