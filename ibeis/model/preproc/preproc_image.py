from __future__ import absolute_import, division, print_function
"""
# DOCTEST ENABLED
DoctestCMD:
    python -c "import doctest, ibeis; print(doctest.testmod(ibeis.model.preproc.preproc_featweight))" --quiet
"""
from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range, map  # NOQA
# UTool
import utool
import utool as ut
import vtool.patch as ptool
import vtool.image as gtool  # NOQA
#import vtool.image as gtool
import numpy as np
from ibeis.model.preproc import preproc_chip
from os.path import exists
# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_featweight]')


def gen_featweight_worker(tup):
    """
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    Args:
        tup (aid, tuple(kpts(ndarray), probchip_fpath )): keypoints and probability chip file path

    Example:
        >>> # DOCTEST ENABLE
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> ax = 12
        >>> aid_list = ibs.get_valid_aids()[ax:ax + 1]
        >>> chip_list = ibs.get_annot_chips(aid_list)
        >>> kpts_list = ibs.get_annot_kpts(aid_list)
        >>> probchip_fpath_list = preproc_chip.compute_and_write_probchip(ibs, aid_list)
        >>> probchip_list = [gtool.imread(fpath, grayscale=False) if exists(fpath) else None for fpath in probchip_fpath_list]
        >>> kpts  = kpts_list[0]
        >>> aid   = aid_list[0]
        >>> probchip = probchip_list[0]
        >>> tup = (aid, kpts, probchip)
        >>> (aid, weights) = gen_featweight_worker(tup)
        >>> print(weights.sum())
        275.025
    """
    (aid, kpts, probchip) = tup
    if probchip is None:
        # hack for undetected chips. SETS ALL FEATWEIGHTS TO .25 = 1/4
        weights = np.full(len(kpts), .25, dtype=np.float32)
    else:
        #ptool.get_warped_patches()
        patch_list = [ptool.get_warped_patch(probchip, kp)[0].astype(np.float32) / 255.0 for kp in kpts]
        weight_list = [patch.sum() / (patch.size) for patch in patch_list]
        weights = np.array(weight_list, dtype=np.float32)
    return (aid, weights)


def compute_fgweights(ibs, aid_list, qreq_=None):
    """

    Example:
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> qreq_ = None
        >>> featweight_list = compute_fgweights(ibs, aid_list)
    """
    print('[preproc_featweight] Preparing to compute fgweights')
    probchip_fpath_list = preproc_chip.compute_and_write_probchip(ibs, aid_list, qreq_=qreq_)
    if ut.DEBUG2:
        from PIL import Image
        probchip_size_list = [Image.open(fpath).size for fpath in probchip_fpath_list]
        chipsize_list = ibs.get_annot_chipsizes(aid_list)
        assert chipsize_list == probchip_size_list, 'probably need to clear chip or probchip cache'

    kpts_list = ibs.get_annot_kpts(aid_list)
    probchip_list = [gtool.imread(fpath) if exists(fpath) else None for fpath in probchip_fpath_list]

    print('[preproc_featweight] Computing fgweights')
    arg_iter = zip(aid_list, kpts_list, probchip_list)
    featweight_gen = utool.util_parallel.generate(gen_featweight_worker, arg_iter, nTasks=len(aid_list))
    featweight_param_list = list(featweight_gen)
    #arg_iter = zip(aid_list, kpts_list, probchip_list)
    #featweight_param_list1 = [gen_featweight_worker((aid, kpts, probchip)) for aid, kpts, probchip in arg_iter]
    #featweight_aids = ut.get_list_column(featweight_param_list, 0)
    featweight_list = ut.get_list_column(featweight_param_list, 1)
    print('[preproc_featweight] Done computing fgweights')
    return featweight_list


def add_featweight_params_gen(ibs, fid_list, qreq_=None):
    """
    add_featweight_params_gen

    Args:
        ibs (IBEISController):
        fid_list (list):

    Returns:
        featweight_list

    Example:
        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> fid_list = ibs.get_valid_fids()
        >>> result = add_featweight_params_gen(ibs, fid_list)
        >>> print(result)
    """
    # HACK: TODO AUTOGENERATE THIS
    from ibeis import constants
    cid_list = ibs.dbcache.get(constants.FEATURE_TABLE, ('chip_rowid',), fid_list)
    aid_list = ibs.dbcache.get(constants.CHIP_TABLE, ('annot_rowid',), cid_list)
    return compute_fgweights(ibs, aid_list, qreq_=qreq_)


#def get_annot_probchip_fname_iter(ibs, aid_list):
#    """ Returns probability chip path iterator

#    Args:
#        ibs (IBEISController):
#        aid_list (list):

#    Returns:
#        probchip_fname_iter

#    Example:
#        >>> from ibeis.model.preproc.preproc_featweight import *  # NOQA
#        >>> import ibeis
#        >>> ibs = ibeis.opendb('testdb1')
#        >>> aid_list = ibs.get_valid_aids()
#        >>> probchip_fname_iter = get_annot_probchip_fname_iter(ibs, aid_list)
#        >>> probchip_fname_list = list(probchip_fname_iter)
#    """
#    cfpath_list = ibs.get_annot_cpaths(aid_list)
#    cfname_list = [splitext(basename(cfpath))[0] for cfpath in cfpath_list]
#    suffix = ibs.cfg.detect_cfg.get_cfgstr()
#    ext = '.png'
#    probchip_fname_iter = (''.join([cfname, suffix, ext]) for cfname in cfname_list)
#    return probchip_fname_iter


#def get_annot_probchip_fpath_list(ibs, aid_list):
#    cachedir = get_probchip_cachedir(ibs)
#    probchip_fname_list = get_annot_probchip_fname_iter(ibs, aid_list)
#    probchip_fpath_list = [join(cachedir, fname) for fname in probchip_fname_list]
#    return probchip_fpath_list


#class FeatWeightConfig(object):
#    # TODO: Put this in a config
#    def __init__(fw_cfg):
#        fw_cfg.sqrt_area   = 800
# UTool
import utool
import vtool.exif as exif
from PIL import Image
from os.path import splitext, basename
import numpy as np  # NOQA
import hashlib
import uuid
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[preproc_img]', DEBUG=False)


#@profile
def parse_exif(pil_img):
    """ Image EXIF helper

    Cyth::
        cdef:
            Image pil_img
            dict exif_dict
            long lat
            long lon
            long exiftime
    """
    exif_dict = exif.get_exif_dict(pil_img)
    # TODO: More tags
    lat, lon = exif.get_lat_lon(exif_dict)
    time = exif.get_unixtime(exif_dict)
    return time, lat, lon


@profile
def get_image_uuid(pil_img):
    """ DEPRICATE

    Cyth::

        cdef:
            UUID uuid_
            object img_bytes_
            object bytes_sha1
            object hashbytes_16
            object hashbytes_20

    """
    # DEPRICATE
    # Read PIL image data (every 64th byte)
    img_bytes_ = np.asarray(pil_img).ravel()[::64].tostring()
    #print('[ginfo] npimg.sum() = %r' % npimg.sum())
    #img_bytes_ = np.asarray(pil_img).ravel().tostring()
    # hash the bytes using sha1
    bytes_sha1 = hashlib.sha1(img_bytes_)
    hashbytes_20 = bytes_sha1.digest()
    # sha1 produces 20 bytes, but UUID requires 16 bytes
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)
    #uuid_ = uuid.uuid4()
    #print('[ginfo] hashbytes_16 = %r' % (hashbytes_16,))
    #print('[ginfo] uuid_ = %r' % (uuid_,))
    return uuid_


def get_standard_ext(gpath):
    """ Returns standardized image extension

    Cyth::
        cdef:
            str gpath
            str ext

    """
    ext = splitext(gpath)[1].lower()
    return '.jpg' if ext == '.jpeg' else ext


@profile
def parse_imageinfo(tup):
    """ Worker function: gpath must be in UNIX-PATH format!

    Input:
        a tuple of arguments (so the function can be parallelized easily)

    Returns:
        if successful returns a tuple of image parameters which are values for
        SQL columns on else returns None

    Cyth::

        cdef:
            str gpath
            Image pil_img
            str orig_gname
            str ext
            long width
            long height
            long time
            long lat
            long lon
            str notes

    """
    # Parse arguments from tuple
    gpath = tup
    #print('[ginfo] gpath=%r' % gpath)
    # Try to open the image
    try:
        pil_img = Image.open(gpath, 'r')  # Open PIL Image
    except IOError as ex:
        print('[preproc] IOError: %s' % (str(ex),))
        return None
    # Parse out the data
    width, height  = pil_img.size         # Read width, height
    time, lat, lon = parse_exif(pil_img)  # Read exif tags
    # We cannot use pixel data as libjpeg is not determenistic (even for reads!)
    image_uuid = utool.get_file_uuid(gpath)  # Read file ]-hash-> guid = gid
    #orig_gpath = gpath
    orig_gname = basename(gpath)
    ext = get_standard_ext(gpath)
    notes = ''
    # Build parameters tuple
    param_tup = (
        image_uuid,
        gpath,
        orig_gname,
        #orig_gpath,
        ext,
        width,
        height,
        time,
        lat,
        lon,
        notes
    )
    #print('[ginfo] %r %r' % (image_uuid, orig_gname))
    return param_tup


@profile
def add_images_params_gen(gpath_list, **kwargs):
    """
    generates values for add_images sqlcommands asychronously

    Examples:
        >>> from ibeis.all_imports import *
        >>> gpath_list = grabdata.get_test_gpaths(ndata=3) + ['doesnotexist.jpg']
        >>> params_list = list(preproc_image.add_images_params_gen(gpath_list))

    Cyth::
        cdef:
            list gpath_list
            dict kwargs

    """
    #preproc_args = [(gpath, kwargs) for gpath in gpath_list]
    #print('[about to parse]: gpath_list=%r' % (gpath_list,))
    params_gen = utool.generate(parse_imageinfo, gpath_list, **kwargs)
    return params_gen


def on_delete(ibs, featweight_rowid_list, qreq_=None):
    print('Warning: Not Implemented')
