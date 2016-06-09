# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import utool as ut
import six
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[feat]', DEBUG=False)


def extract_features(img_or_fpath, feat_type='hesaff+sift', **kwargs):
    r"""
    calls pyhesaff's main driver function for detecting hessian affine keypoints.
    extra parameters can be passed to the hessian affine detector by using
    kwargs.

    Args:
        img_or_fpath (str): image file path on disk
        use_adaptive_scale (bool):
        nogravity_hack (bool):

    Returns:
        tuple : (kpts, vecs)


    CommandLine:
        python -m vtool.features --test-extract_features
        python -m vtool.features --test-extract_features --show
        python -m vtool.features --test-extract_features --feat-type=hesaff+siam128 --show
        python -m vtool.features --test-extract_features --feat-type=hesaff+siam128 --show
        python -m vtool.features --test-extract_features --feat-type=hesaff+siam128 --show --no-affine-invariance

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.features import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> img_fpath = ut.grab_test_imgpath(ut.get_argval('--fname', default='lena.png'))
        >>> imgBGR = vt.imread(img_fpath)
        >>> feat_type = ut.get_argval('--feat_type', default='hesaff+sift')
        >>> import pyhesaff
        >>> kwargs = ut.parse_dict_from_argv(pyhesaff.get_hesaff_default_params())
        >>> # execute function
        >>> #(kpts, vecs) = extract_features(img_fpath)
        >>> (kpts, vecs) = extract_features(imgBGR, feat_type, **kwargs)
        >>> # verify results
        >>> result = str((kpts, vecs))
        >>> print(result)
        >>> # Show keypoints
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> #pt.figure(fnum=1, doclf=True, docla=True)
        >>> #pt.imshow(imgBGR)
        >>> #pt.draw_kpts2(kpts, ori=True)
        >>> pt.interact_keypoints.ishow_keypoints(imgBGR, kpts, vecs, ori=True, ell_alpha=.4, color='distinct')
        >>> pt.show_if_requested()
    """
    import pyhesaff
    if feat_type == 'hesaff+sift':
        #(kpts, vecs) = pyhesaff.detect_feats(img_fpath, **kwargs)
        (kpts, vecs) = pyhesaff.detect_feats2(img_or_fpath, **kwargs)
    elif feat_type == 'hesaff+siam128':
        # hacky
        from ibeis_cnn import _plugin
        (kpts, sift) = pyhesaff.detect_feats2(img_or_fpath, **kwargs)
        if isinstance(img_or_fpath, six.string_types):
            import vtool as vt
            img_or_fpath = vt.imread(img_or_fpath)
        vecs_list = _plugin.extract_siam128_vecs([img_or_fpath], [kpts])
        vecs = vecs_list[0]
        pass
    else:
        raise AssertionError('Unknown feat_type=%r' % (feat_type,))
    return (kpts, vecs)


def get_extract_features_default_params():
    r"""
    Returns:
        dict:

    CommandLine:
        python -m vtool.features --test-get_extract_features_default_params

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.features import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> param_dict = get_extract_features_default_params()
        >>> result = ut.dict_str(param_dict)
        >>> # verify results
        >>> print(result)
    """
    import pyhesaff
    param_dict = pyhesaff.get_hesaff_default_params()
    return param_dict


def detect_opencv_keypoints():
    import cv2
    import vtool as vt
    import numpy as np  # NOQA

    #img_fpath = ut.grab_test_imgpath(ut.get_argval('--fname', default='lena.png'))
    img_fpath = ut.grab_test_imgpath(ut.get_argval('--fname', default='zebra.png'))
    imgBGR = vt.imread(img_fpath)
    imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

    def from_cv2_kpts(cv2_kp):
        kp = (cv2_kp.pt[0], cv2_kp.pt[1], cv2_kp.size, 0, cv2_kp.size, cv2_kp.angle)
        return kp

    print('\n'.join(ut.search_module(cv2, 'create', recursive=True)))

    detect_factory = {
        #'BLOB': cv2.SimpleBlobDetector_create,
        #'HARRIS' : HarrisWrapper.create,
        #'SIFT': cv2.xfeatures2d.SIFT_create,  # really DoG
        'SURF': cv2.xfeatures2d.SURF_create,  # really harris corners
        'MSER': cv2.MSER_create,
        #'StarDetector_create',

    }

    extract_factory = {
        'SIFT': cv2.xfeatures2d.SIFT_create,
        'SURF': cv2.xfeatures2d.SURF_create,
        #'DAISY': cv2.xfeatures2d.DAISY_create,
        'FREAK': cv2.xfeatures2d.FREAK_create,
        #'LATCH': cv2.xfeatures2d.LATCH_create,
        #'LUCID': cv2.xfeatures2d.LUCID_create,
        #'ORB': cv2.ORB_create,
    }
    mask = None

    type_to_kpts = {}
    type_to_desc = {}

    key = 'BLOB'
    key = 'MSER'

    for key in detect_factory.keys():
        factory = detect_factory[key]
        extractor = factory()

        # For MSERS need to adapt shape and then convert into a keypoint repr
        if hasattr(extractor, 'detectRegions'):
            # bboxes are x,y,w,h
            regions, bboxes = extractor.detectRegions(imgGray)
            # ellipse definition from [Fitzgibbon95]
            # http://www.bmva.org/bmvc/1995/bmvc-95-050.pdf p518
            # ell = [c_x, c_y, R_x, R_y, theta]
            # (cx, cy) = conic center
            # Rx and Ry = conic radii
            # theta is the counterclockwise angle
            fitz_ellipses = [cv2.fitEllipse(mser) for mser in regions]

            # http://answers.opencv.org/question/19015/how-to-use-mser-in-python/
            #hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
            #hull_ells = [cv2.fitEllipse(hull[:, 0]) for hull in hulls]
            kpts_ = []
            for ell in fitz_ellipses:
                ((cx, cy), (rx, ry), degrees) = ell
                theta = np.radians(degrees)  # opencv lives in radians
                S = vt.scale_mat3x3(rx, ry)
                T = vt.translation_mat3x3(cx, cy)
                R = vt.rotation_mat3x3(theta)
                #R = np.eye(3)
                invVR = T.dot(R.dot(S))
                kpt = vt.flatten_invV_mats_to_kpts(np.array([invVR]))[0]
                kpts_.append(kpt)
            kpts_ = np.array(kpts_)

        tt = ut.tic('Computing %r keypoints' % (key,))
        try:
            cv2_kpts = extractor.detect(imgGray, mask)
        except Exception as ex:
            ut.printex(ex, 'Failed to computed %r keypoints' % (key,), iswarning=True)
            pass
        else:
            ut.toc(tt)
            type_to_kpts[key] = cv2_kpts

    print(list(type_to_kpts.keys()))
    print(ut.depth_profile(list(type_to_kpts.values())))
    print('type_to_kpts = ' + ut.repr3(type_to_kpts, truncate=True))

    cv2_kpts = type_to_kpts['MSER']
    kp = cv2_kpts[0]  # NOQA
    #cv2.fitEllipse(cv2_kpts[0])
    cv2_kpts = type_to_kpts['SURF']

    for key in extract_factory.keys():
        factory = extract_factory[key]
        extractor = factory()
        tt = ut.tic('Computing %r descriptors' % (key,))
        try:
            filtered_cv2_kpts, desc = extractor.compute(imgGray, cv2_kpts)
        except Exception as ex:
            ut.printex(ex, 'Failed to computed %r descriptors' % (key,), iswarning=True)
            pass
        else:
            ut.toc(tt)
            type_to_desc[key] = desc

    print(list(type_to_desc.keys()))
    print(ut.depth_profile(list(type_to_desc.values())))
    print('type_to_desc = ' + ut.repr3(type_to_desc, truncate=True))


def test_mser():
    import cv2
    import vtool as vt
    import plottool as pt
    import numpy as np
    pt.qt4ensure()
    class Keypoints(ut.NiceRepr):
        """
        Convinence class for dealing with keypoints
        """
        def __init__(self, kparr, info=None):
            self.kparr = kparr
            if info is None:
                info = {}
            self.info = info

        def add_info(self, key, val):
            self.info[key] = val

        def __nice__(self):
            return ' ' + str(len(self.kparr))

        @property
        def scale(self):
            return vt.get_scales(self.kparr)

        @property
        def eccentricity(self):
            return vt.get_kpts_eccentricity(self.kparr)

        def compress(self, flags, inplace=False):
            subarr = self.kparr.compress(flags, axis=0)
            info = {key: ut.compress(val, flags) for key, val in self.info.items()}
            return Keypoints(subarr, info)

    img_fpath = ut.grab_test_imgpath(ut.get_argval('--fname', default='zebra.png'))
    imgBGR = vt.imread(img_fpath)
    imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    # http://docs.opencv.org/master/d3/d28/classcv_1_1MSER.html#gsc.tab=0
    # http://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x
    factory = cv2.MSER_create
    img_area = np.product(np.array(vt.get_size(imgGray)))
    _max_area = (img_area // 10)
    _delta = 8
    _min_diversity = .5

    extractor = factory(_delta=_delta, _max_area=_max_area, _min_diversity=_min_diversity)
    # bboxes are x,y,w,h
    regions, bboxes = extractor.detectRegions(imgGray)
    # ellipse definition from [Fitzgibbon95]
    # http://www.bmva.org/bmvc/1995/bmvc-95-050.pdf p518
    # ell = [c_x, c_y, R_x, R_y, theta]
    # (cx, cy) = conic center
    # Rx and Ry = conic radii
    # theta is the counterclockwise angle
    fitz_ellipses = [cv2.fitEllipse(mser) for mser in regions]

    # http://answers.opencv.org/question/19015/how-to-use-mser-in-python/
    #hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    #hull_ells = [cv2.fitEllipse(hull[:, 0]) for hull in hulls]
    invVR_mats = []
    for ell in fitz_ellipses:
        ((cx, cy), (dx, dy), degrees) = ell
        theta = np.radians(degrees)  # opencv lives in radians
        # Convert diameter to radians
        rx = dx / 2
        ry = dy / 2
        S = vt.scale_mat3x3(rx, ry)
        T = vt.translation_mat3x3(cx, cy)
        R = vt.rotation_mat3x3(theta)
        invVR = T.dot(R.dot(S))
        invVR_mats.append(invVR)
    invVR_mats = np.array(invVR_mats)
    #_oris = vt.get_invVR_mats_oris(invVR_mats)
    kpts2_ = vt.flatten_invV_mats_to_kpts(invVR_mats)

    self = Keypoints(kpts2_)
    self.add_info('regions', regions)
    flags = (self.eccentricity < .9)
    #flags = self.scale < np.mean(self.scale)
    #flags = self.scale < np.median(self.scale)
    self = self.compress(flags)
    import plottool as pt
    #pt.interact_keypoints.ishow_keypoints(imgBGR, self.kparr, None, ell_alpha=.4, color='distinct', fnum=2)
    #import plottool as pt
    vis = imgBGR.copy()

    for region in self.info['regions']:
        vis[region.T[1], region.T[0], :] = 0

    #regions, bbox = mser.detectRegions(gray)
    #hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in self.info['regions']]
    #cv2.polylines(vis, hulls, 1, (0, 255, 0))
    #for region in self.info['regions']:
    #    ell = cv2.fitEllipse(region)
    #    cv2.ellipse(vis, ell, (255))
    pt.interact_keypoints.ishow_keypoints(vis, self.kparr, None, ell_alpha=.4, color='distinct', fnum=2)
    #pt.imshow(vis, fnum=2)
    pt.update()

    #extractor = extract_factory['DAISY']()

    #desc_type_to_dtype = {
    #    cv2.CV_8U: np.uint8,
    #    cv2.CV_8s: np.uint,
    #}
    #def alloc_desc(extractor):
    #    desc_type = extractor.descriptorType()
    #    desc_size = extractor.descriptorSize()
    #    dtype = desc_type_to_dtype[desc_type]
    #    shape = (len(cv2_kpts), desc_size)
    #    desc = np.empty(shape, dtype=dtype)
    #    return desc

    #ut.search_module(cv2, 'array', recursive=True)
    #ut.search_module(cv2, 'freak', recursive=True)
    #ut.search_module(cv2, 'out', recursive=True)

    #cv2_kpts = cv2_kpts[0:2]

    #for key, factory in just_desc_factory_.items():
    #    extractor = factory()
    #    desc = alloc_desc(extractor)
    #    desc = extractor.compute(imgGray, cv2_kpts)
    #    feats[key] = (desc,)
    #    #extractor.compute(imgGray, cv2_kpts, desc)
    #    pass
    #kpts = np.array(list(map(from_cv2_kpts, cv2_kpts)))

    #orb = cv2.ORB()
    #kp1, des1 = orb.detectAndCompute(imgGray, None)
    #blober = cv2.SimpleBlobDetector_create()
    #haris_kpts = cv2.cornerHarris(imgGray, 2, 3, 0.04)

    #[name for name in dir(cv2) if 'mat' in name.lower()]
    #[name for name in dir(cv2.xfeatures2d) if 'desc' in name.lower()]

    #[name for name in dir(cv2) if 'detect' in name.lower()]
    #[name for name in dir(cv2) if 'extract' in name.lower()]
    #[name for name in dir(cv2) if 'ellip' in name.lower()]

    #sift = cv2.xfeatures2d.SIFT_create()
    #cv2_kpts = sift.detect(imgGray)
    #desc = sift.compute(imgGray, cv2_kpts)[1]

    #freak = cv2.xfeatures2d.FREAK_create()
    #cv2_kpts = freak.detect(imgGray)
    #desc = freak.compute(imgGray, cv2_kpts)[1]
    pass


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.features
        python -m vtool.features --allexamples
        python -m vtool.features --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
