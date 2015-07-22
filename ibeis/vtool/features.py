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
        #(kpts, vecs) = pyhesaff.detect_kpts(img_fpath, **kwargs)
        (kpts, vecs) = pyhesaff.detect_kpts2(img_or_fpath, **kwargs)
    elif feat_type == 'hesaff+siam128':
        # hacky
        from ibeis_cnn import _plugin
        (kpts, sift) = pyhesaff.detect_kpts2(img_or_fpath, **kwargs)
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
    """
    import cv2
    import vtool as vt
    img_fpath = ut.grab_test_imgpath(ut.get_argval('--fname', default='lena.png'))
    imgBGR = vt.imread(img_fpath)
    gray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    x = cv2.cornerHarris(gray,2,3,0.04)

    orb = cv2.ORB()
    kp1, des1 = orb.detectAndCompute(gray, None)

    [name for name in dir(cv2) if 'detect' in name.lower()]
    [name for name in dir(cv2) if 'extract' in name.lower()]
    [name for name in dir(cv2) if 'ellip' in name.lower()]

    sift = cv2.xfeatures2d.SIFT_create()
    cv2_kpts = sift.detect(gray)
    desc = sift.compute(gray, cv2_kpts)[1]

    freak = cv2.xfeatures2d.FREAK_create()
    cv2_kpts = freak.detect(gray)
    desc = freak.compute(gray, cv2_kpts)[1]

    blober = cv2.SimpleBlobDetector_create()

    def convert_cv2_keypoint(cv2_kp):
        kp = (cv2_kp.pt[0], cv2_kp.pt[1], cv2_kp.size, 0, cv2_kp.size, cv2_kp.angle)
        return kp
    kpts = np.array(map(convert_cv2_keypoint, cv2_kpts))

    """
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
