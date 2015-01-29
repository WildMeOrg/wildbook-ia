from __future__ import absolute_import, print_function, division


def extract_features(img_fpath, **kwargs):
    r"""
    calls pyhesaff's main driver function for detecting hessian affine keypoints.
    extra parameters can be passed to the hessian affine detector by using
    kwargs.

    Args:
        img_fpath (str): image file path on disk
        use_adaptive_scale (bool):
        nogravity_hack (bool):

    Returns:
        tuple : (kpts, vecs)


    CommandLine:
        python -m vtool.features --test-extract_features
        python -m vtool.features --test-extract_features --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.features import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> img_fpath = ut.grab_test_imgpath(ut.get_argval('--fname', default='lena.png'))
        >>> # execute function
        >>> (kpts, vecs) = extract_features(img_fpath)
        >>> # verify results
        >>> result = str((kpts, vecs))
        >>> print(result)
        >>> # Show keypoints
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     pt.figure(fnum=1, doclf=True, docla=True)
        >>>     imgBGR = vt.imread(img_fpath)
        >>>     pt.imshow(imgBGR)
        >>>     pt.draw_kpts2(kpts, ori=True)
        >>>     pt.show_if_requested()
    """
    import pyhesaff
    (kpts, vecs) = pyhesaff.detect_kpts(img_fpath, **kwargs)
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
    param_dict =  pyhesaff.get_hesaff_default_params()
    return param_dict


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
