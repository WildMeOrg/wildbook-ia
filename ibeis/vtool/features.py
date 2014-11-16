from __future__ import absolute_import, print_function, division


def extract_features(img_fpath, **kwargs):
    """
    calls pyhesaff's main driver function for detecting hessian affine keypoints.
    extra parameters can be passed to the hessian affine detector by using
    kwargs.

    Args:
        img_fpath (str): image file path on disk
        use_adaptive_scale (bool):
        nogravity_hack (bool):

    Returns:
        tuple : (kpts, vecs)

    Example:
        >>> from pyhesaff._pyhesaff import *  # NOQA
    """
    import pyhesaff
    (kpts, vecs) = pyhesaff.detect_kpts(img_fpath, **kwargs)
    return (kpts, vecs)
