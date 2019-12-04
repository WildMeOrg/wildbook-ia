"""
abstract which pyflann implementation is used

from vtool_ibeis._pyflann_backend import pyflann
"""
import ubelt as ub

__all__ = ['pyflann', 'FLANN_CLS']


try:
    import pyflann_ibeis as pyflann
    FLANN_CLS = pyflann.FLANN
except ImportError:
    try:
        import pyflann
        FLANN_CLS = pyflann.FLANN
    except ImportError:
        print('no pyflann, using cv2.flann_Index')
        import cv2

        class _CV2_FLANN_CLS:
            def __init__(self):
                self._internal = cv2.flann_Index()
                self.params = {}

            def build_index(self, features, **flann_params):
                # self._internal.build(features, flann_params, distType)
                self._internal.build(features, flann_params)

            def save_index(self, fpath):
                # self._internal.build(features, flann_params, distType)
                self._internal.save(fpath)

            def nn_index(self, query, num_neighbors, checks=ub.NoParam):
                # knnSearch(query, knn[, indices[, dists[, params]]]) -> indices, dists
                return self._internal.knnSearch(query, knn=num_neighbors)

        FLANN_CLS = _CV2_FLANN_CLS

        if 0:
            print('no pyflann, using dummy index')
            class _DUMMY_FLANN_CLS:
                def __init__(self):
                    raise RuntimeError('flann not installed')
            FLANN_CLS = _DUMMY_FLANN_CLS
