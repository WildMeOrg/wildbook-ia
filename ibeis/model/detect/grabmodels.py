from __future__ import absolute_import, division, print_function
import utool as ut
import six
from os.path import exists, join, realpath
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[grabmodels]', DEBUG=False)


#DETECTMODELS_DIR = realpath(join(dirname(__file__), 'rf'))
DEFAULT_DETECTMODELS_DIR = ut.get_app_resource_dir('ibeis', 'detectmodels')

DETECTOR_KEY_RF = 'rf'

MODEL_ALGO_SUBDIRS = {
    DETECTOR_KEY_RF:  'rf',
}

MODEL_URLS = {
    DETECTOR_KEY_RF: 'https://dl.dropboxusercontent.com/s/9814r3d2rkiq5t3/rf.zip'
}


def _expand_modeldir(modeldir='default'):
    """ returns default unless another path is specified """
    if modeldir == 'default':
        modeldir = DEFAULT_DETECTMODELS_DIR
    return modeldir


def get_species_trees_paths(species, modeldir='default'):
    modeldir = _expand_modeldir(modeldir)
    algosubdir = MODEL_ALGO_SUBDIRS[DETECTOR_KEY_RF]
    rf_model_dir = join(modeldir, algosubdir)
    trees_path   = join(rf_model_dir, species)
    return trees_path


def iter_algo_modeldirs(modeldir='default', ensurebase=False):
    modeldir = _expand_modeldir(modeldir)
    if ensurebase:
        ut.ensuredir(modeldir)
    for algo, algosubdir in six.iteritems(MODEL_ALGO_SUBDIRS):
        yield algo, join(modeldir, algosubdir)


def assert_models(modeldir='default'):
    for algo, algo_modeldir in iter_algo_modeldirs(modeldir):
        ut.assertpath(algo_modeldir, verbose=True)
        #assert ut.checkpath(algo_modeldir, verbose=True), ('algo_modeldir=%r does not exist' % algo_modeldir)


def ensure_models(modeldir='default'):
    r"""
    Args:
        modeldir (str):

    CommandLine:
        python -m ibeis.model.detect.grabmodels --test-ensure_models

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.detect.grabmodels import *  # NOQA
        >>> modeldir = 'default'
        >>> result = ensure_models(modeldir)
        >>> print(result)
    """
    for algo, algo_modeldir in iter_algo_modeldirs(modeldir, ensurebase=True):
        if not exists(algo_modeldir):
            _download_model(algo, algo_modeldir)
    assert_models(modeldir)


def redownload_models(modeldir='default'):
    print('[grabmodels] redownload_detection_models')
    modeldir = _expand_modeldir(modeldir)
    ut.delete(modeldir)
    ensure_models(modeldir=modeldir)
    print('[grabmodels] finished redownload_detection_models')


def _download_model(algo, algo_modeldir):
    """
    Download and overwrites models
    """
    zip_fpath = realpath(join(algo_modeldir, algo + '.zip'))
    # Download and unzip model
    print('[grabmodels] Downloading model_dir=%s' % zip_fpath)
    dropbox_link = MODEL_URLS[algo]
    ut.download_url(dropbox_link, zip_fpath)
    ut.unzip_file(zip_fpath)
    # Cleanup
    ut.delete(zip_fpath)


if __name__ == '__main__':
    """

    modeldir = ibs.get_detect_modeldir()

    CommandLine:
        python -m ibeis.model.detect.grabmodels
        python -m ibeis.model.detect.grabmodels --allexamples
        python -m ibeis.model.detect.grabmodels --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
