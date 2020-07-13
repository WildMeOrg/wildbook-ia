# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import six
from os.path import exists, join, realpath

(print, rrr, profile) = ut.inject2(__name__, '[grabmodels]')


# DETECTMODELS_DIR = realpath(join(dirname(__file__), 'rf'))
DEFAULT_DETECTMODELS_DIR = ut.get_app_resource_dir('wbia', 'detectmodels')

DETECTOR_KEY_RF = 'rf'

MODEL_ALGO_SUBDIRS = {
    DETECTOR_KEY_RF: 'rf',
}

MODEL_URLS = {
    DETECTOR_KEY_RF: 'https://wildbookiarepository.azureedge.net/models/rf.v3.zip',
}


def _expand_modeldir(modeldir='default'):
    """ returns default unless another path is specified """
    if modeldir == 'default':
        modeldir = DEFAULT_DETECTMODELS_DIR
    return modeldir


def get_species_trees_paths(species, modeldir='default'):
    r"""
    Args:
        species (?):
        modeldir (str):

    Returns:
        ?: trees_path

    CommandLine:
        python -m wbia.algo.detect.grabmodels --test-get_species_trees_paths

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.detect.grabmodels import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> species = wbia.const.TEST_SPECIES.ZEB_PLAIN
        >>> modeldir = 'default'
        >>> # execute function
        >>> trees_path = get_species_trees_paths(species, modeldir)
        >>> # verify results
        >>> result = str(trees_path)
        >>> print(result)
    """
    modeldir = _expand_modeldir(modeldir)
    algosubdir = MODEL_ALGO_SUBDIRS[DETECTOR_KEY_RF]
    rf_model_dir = join(modeldir, algosubdir)
    trees_path = join(rf_model_dir, species)
    return trees_path


def iter_algo_modeldirs(modeldir='default', ensurebase=False):
    modeldir = _expand_modeldir(modeldir)
    if ensurebase:
        ut.ensuredir(modeldir)
    for algo, algosubdir in six.iteritems(MODEL_ALGO_SUBDIRS):
        yield algo, join(modeldir, algosubdir)


def assert_models(modeldir='default', verbose=True):
    for algo, algo_modeldir in iter_algo_modeldirs(modeldir):
        ut.assertpath(algo_modeldir, verbose=verbose)
        # assert ut.checkpath(algo_modeldir, verbose=True), ('algo_modeldir=%r does not exist' % algo_modeldir)


def ensure_models(modeldir='default', verbose=True):
    r"""
    Args:
        modeldir (str):

    CommandLine:
        python -m wbia.algo.detect.grabmodels --test-ensure_models

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.detect.grabmodels import *  # NOQA
        >>> modeldir = 'default'
        >>> result = ensure_models(modeldir)
        >>> print(result)
    """
    modeldir = _expand_modeldir(modeldir)
    for algo, algo_modeldir in iter_algo_modeldirs(modeldir, ensurebase=True):
        if not exists(algo_modeldir):
            _download_model(algo, modeldir)
    assert_models(modeldir, verbose=verbose)


def redownload_models(modeldir='default', verbose=True):
    r"""
    Args:
        modeldir (str): (default = 'default')
        verbose (bool):  verbosity flag(default = True)

    CommandLine:
        python -m wbia.algo.detect.grabmodels --test-redownload_models

    Example:
        >>> # SCRIPT
        >>> from wbia.algo.detect.grabmodels import *  # NOQA
        >>> result = redownload_models()
    """
    print('[grabmodels] redownload_detection_models')
    modeldir = _expand_modeldir(modeldir)
    ut.delete(modeldir)
    ensure_models(modeldir=modeldir, verbose=verbose)
    if verbose:
        print('[grabmodels] finished redownload_detection_models')


def _download_model(algo, algo_modeldir):
    """
    Download and overwrites models
    """
    zip_fpath = realpath(join(algo_modeldir, algo + '.zip'))
    # Download and unzip model
    print('[grabmodels] Downloading model_dir=%s' % zip_fpath)
    model_link = MODEL_URLS[algo]
    ut.download_url(model_link, zip_fpath)
    ut.unzip_file(zip_fpath)
    # Cleanup
    ut.delete(zip_fpath)


if __name__ == '__main__':
    """

    modeldir = ibs.get_detect_modeldir()

    CommandLine:
        python -m wbia.algo.detect.grabmodels
        python -m wbia.algo.detect.grabmodels --allexamples
        python -m wbia.algo.detect.grabmodels --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
