from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip, range
# Science
import pyhesaff
# UTool
import utool


# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[preproc_feat]', DEBUG=False)


USE_OPENMP = not utool.WIN32
USE_OPENMP = False  # do not use openmp until we have the gravity vector


def gen_feat_worker(tup):
    """
    <CYTH: returns=tuple>
    cdef:
        long cid
        str cpath
        dict dict_args
        np.ndarray[kpts_t, ndims=2] kpts
        np.ndarray[desc_t, ndims=2] desc
    </CYTH>
    Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async
    """
    cid, cpath, dict_args = tup
    kpts, desc = pyhesaff.detect_kpts(cpath, **dict_args)
    return (cid, len(kpts), kpts, desc)


def gen_feat_openmp(cid_list, cfpath_list, dict_args):
    """ Compute features in parallel on the C++ side, return generator here """
    print('Detecting %r features in parallel: ' % len(cid_list))
    kpts_list, desc_list = pyhesaff.detect_kpts_list(cfpath_list, **dict_args)
    for cid, kpts, desc in zip(cid_list, kpts_list, desc_list):
        yield cid, len(kpts), kpts, desc


def add_feat_params_gen(ibs, cid_list, nInput=None, **kwargs):
    """ Computes features and yields results asynchronously:
        TODO: Remove IBEIS from this equation. Move the firewall towards the
        controller """
    if nInput is None:
        nInput = len(cid_list)
    # Get config from IBEIS controller
    feat_cfg          = ibs.cfg.feat_cfg
    dict_args         = feat_cfg.get_dict_args()
    feat_config_rowid = ibs.get_feat_config_rowid()
    cfpath_list       = ibs.get_chip_paths(cid_list)
    print('[preproc_feat] cfgstr = %s' % feat_cfg.get_cfgstr())
    if USE_OPENMP:
        # Use Avi's openmp parallelization
        return gen_feat_openmp(cid_list, cfpath_list, dict_args, feat_config_rowid)
    else:
        # Multiprocessing parallelization
        featgen = generate_feats(cfpath_list, dict_args=dict_args,
                                 cid_list=cid_list, nInput=nInput, **kwargs)
        return ((cid, nKpts, kpts, desc, feat_config_rowid)
                for cid, nKpts, kpts, desc in featgen)


def generate_feats(cfpath_list, dict_args={}, cid_list=None, nInput=None, **kwargs):
    # chip-ids are an artifact of the IBEIS Controller. Make dummyones if needbe.
    """ Function to be parallelized by multiprocessing / joblib / whatever.
    Must take in one argument to be used by multiprocessing.map_async

    <CYTH: yeilds=tuple>
    cdef:
        list cfpath_list
        long nInput
        object cid_list
        dict dict_args
        dict kwargs
    </CYTH>
    """
    if cid_list is None:
        cid_list = list(range(len(cfpath_list)))
    if nInput is None:
        nInput = len(cfpath_list)
    dictargs_iter = (dict_args for _ in range(nInput))
    arg_iter = zip(cid_list, cfpath_list, dictargs_iter)
    arg_list = list(arg_iter)
    featgen = utool.util_parallel.generate(gen_feat_worker, arg_list, **kwargs)
    return featgen


def test_hdf5():
    try:
        import h5py
        import numpy as np
        data = np.array([(1, 1), (2, 2), (3, 3)], dtype=[('x', float), ('y', float)])
        h5file = h5py.File('FeatTable.h5', 'w')
        h5file.create_group('feats')
        h5file.create_dataset('FeatsTable', data=data)
        h5file['feats'].attrs['rowid'] = np.ones((4, 3, 2), 'f')
    except Exception:
        pass
    finally:
        h5file.close()


def pytable_test(cid_list, kpts_list, desc_list, weights_list):
    """
    >>> cid_list = [1, 2]
    >>> kpts_list = [1, 2]
    >>> desc_list = [1, 2]
    >>> weights_list = [1, 2]
    """
    import tables

    # Define a user record to characterize some kind of particles
    class FeatureVector(tables.IsDescription):
        feat_rowid = tables.Int32Col()
        chip_rowid = tables.Int32Col()
        feat_vecs = tables.Array
        feat_kpts = tables.Array
        feat_nKpts  = tables.Int32Col
        feat_weights = tables.Array

    filename = "test.h5"
    # Open a file in "w"rite mode
    h5file = tables.open_file(filename, mode="w", title="Test file")
    # Create a new group under "/" (root)
    group = h5file.create_group("/", 'detector', 'Detector information')
    # Create one table on it
    table = h5file.create_table(group, 'readout', FeatureVector, "Readout example")
    # Fill the table with 10 particles
    feature = table.row
    for rowid, cid, kpts, desc, weights in enumerate(zip(cid_list, kpts_list, desc_list,
                                                         weights_list)):
        feature['feat_rowid'] = rowid
        feature['chip_rowid'] = cid
        feature['feat_vecs'] = kpts
        feature['feat_kpts'] = desc
        feature['feat_nKpts'] = len(feature['feat_kpts'])
        feature['feat_weights'] = weights
        # Insert a new feature record
        feature.append()
    # Close (and flush) the file
    h5file.close()

    f = tables.open_file("test.h5")
    f.root
    f.root.detector.readout
    f.root.detector.readout.attrs.TITLE
