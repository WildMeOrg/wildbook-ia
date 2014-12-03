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
    Example:
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



def train_paris_vocab(ibs):
    """
    CommandLine:
        python dev.py --db Paris --cmd
    """
    # UNFINISHED
    aid_list = []
    # use only one annotion per image
    for aids in ibs.get_image_aids(ibs.get_valid_gids()):
        if len(aids) == 1:
            aid_list.append(aids[0])
        else:
            # use annote with largest area
            aid_list.append(aids[np.argmax(ibs.get_annot_bbox_area(aids))])

    vecs_list = ibs.get_annot_vecs(aid_list)
    vecs = np.vstack(vecs_list)
    nWords = 8000
    from vtool import clustering2 as clustertool
    print('vecs are: %r' % utool.get_object_size_str(vecs))

    _words = clustertool.cached_akmeans(vecs, nWords, max_iters=500, use_cache=True, appname='smk')  # NOQA

    vec_mean = vecs.mean(axis=0).astype(np.float32)
    vec_mean.shape = (1, vec_mean.shape[0])
    vecs_centered = vecs - vec_mean
    norm_ = npl.norm(arr1, axis=1)
    norm_.shape = (norm_.size, 1)
    vecs_norm = np.divide(arr1, norm_)  # , out=out)
    print('vecs_centered are: %r' % utool.get_object_size_str(vecs_centered))
    vecs_post = np.round(128 * np.sqrt(np.abs(vecs_norm)) * np.sign(vecs_norm)).astype(np.int8)  # NOQA


def postprocess_sift(vecs, vec_mean=None):
    # UNFINISHED
    out = None
    out = vecs.astype(np.float32, copy=True)
    if vec_mean is not None:
        # Centering
        vec_mean = vec_mean.astype(np.float32)
        vec_mean.shape = (1, vec_mean.size)
        np.subtract(out, vec_mean, out=out)
    # L2 norm
    norm_ = npl.norm(out, axis=1)
    norm_.shape = (norm_.size, 1)
    np.divide(out, norm_, out=out)
    # Power Law
    sign_ = np.sign(out)
    np.abs(out, out=out)
    np.sqrt(out, out=out)
    np.multiply(out, sign_, out=out)
    # L2 norm
    norm_ = npl.norm(out, axis=1)
    norm_.shape = (norm_.size, 1)
    np.divide(out, norm_, out=out)
    # 8-bit quantization
    np.multiply(out, (127), out=out)
    np.round(out, out=out)
    final_vecs = out.astype(np.int8)
    return final_vecs


def center_descriptors():
    # UNFINISHED
    pass


