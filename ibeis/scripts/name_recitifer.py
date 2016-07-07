# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut


def reasign_names1(ibs, aid_list=None, old_img2_names=None, common_prefix=''):
    r"""
    Changes the names in the IA-database to correspond to an older naming
    convention.  If splits and merges were preformed tries to find the
    maximally consistent renaming scheme.

    Notes:
        For each annotation:
        * get the image
        * get the image full path
        * strip the full path down to the file name prefix:
             [ example /foo/bar/pic.jpg -> pic ]
        * make the name of the individual associated with that annotation be the
          file name prefix
        * save the new names to the image analysis database
        * wildbook will make a request to get all of the annotations, image
          file names, image names and animal ids

    CommandLine:
        python -m ibeis.scripts.name_recitifer rectify_names --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.name_recitifer import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = None
        >>> common_prefix = ''
        >>> old_img2_names = None #['img_fred.png', ']
        >>> result = reasign_names1(ibs, aid_list, img_list, name_list)
    """
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    # Group annotations by their current IA-name
    nid_list = ibs.get_annot_name_rowids(aid_list)
    nid2_aids = ut.group_items(aid_list, nid_list)
    unique_nids = list(nid2_aids.keys())
    grouped_aids = list(nid2_aids.values())

    # Get grouped images
    grouped_imgnames = ibs.unflat_map(ibs.get_annot_image_names, grouped_aids)

    # Assume a mapping from old image names to old names is given.
    # Or just hack it in the Lewa case.
    if old_img2_names is None:
        def get_name_from_gname(gname):
            from os.path import splitext
            gname_, ext = splitext(gname)
            assert gname_.startswith(common_prefix), 'prefix assumption is invalidated'
            gname_ = gname_[len(common_prefix):]
            return gname_
        # Create mapping from image name to the desired "name" for the image.
        old_img2_names = {gname: get_name_from_gname(gname)
                          for gname in ut.flatten(grouped_imgnames)}

    # Make the name of the individual associated with that annotation be the file name prefix
    grouped_oldnames = [ut.take(old_img2_names, gnames) for gnames in grouped_imgnames]

    # The task is now to map each name in unique_nids to one of these names
    # subject to the contraint that each name can only be used once.  This is
    # solved using a maximum bipartite matching. The new names are the left
    # nodes, the old name are the right nodes, and grouped_oldnames definse the
    # adjacency matrix.
    # NOTE: In rare cases it may be impossible to find a correct labeling using
    # only old names.  In this case new names will be created.
    new_name_text = find_consistent_labeling(grouped_oldnames)

    dry = False
    if not dry:
        # Save the new names to the image analysis database
        ibs.set_name_texts(unique_nids, new_name_text)


def reasign_names2(ibs, gname_name_pairs, aid_list=None):
    """

    Notes:
        * Given a list of pairs:  image file names (full path), animal name.
        * Go through all the images in the database and create a dictionary that
          associates the file name (full path) of the image in the database with the
          annotation or annotations associated with that image.
        * Go through the list of pairs:
          For each image file name, look up in the dictionary the image file
          name and assign the annotation associated with the image file name
          the animal name
        * Throughout this, keep a list of annotations that have been changed
        * Wildbook will issue a pull request to get these annotation.

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.name_recitifer import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = None
        >>> common_prefix = ''
        >>> gname_name_pairs = [
        >>>     ('easy1.JPG', 'easy'),
        >>>     ('easy2.JPG', 'easy'),
        >>>     ('easy3.JPG', 'easy'),
        >>>     ('hard1.JPG', 'hard')
        >>> ]
        >>> changed_pairs = reasign_names2(gname_name_pairs)
    """
    from os.path import basename
    if aid_list is None:
        aid_list = ibs.get_valid_aids()
    annot_gnames = ibs.get_annot_image_names(aid_list)
    # Other image name getters that may be useful
    # ibs.get_annot_image_paths(aid_list)
    # ibs.get_image_uris_original(ibs.get_annot_gids(aid_list))
    gname2_aids = ut.group_items(aid_list, annot_gnames)

    changed_aids = []
    changed_names = []

    for gname, name in gname_name_pairs:
        # make sure its just the last part of the name.
        # Ignore preceding path
        gname = basename(gname)
        aids = gname2_aids[gname]
        texts = ibs.get_annot_name_texts(aids)
        flags = [text != name for text in texts]
        aids_ = ut.compress(aids, flags)
        if len(aids_):
            changed_aids.extend(aids_)
            changed_names.extend([name] * len(aids_))

    dry = False
    if not dry:
        # Save the new names to the image analysis database
        ibs.set_annot_name_texts(changed_aids, changed_names)

    # Returned list tells you who was changed.
    changed_pairs = list(zip(changed_names, changed_aids))
    return changed_pairs


def find_consistent_labeling(grouped_oldnames):
    """
    Solves a a maximum bipirtite matching problem to find a consistent
    name assignment.

    Notes:
        # Install module containing the Hungarian algorithm for matching
        pip install munkres

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.name_recitifer import *  # NOQA
        >>> grouped_oldnames = [['a', 'b'], ['b', 'c'], ['c', 'a', 'a']]
        >>> new_names = find_consistent_labeling(grouped_oldnames)
        >>> print(new_names)
        [u'b', u'c', u'a']

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.name_recitifer import *  # NOQA
        >>> grouped_oldnames = [['a', 'b', 'c'], ['b', 'c'], ['c', 'e', 'e']]
        >>> new_names = find_consistent_labeling(grouped_oldnames)
        >>> print(new_names)
        [u'a', u'b', u'e']

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.name_recitifer import *  # NOQA
        >>> grouped_oldnames = [['a', 'b'], ['a', 'a', 'b'], ['a']]
        >>> new_names = find_consistent_labeling(grouped_oldnames)
        >>> print(new_names)
        [u'a', u'b', u'e']
    """
    import numpy as np
    try:
        import munkres
    except ImportError:
        print('Need to install Hungrian algorithm bipartite matching solver.')
        print('Run:')
        print('pip install munkres')
        raise
    unique_old_names = ut.unique(ut.flatten(grouped_oldnames))
    num_new_names = len(grouped_oldnames)
    num_old_names = len(unique_old_names)
    extra_oldnames = []

    # Create padded dummy values.  This accounts for the case where it is
    # impossible to uniquely map to the old db
    num_extra = num_new_names - num_old_names
    if num_extra > 0:
        extra_oldnames = ['_extra_name%d' % (count,) for count in
                          range(num_extra)]
    elif num_extra < 0:
        pass
    else:
        extra_oldnames = []
    assignable_names = unique_old_names + extra_oldnames

    total = len(assignable_names)

    # Allocate assignment matrix
    profit_matrix = np.zeros((total, total), dtype=np.int)
    # Populate assignment profit matrix
    oldname2_idx = ut.make_index_lookup(assignable_names)
    name_freq_list = [ut.dict_hist(names) for names in grouped_oldnames]
    for rowx, name_freq in enumerate(name_freq_list):
        for name, freq in name_freq.items():
            colx = oldname2_idx[name]
            profit_matrix[rowx, colx] += freq
    # Add extra profit for using a previously used name
    profit_matrix[profit_matrix > 0] += 2
    # Add small profit for using an extra name
    extra_colxs = ut.take(oldname2_idx, extra_oldnames)
    profit_matrix[:, extra_colxs] += 1

    # Convert to minimization problem
    big_value = (profit_matrix.max())
    cost_matrix = big_value - profit_matrix
    m = munkres.Munkres()
    indexes = m.compute(cost_matrix)

    # Map output to be aligned with input
    rx2_cx = dict(indexes)
    assignment = [assignable_names[rx2_cx[rx]]
                  for rx in range(num_new_names)]
    return assignment
