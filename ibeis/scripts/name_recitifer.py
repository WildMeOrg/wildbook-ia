# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut


def rectify_names(ibs, aid_list=None, old_img2_names=None, hack_prefix=''):
    r"""
    Changes the names in the IA-database to correspond to an older naming
    convention.  If splits and merges were preformed tries to find the
    maximally consistent renaming scheme.

    Args:
        ibs (ibeis.IBEISController): image analysis api
        aid_list (list):  list of annotation rowids
        img_list (list):
        name_list (list): (default = None)

    CommandLine:
        python -m ibeis.scripts.name_recitifer rectify_names --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.scripts.name_recitifer import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = None
        >>> hack_prefix = ''
        >>> old_img2_names = None #['img_fred.png', ']
        >>> result = rectify_names(ibs, aid_list, img_list, name_list)
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
        def hackkey(gname):
            from os.path import splitext
            gname_, ext = splitext(gname)
            gname_.lstrip(hack_prefix)
            return gname_
        # Create mapping from image name to the desired "name" for the image.
        old_img2_names = {gname: hackkey(gname) for gname in ut.flatten(grouped_imgnames)}

    # Find which old names correspond to the current IA-name grouping
    grouped_oldnames = [ut.take(old_img2_names, gnames) for gnames in grouped_imgnames]

    # The task is now to map each name in unique_nids to one of these names
    # subject to the contraint that each name can only be used once.
    # This is solved using a maximum bipartite matching. The new names are the left nodes,
    # the old name are the right nodes, and grouped_oldnames definse the adjacency matrix.
    # NOTE: In rare cases it may be impossible to find a correct labeling using
    # only old names.  In this case new names will be created.
    new_name_text = find_consistent_labeling(grouped_oldnames)

    dry = False
    if not dry:
        # Update the state of the image analysis database
        ibs.set_name_texts(unique_nids, new_name_text)


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
