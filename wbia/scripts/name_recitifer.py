# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def reasign_names1(ibs, aid_list=None, old_img2_names=None, common_prefix=''):
    r"""
    Changes the names in the IA-database to correspond to an older
    naming convention.  If splits and merges were preformed tries to
    find the maximally consistent renaming scheme.

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
        python -m wbia.scripts.name_recitifer rectify_names --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
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
            gname_ = gname_[len(common_prefix) :]
            return gname_

        # Create mapping from image name to the desired "name" for the image.
        old_img2_names = {
            gname: get_name_from_gname(gname) for gname in ut.flatten(grouped_imgnames)
        }

    # Make the name of the individual associated with that annotation be the
    # file name prefix
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
        * Go through all the images in the database and create a dictionary
        that associates the file name (full path) of the image in the database
        with the
          annotation or annotations associated with that image.
        * Go through the list of pairs:
          For each image file name, look up in the dictionary the image file
          name and assign the annotation associated with the image file name
          the animal name
        * Throughout this, keep a list of annotations that have been changed
        * Wildbook will issue a pull request to get these annotation.

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
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


def testdata_oldnames(
    n_incon_groups=10,
    n_con_groups=2,
    n_per_con=5,
    n_per_incon=5,
    con_sep=4,
    n_empty_groups=0,
):
    import numpy as np

    rng = np.random.RandomState(42)

    rng.randint(1, con_sep + 1)

    n_incon_labels = rng.randint(0, n_incon_groups + 1)
    incon_labels = list(range(n_incon_labels))

    # Build up inconsistent groups that may share labels with other groups
    n_per_incon_list = [
        rng.randint(min(2, n_per_incon), n_per_incon + 1) for _ in range(n_incon_groups)
    ]
    incon_groups = [
        rng.choice(incon_labels, n, replace=True).tolist() for n in n_per_incon_list
    ]

    # Build up consistent groups that may have multiple lables, but does not
    # share labels with any other group
    con_groups = []
    offset = n_incon_labels + 1
    for _ in range(n_con_groups):
        this_n_per = rng.randint(1, n_per_con + 1)
        this_n_avail = rng.randint(1, con_sep + 1)
        this_avail_labels = list(range(offset, offset + this_n_avail))
        this_labels = rng.choice(this_avail_labels, this_n_per, replace=True)
        con_groups.append(this_labels.tolist())
        offset += this_n_avail

    empty_groups = [[] for _ in range(n_empty_groups)]

    grouped_oldnames = incon_groups + con_groups + empty_groups
    # rng.shuffle(grouped_oldnames)
    return grouped_oldnames


def simple_munkres(part_oldnames):
    """
    Defines a munkres problem to solve name rectification.

    Notes:
        We create a matrix where each rows represents a group of annotations in
        the same PCC and each column represents an original name. If there are
        more PCCs than original names the columns are padded with extra values.
        The matrix is first initialized to be negative infinity representing
        impossible assignments. Then for each column representing a padded
        name, we set we its value to $1$ indicating that each new name could be
        assigned to a padded name for some small profit.  Finally, let $f_{rc}$
        be the the number of annotations in row $r$ with an original name of
        $c$. Each matrix value $(r, c)$ is set to $f_{rc} + 1$ if $f_{rc} > 0$,
        to represent how much each name ``wants'' to be labeled with a
        particular original name, and the extra one ensures that these original
        names are always preferred over padded names.

    CommandLine:
        python -m wbia.scripts.name_recitifer simple_munkres

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> part_oldnames = [['a', 'b'], ['b', 'c'], ['c', 'a', 'a']]
        >>> new_names = simple_munkres(part_oldnames)
        >>> result = ut.repr2(new_names)
        >>> print(new_names)
        ['b', 'c', 'a']

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> part_oldnames = [[], ['a', 'a'], [],
        >>>                  ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'b'], ['a']]
        >>> new_names = simple_munkres(part_oldnames)
        >>> result = ut.repr2(new_names)
        >>> print(new_names)
        [None, 'a', None, 'b', None]

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> part_oldnames = [[], ['b'], ['a', 'b', 'c'], ['b', 'c'], ['c', 'e', 'e']]
        >>> new_names = find_consistent_labeling(part_oldnames)
        >>> result = ut.repr2(new_names)
        >>> print(new_names)
        ['_extra_name0', 'b', 'a', 'c', 'e']

        Profit Matrix
            b   a   c   e  _0
        0 -10 -10 -10 -10   1
        1   2 -10 -10 -10   1
        2   2   2   2 -10   1
        3   2 -10   2 -10   1
        4 -10 -10   2   3   1
    """
    import numpy as np
    import scipy.optimize

    unique_old_names = ut.unique(ut.flatten(part_oldnames))
    num_new_names = len(part_oldnames)
    num_old_names = len(unique_old_names)

    # Create padded dummy values.  This accounts for the case where it is
    # impossible to uniquely map to the old db
    num_pad = max(num_new_names - num_old_names, 0)
    total = num_old_names + num_pad
    shape = (total, total)

    # Allocate assignment matrix.
    # rows are new-names and cols are old-names.
    # Initially the profit of any assignment is effectively -inf
    # This effectively marks all assignments as invalid
    profit_matrix = np.full(shape, -2 * total, dtype=np.int)
    # Overwrite valid assignments with positive profits
    oldname2_idx = ut.make_index_lookup(unique_old_names)
    name_freq_list = [ut.dict_hist(names) for names in part_oldnames]
    # Initialize profit of a valid assignment as 1 + freq
    # This incentivizes using a previously used name
    for rowx, name_freq in enumerate(name_freq_list):
        for name, freq in name_freq.items():
            colx = oldname2_idx[name]
            profit_matrix[rowx, colx] = freq + 1
    # Set a much smaller profit for using an extra name
    # This allows the solution to always exist
    profit_matrix[:, num_old_names:total] = 1

    # Convert to minimization problem
    big_value = (profit_matrix.max()) - (profit_matrix.min())
    cost_matrix = big_value - profit_matrix

    # Use scipy implementation of munkres algorithm.
    rx2_cx = dict(zip(*scipy.optimize.linear_sum_assignment(cost_matrix)))

    # Each row (new-name) has now been assigned a column (old-name)
    # Map this back to the input-space (using None to indicate extras)
    cx2_name = dict(enumerate(unique_old_names))

    if False:
        import pandas as pd

        columns = unique_old_names + ['_%r' % x for x in range(num_pad)]
        print('Profit Matrix')
        print(pd.DataFrame(profit_matrix, columns=columns))

        print('Cost Matrix')
        print(pd.DataFrame(cost_matrix, columns=columns))

    assignment_ = [cx2_name.get(rx2_cx[rx], None) for rx in range(num_new_names)]
    return assignment_


def find_consistent_labeling(grouped_oldnames, extra_prefix='_extra_name', verbose=False):
    r"""
    Solves a a maximum bipirtite matching problem to find a consistent
    name assignment that minimizes the number of annotations with different
    names. For each new grouping of annotations we assign

    For each group of annotations we must assign them all the same name, either from




    To reduce the running time

    Args:
        gropued_oldnames (list): A group of old names where the grouping is
            based on new names. For instance:

                Given:
                    aids      = [1, 2, 3, 4, 5]
                    old_names = [0, 1, 1, 1, 0]
                    new_names = [0, 0, 1, 1, 0]

                The grouping is
                    [[0, 1, 0], [1, 1]]

                This lets us keep the old names in a split case and
                re-use exising names and make minimal changes to
                current annotation names while still being consistent
                with the new and improved grouping.

                The output will be:
                    [0, 1]

                Meaning that all annots in the first group are assigned the
                name 0 and all annots in the second group are assigned the name
                1.

    References:
        http://stackoverflow.com/questions/1398822/assignment-problem-numpy

    CommandLine:
        python -m wbia.scripts.name_recitifer find_consistent_labeling


    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> grouped_oldnames = testdata_oldnames(25, 15,  5, n_per_incon=5)
        >>> new_names = find_consistent_labeling(grouped_oldnames, verbose=1)
        >>> grouped_oldnames = testdata_oldnames(0, 15,  5, n_per_incon=1)
        >>> new_names = find_consistent_labeling(grouped_oldnames, verbose=1)
        >>> grouped_oldnames = testdata_oldnames(0, 0, 0, n_per_incon=1)
        >>> new_names = find_consistent_labeling(grouped_oldnames, verbose=1)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> ydata = []
        >>> xdata = list(range(10, 150, 50))
        >>> for x in xdata:
        >>>     print('x = %r' % (x,))
        >>>     grouped_oldnames = testdata_oldnames(x, 15,  5, n_per_incon=5)
        >>>     t = ut.Timerit(3, verbose=1)
        >>>     for timer in t:
        >>>         with timer:
        >>>             new_names = find_consistent_labeling(grouped_oldnames)
        >>>     ydata.append(t.ave_secs)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> pt.qtensure()
        >>> pt.multi_plot(xdata, [ydata])
        >>> ut.show_if_requested()

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> grouped_oldnames = [['a', 'b', 'c'], ['b', 'c'], ['c', 'e', 'e']]
        >>> new_names = find_consistent_labeling(grouped_oldnames, verbose=1)
        >>> result = ut.repr2(new_names)
        >>> print(new_names)
        ['a', 'b', 'e']

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> grouped_oldnames = [['a', 'b'], ['a', 'a', 'b'], ['a']]
        >>> new_names = find_consistent_labeling(grouped_oldnames)
        >>> result = ut.repr2(new_names)
        >>> print(new_names)
        ['b', 'a', '_extra_name0']

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> grouped_oldnames = [['a', 'b'], ['e'], ['a', 'a', 'b'], [], ['a'], ['d']]
        >>> new_names = find_consistent_labeling(grouped_oldnames)
        >>> result = ut.repr2(new_names)
        >>> print(new_names)
        ['b', 'e', 'a', '_extra_name0', '_extra_name1', 'd']

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.scripts.name_recitifer import *  # NOQA
        >>> grouped_oldnames = [[], ['a', 'a'], [],
        >>>                     ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'b'], ['a']]
        >>> new_names = find_consistent_labeling(grouped_oldnames)
        >>> result = ut.repr2(new_names)
        >>> print(new_names)
        ['_extra_name0', 'a', '_extra_name1', 'b', '_extra_name2']
    """
    unique_old_names = ut.unique(ut.flatten(grouped_oldnames))
    n_old_names = len(unique_old_names)
    n_new_names = len(grouped_oldnames)

    # Initialize assignment to all Nones
    assignment = [None for _ in range(n_new_names)]

    if verbose:
        print('finding maximally consistent labeling')
        print('n_old_names = %r' % (n_old_names,))
        print('n_new_names = %r' % (n_new_names,))

    # For each old_name, determine now many new_names use it.
    oldname_sets = list(map(set, grouped_oldnames))
    oldname_usage = ut.dict_hist(ut.flatten(oldname_sets))

    # Any name used more than once is a conflict and must be resolved
    conflict_oldnames = {k for k, v in oldname_usage.items() if v > 1}

    # Partition into trivial and non-trivial cases
    nontrivial_oldnames = []
    nontrivial_new_idxs = []

    trivial_oldnames = []
    trivial_new_idxs = []
    for new_idx, group in enumerate(grouped_oldnames):
        if set(group).intersection(conflict_oldnames):
            nontrivial_oldnames.append(group)
            nontrivial_new_idxs.append(new_idx)
        else:
            trivial_oldnames.append(group)
            trivial_new_idxs.append(new_idx)

    # Rectify trivial cases
    # Any new-name that does not share any of its old-names with other
    # new-names can be resolved trivially
    n_trivial_unchanged = 0
    n_trivial_ignored = 0
    n_trivial_merges = 0
    for group, new_idx in zip(trivial_oldnames, trivial_new_idxs):
        if len(group) > 0:
            # new-names that use more than one old-name are simple merges
            h = ut.dict_hist(group)
            if len(h) > 1:
                n_trivial_merges += 1
            else:
                n_trivial_unchanged += 1
            hitems = list(h.items())
            hvals = [i[1] for i in hitems]
            maxval = max(hvals)
            g = min([k for k, v in hitems if v == maxval])
            assignment[new_idx] = g
        else:
            # new-names that use no old-names can be ignored
            n_trivial_ignored += 1

    if verbose:
        n_trivial = len(trivial_oldnames)
        n_nontrivial = len(nontrivial_oldnames)
        print('rectify %d trivial groups' % (n_trivial,))
        print('  * n_trivial_unchanged = %r' % (n_trivial_unchanged,))
        print('  * n_trivial_merges = %r' % (n_trivial_merges,))
        print('  * n_trivial_ignored = %r' % (n_trivial_ignored,))
        print('rectify %d non-trivial groups' % (n_nontrivial,))

    # Partition nontrivial_oldnames into smaller disjoint sets
    nontrivial_oldnames_sets = list(map(set, nontrivial_oldnames))
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(len(nontrivial_oldnames_sets)))
    for u, group1 in enumerate(nontrivial_oldnames_sets):
        rest = nontrivial_oldnames_sets[u + 1 :]
        for v, group2 in enumerate(rest, start=u + 1):
            if group1.intersection(group2):
                g.add_edge(u, v)
    nontrivial_partition = list(nx.connected_components(g))
    if verbose:
        print(
            '  * partitioned non-trivial into %d subgroups' % (len(nontrivial_partition))
        )
        part_size_stats = ut.get_stats(map(len, nontrivial_partition))
        stats_str = ut.repr2(part_size_stats, precision=2, strkeys=True)
        print('  * partition size stats = %s' % (stats_str,))

    # Rectify nontrivial cases
    for part_idxs in ut.ProgIter(
        nontrivial_partition, labels='rectify parts', enabled=verbose
    ):
        part_oldnames = ut.take(nontrivial_oldnames, part_idxs)
        part_newidxs = ut.take(nontrivial_new_idxs, part_idxs)
        # Rectify this part
        assignment_ = simple_munkres(part_oldnames)
        for new_idx, new_name in zip(part_newidxs, assignment_):
            assignment[new_idx] = new_name

    # Any unassigned name is now given a new unique label with a prefix
    if extra_prefix is not None:
        num_extra = 0
        for idx, val in enumerate(assignment):
            if val is None:
                assignment[idx] = '%s%d' % (extra_prefix, num_extra,)
                num_extra += 1
    return assignment


def find_consistent_labeling_old(
    grouped_oldnames, extra_prefix='_extra_name', verbose=False
):
    import numpy as np
    import scipy.optimize

    unique_old_names = ut.unique(ut.flatten(grouped_oldnames))

    # TODO: find names that are only used once, and just ignore those for
    # optimization.
    # unique_set = set(unique_old_names)
    oldname_sets = list(map(set, grouped_oldnames))
    usage_hist = ut.dict_hist(ut.flatten(oldname_sets))
    conflicts = {k for k, v in usage_hist.items() if v > 1}
    # nonconflicts = {k for k, v in usage_hist.items() if v == 1}

    conflict_groups = []
    orig_idxs = []
    assignment = [None] * len(grouped_oldnames)
    ntrivial = 0
    for idx, group in enumerate(grouped_oldnames):
        if set(group).intersection(conflicts):
            orig_idxs.append(idx)
            conflict_groups.append(group)
        else:
            ntrivial += 1
            if len(group) > 0:
                h = ut.dict_hist(group)
                hitems = list(h.items())
                hvals = [i[1] for i in hitems]
                maxval = max(hvals)
                g = min([k for k, v in hitems if v == maxval])
                assignment[idx] = g
            else:
                assignment[idx] = None

    if verbose:
        print('rectify %d non-trivial groups' % (len(conflict_groups),))
        print('rectify %d trivial groups' % (ntrivial,))

    num_extra = 0

    if len(conflict_groups) > 0:
        grouped_oldnames_ = conflict_groups
        unique_old_names = ut.unique(ut.flatten(grouped_oldnames_))
        num_new_names = len(grouped_oldnames_)
        num_old_names = len(unique_old_names)
        extra_oldnames = []

        # Create padded dummy values.  This accounts for the case where it is
        # impossible to uniquely map to the old db
        num_extra = num_new_names - num_old_names
        if num_extra > 0:
            extra_oldnames = [
                '%s%d' % (extra_prefix, count,) for count in range(num_extra)
            ]
        elif num_extra < 0:
            pass
        else:
            extra_oldnames = []
        assignable_names = unique_old_names + extra_oldnames

        total = len(assignable_names)

        # Allocate assignment matrix
        # Start with a large negative value indicating
        # that you must select from your assignments only
        profit_matrix = -np.ones((total, total), dtype=np.int) * (2 * total)
        # Populate assignment profit matrix
        oldname2_idx = ut.make_index_lookup(assignable_names)
        name_freq_list = [ut.dict_hist(names) for names in grouped_oldnames_]
        # Initialize base profit for using a previously used name
        for rowx, name_freq in enumerate(name_freq_list):
            for name, freq in name_freq.items():
                colx = oldname2_idx[name]
                profit_matrix[rowx, colx] = 1
        # Now add in the real profit
        for rowx, name_freq in enumerate(name_freq_list):
            for name, freq in name_freq.items():
                colx = oldname2_idx[name]
                profit_matrix[rowx, colx] += freq
        # Set a small profit for using an extra name
        extra_colxs = ut.take(oldname2_idx, extra_oldnames)
        profit_matrix[:, extra_colxs] = 1

        # Convert to minimization problem
        big_value = (profit_matrix.max()) - (profit_matrix.min())
        cost_matrix = big_value - profit_matrix

        # Don't use munkres, it is pure python and very slow. Use scipy instead
        indexes = list(zip(*scipy.optimize.linear_sum_assignment(cost_matrix)))

        # Map output to be aligned with input
        rx2_cx = dict(indexes)
        assignment_ = [assignable_names[rx2_cx[rx]] for rx in range(num_new_names)]

        # Reintegrate trivial values
        for idx, g in zip(orig_idxs, assignment_):
            assignment[idx] = g

    for idx, val in enumerate(assignment):
        if val is None:
            assignment[idx] = '%s%d' % (extra_prefix, num_extra,)
            num_extra += 1
    return assignment


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.scripts.name_recitifer
        python -m wbia.scripts.name_recitifer --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
