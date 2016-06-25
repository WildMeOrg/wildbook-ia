# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
from six.moves import zip, map, filter, range  # NOQA
from functools import partial  # NOQA
from ibeis.control import controller_inject
print, rrr, profile = ut.inject2(__name__, '[annotmatch_funcs]')

# Create dectorator to inject functions in this module into the IBEISController
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)


def setup_pzmtest_subgraph():
    import ibeis
    ibs = ibeis.opendb(db='PZ_MTEST')
    nids = ibs.get_valid_nids()
    aids_list = ibs.get_name_aids(nids)

    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)

    rng = np.random.RandomState(0)
    flags = rng.rand(len(aids1)) > .878
    aids1 = ut.compress(aids1, flags)
    aids2 = ut.compress(aids2, flags)

    for aid1, aid2 in zip(aids1, aids2):
        ibs.set_annot_pair_as_positive_match(aid1, aid2)
        ibs.set_annot_pair_as_positive_match(aid2, aid1)

    rowids = ibs._get_all_annotmatch_rowids()
    aids1 = ibs.get_annotmatch_aid1(rowids)
    aids2 = ibs.get_annotmatch_aid2(rowids)


@register_ibs_method
def get_annotmatch_rowids_from_aid1(ibs, aid1_list, eager=True, nInput=None):
    """
    TODO autogenerate

    Returns a list of the aids that were reviewed as candidate matches to the input aid

    aid_list = ibs.get_valid_aids()
    Args:
        ibs (IBEISController):  ibeis controller object
        aid1_list (list):
        eager (bool): (default = True)
        nInput (None): (default = None)

    Returns:
        list: annotmatch_rowid_list
    """
    from ibeis.control import _autogen_annotmatch_funcs
    colnames = (_autogen_annotmatch_funcs.ANNOTMATCH_ROWID,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid1_list)
    andwhere_colnames = [_autogen_annotmatch_funcs.ANNOT_ROWID1]
    annotmatch_rowid_list = ibs.db.get_where2(
        ibs.const.ANNOTMATCH_TABLE, colnames, params_iter, andwhere_colnames,
        eager=eager, nInput=nInput, unpack_scalars=False)
    annotmatch_rowid_list = list(map(sorted, annotmatch_rowid_list))
    return annotmatch_rowid_list


@register_ibs_method
def get_annotmatch_rowids_from_aid2(ibs, aid2_list, eager=True, nInput=None,
                                    force_method=None):
    """
    # This one is slow because aid2 is the second part of the index

    TODO autogenerate

    Returns a list of the aids that were reviewed as candidate matches to the input aid

    aid_list = ibs.get_valid_aids()

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-get_annotmatch_rowids_from_aid2 --show

    Example2:
        >>> # TIME TEST
        >>> # setup_pzmtest_subgraph()
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> aid2_list = ibs.get_valid_aids()
        >>> func_list = [
        >>>     partial(ibs.get_annotmatch_rowids_from_aid2, force_method=1),
        >>>     partial(ibs.get_annotmatch_rowids_from_aid2, force_method=2),
        >>> ]
        >>> num_list = [1, 10, 50, 100, 300, 325, 350, 400, 500]
        >>> def args_list(count, aid2_list=aid2_list, num_list=num_list):
        >>>    return (aid2_list[0:num_list[count]],)
        >>> searchkw = dict(
        >>>     func_labels=['sql', 'numpy'],
        >>>     count_to_xtick=lambda count, args: len(args[0]),
        >>>     title='Timings of get_annotmatch_rowids_from_aid2',
        >>> )
        >>> niters = len(num_list)
        >>> time_result = ut.gridsearch_timer(func_list, args_list, niters, **searchkw)
        >>> time_result['plot_timings']()
        >>> ut.show_if_requested()
    """
    from ibeis.control import _autogen_annotmatch_funcs
    if force_method != 2 and (nInput < 128 or (force_method == 1)):
        colnames = (_autogen_annotmatch_funcs.ANNOTMATCH_ROWID,)
        # FIXME: col_rowid is not correct
        params_iter = zip(aid2_list)
        andwhere_colnames = [_autogen_annotmatch_funcs.ANNOT_ROWID2]
        annotmatch_rowid_list = ibs.db.get_where2(
            ibs.const.ANNOTMATCH_TABLE, colnames, params_iter, andwhere_colnames,
            eager=eager, nInput=nInput, unpack_scalars=False)
    elif force_method == 2:
        import vtool as vt
        all_annotmatch_rowids = np.array(ibs._get_all_annotmatch_rowids())
        aids2 = np.array(ibs.get_annotmatch_aid2(all_annotmatch_rowids))
        unique_aid2, groupxs2 = vt.group_indices(aids2)
        rowids2_ = vt.apply_grouping(all_annotmatch_rowids, groupxs2)
        rowids2_ = [_.tolist() for _ in rowids2_]
        maping2 = ut.defaultdict(list, zip(unique_aid2, rowids2_))
        annotmatch_rowid_list = ut.dict_take(maping2, aid2_list)
    annotmatch_rowid_list = list(map(sorted, annotmatch_rowid_list))
    return annotmatch_rowid_list


@register_ibs_method
@profile
def get_annotmatch_rowids_from_aid(ibs, aid_list, eager=True, nInput=None, force_method=None):
    """
    Undirected version

    TODO autogenerate

    Returns a list of the aids that were reviewed as candidate matches to the input aid

    aid_list = ibs.get_valid_aids()

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-get_annotmatch_rowids_from_aid
        python -m ibeis.annotmatch_funcs --exec-get_annotmatch_rowids_from_aid:1 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> # setup_pzmtest_subgraph()
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()[0:4]
        >>> eager = True
        >>> nInput = None
        >>> annotmatch_rowid_list = get_annotmatch_rowids_from_aid(ibs, aid_list,
        >>>                                                        eager, nInput)
        >>> result = ('annotmatch_rowid_list = %s' % (str(annotmatch_rowid_list),))
        >>> print(result)

    Example2:
        >>> # TIME TEST
        >>> # setup_pzmtest_subgraph()
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> aid_list = ibs.get_valid_aids()
        >>> from functools import partial
        >>> func_list = [
        >>>     partial(ibs.get_annotmatch_rowids_from_aid),
        >>>     partial(ibs.get_annotmatch_rowids_from_aid, force_method=1),
        >>>     partial(ibs.get_annotmatch_rowids_from_aid, force_method=2),
        >>> ]
        >>> num_list = [1, 10, 50, 100, 300, 325, 350, 400, 500]
        >>> def args_list(count, aid_list=aid_list, num_list=num_list):
        >>>    return (aid_list[0:num_list[count]],)
        >>> searchkw = dict(
        >>>     func_labels=['combo', 'sql', 'numpy'],
        >>>     count_to_xtick=lambda count, args: len(args[0]),
        >>>     title='Timings of get_annotmatch_rowids_from_aid',
        >>> )
        >>> niters = len(num_list)
        >>> time_result = ut.gridsearch_timer(func_list, args_list, niters, **searchkw)
        >>> time_result['plot_timings']()
        >>> ut.show_if_requested()
    """
    from ibeis.control import _autogen_annotmatch_funcs
    if nInput is None:
        nInput = len(aid_list)

    if force_method != 2 and (nInput < 256 or (force_method == 1)):
        rowids1 = ibs.get_annotmatch_rowids_from_aid1(aid_list)
        # This one is slow because aid2 is the second part of the index
        rowids2 = ibs.get_annotmatch_rowids_from_aid2(aid_list)
        annotmatch_rowid_list = list(map(ut.flatten, zip(rowids1, rowids2)))  # NOQA
    else:
        # This is much much faster than the other methods for large queries
        import vtool as vt
        all_annotmatch_rowids = np.array(ibs._get_all_annotmatch_rowids())
        aids1 = np.array(ibs.get_annotmatch_aid1(all_annotmatch_rowids))
        aids2 = np.array(ibs.get_annotmatch_aid2(all_annotmatch_rowids))
        unique_aid1, groupxs1 = vt.group_indices(aids1)
        unique_aid2, groupxs2 = vt.group_indices(aids2)
        rowids1_ = vt.apply_grouping(all_annotmatch_rowids, groupxs1)
        rowids2_ = vt.apply_grouping(all_annotmatch_rowids, groupxs2)
        rowids1_ = [_.tolist() for _ in rowids1_]
        rowids2_ = [_.tolist() for _ in rowids2_]
        maping1 = dict(zip(unique_aid1, rowids1_))
        maping2 = dict(zip(unique_aid2, rowids2_))
        mapping = ut.defaultdict(list, ut.dict_union3(maping1, maping2))
        annotmatch_rowid_list = ut.dict_take(mapping, aid_list)

    if False:
        # VERY SLOW
        colnames = (_autogen_annotmatch_funcs.ANNOTMATCH_ROWID,)
        # FIXME: col_rowid is not correct
        params_iter = list(zip(aid_list, aid_list))
        where_colnames = [_autogen_annotmatch_funcs.ANNOT_ROWID1, _autogen_annotmatch_funcs.ANNOT_ROWID2]
        with ut.Timer('one'):
            annotmatch_rowid_list1 = ibs.db.get_where3(  # NOQA
                ibs.const.ANNOTMATCH_TABLE, colnames, params_iter, where_colnames,
                logicop='OR', eager=eager, nInput=nInput, unpack_scalars=False)
    # Ensure funciton output is consistent
    annotmatch_rowid_list = list(map(sorted, annotmatch_rowid_list))
    return annotmatch_rowid_list


def get_annotmatch_subgraph(ibs):
    r"""
    http://bokeh.pydata.org/en/latest/
    https://github.com/jsexauer/networkx_viewer

    TODO: Need a special visualization
        In the web I need:
            * graph of annotations matches.
            * can move them around.
            * edit lines between them.
            * http://stackoverflow.com/questions/15373530/web-graph-visualization-tool

            This should  share functionality with a name view.

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-get_annotmatch_subgraph --show

        # Networkx example
        python -m ibeis.viz.viz_graph --test-show_chipmatch_graph:0 --show

    Ignore:

        from ibeis import viz

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> result = get_annotmatch_subgraph(ibs)
        >>> ut.show_if_requested()
    """
    #import ibeis
    #ibs = ibeis.opendb(db='PZ_MTEST')
    #rowids = ibs._get_all_annotmatch_rowids()
    #aids1 = ibs.get_annotmatch_aid1(rowids)
    #aids2 = ibs.get_annotmatch_aid2(rowids)
    #
    #
    nids = ibs.get_valid_nids()
    nids = nids[0:5]
    aids_list = ibs.get_name_aids(nids)
    import itertools
    unflat_edges = (list(itertools.product(aids, aids)) for aids in aids_list)
    aid_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]
    aids1 = ut.get_list_column(aid_pairs, 0)
    aids2 = ut.get_list_column(aid_pairs, 1)

    # Enumerate annotmatch properties
    rng = np.random.RandomState(0)
    edge_props = {
        'weight': rng.rand(len(aids1)),
        'reviewer_confidence': rng.rand(len(aids1)),
        'algo_confidence': rng.rand(len(aids1)),
    }

    # Remove data that does not need to be visualized
    # (dont show all the aids if you dont have to)
    thresh = .5
    flags = edge_props['weight'] > thresh
    aids1_ = ut.compress(aids1, flags)
    aids2_ = ut.compress(aids2, flags)
    chosen_props = ut.dict_subset(edge_props, ['weight'])
    edge_props = ut.map_dict_vals(ut.partial(ut.compress, flag_list=flags), chosen_props)

    edge_keys = list(edge_props.keys())
    edge_vals = ut.dict_take(edge_props, edge_keys)
    edge_attr_list = [dict(zip(edge_keys, vals_)) for vals_ in zip(*edge_vals)]

    unique_aids = list(set(aids1_ + aids2_))
    # Make a graph between the chips
    nodes = unique_aids
    edges = list(zip(aids1_, aids2_, edge_attr_list))
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    from ibeis.viz import viz_graph
    fnum = None
    #zoom = kwargs.get('zoom', .4)
    viz_graph.viz_netx_chipgraph(ibs, graph, fnum=fnum, with_images=True, augment_graph=False)


@register_ibs_method
def set_annot_pair_as_reviewed(ibs, aid1, aid2):
    """ denote that this match was reviewed and keep whatever status it is given """
    isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
    if isunknown1 or isunknown2:
        truth = ibs.const.TRUTH_UNKNOWN
    else:
        nid1, nid2 = ibs.get_annot_name_rowids((aid1, aid2))
        truth = ibs.const.TRUTH_MATCH if (nid1 == nid2) else ibs.const.TRUTH_NOT_MATCH

    #annotmatch_rowid = ibs.get_annotmatch_rowid_from_superkey([aid1], [aid2])[0]
    annotmatch_rowids = ibs.add_annotmatch([aid1], [aid2])
    ibs.set_annotmatch_reviewed(annotmatch_rowids, [True])

    # Old functionality, remove. Reviewing should not set truth
    confidence  = 0.5
    #ibs.add_or_update_annotmatch(aid1, aid2, truth, confidence)
    ibs.set_annotmatch_truth(annotmatch_rowids, [truth])
    ibs.set_annotmatch_confidence(annotmatch_rowids, [confidence])
    print('... set truth=%r' % (truth,))

    #if annotmatch_rowid is not None:
    #    ibs.set_annotmatch_truth([annotmatch_rowid], [truth])
    #    ibs.set_annotmatch_confidence([annotmatch_rowid], [confidence])
    #else:
    #    ibs.add_annotmatch([aid1], [aid2], annotmatch_truth_list=[truth], annotmatch_confidence_list=[confidence])


#@register_ibs_method
#def add_or_update_annotmatch(ibs, aid1, aid2, truth, confidence):
#    annotmatch_rowid = ibs.get_annotmatch_rowid_from_superkey([aid1], [aid2])[0]
#    # TODO: sql add or update?
#    if annotmatch_rowid is not None:
#        ibs.set_annotmatch_truth([annotmatch_rowid], [truth])
#        ibs.set_annotmatch_confidence([annotmatch_rowid], [confidence])
#    else:
#        ibs.add_annotmatch([aid1], [aid2], annotmatch_truth_list=[truth], annotmatch_confidence_list=[confidence])


@register_ibs_method
def set_annot_pair_as_positive_match(ibs, aid1, aid2, dryrun=False,
                                     on_nontrivial_merge=None):
    """
    Safe way to perform links. Errors on invalid operations.

    TODO: ELEVATE THIS FUNCTION
    Change into make_task_set_annot_pair_as_positive_match and it returns what
    needs to be done.

    Need to test several cases:
        uknown, unknown
        knownA, knownA
        knownB, knownA
        unknown, knownA
        knownA, unknown

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1 (int):  query annotation id
        aid2 (int):  matching annotation id

    CommandLine:
        python -m ibeis.annotmatch_funcs --test-set_annot_pair_as_positive_match

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1, aid2 = ibs.get_valid_aids()[0:2]
        >>> dryrun = True
        >>> status = set_annot_pair_as_positive_match(ibs, aid1, aid2, dryrun)
        >>> # verify results
        >>> print(status)
    """
    def _set_annot_name_rowids(aid_list, nid_list):
        if not ut.QUIET:
            print('... _set_annot_name_rowids(aids=%r, nids=%r)' % (aid_list, nid_list))
            print('... names = %r' % (ibs.get_name_texts(nid_list)))
        assert len(aid_list) == len(nid_list), 'list must correspond'
        if not dryrun:
            ibs.set_annot_name_rowids(aid_list, nid_list)
            ibs.set_annot_pair_as_reviewed(aid1, aid2)
        # Return the new annots in this name
        _aids_list = ibs.get_name_aids(nid_list)
        _combo_aids_list = [_aids + [aid] for _aids, aid, in zip(_aids_list, aid_list)]
        status = _combo_aids_list
        return status
    print('[marking_match] aid1 = %r, aid2 = %r' % (aid1, aid2))

    nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2])
    if nid1 == nid2:
        print('...images already matched')
        #truth = get_annot_pair_truth([aid1], [aid2])[0]
        #if truth != ibs.const.TRUTH_MATCH:
        status = None
        ibs.set_annot_pair_as_reviewed(aid1, aid2)
    else:
        isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
        if isunknown1 and isunknown2:
            print('...match unknown1 to unknown2 into 1 new name')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1, aid2], next_nids * 2)
        elif not isunknown1 and not isunknown2:
            print('...merge known1 into known2')
            aid1_and_groundtruth = ibs.get_annot_groundtruth(aid1, noself=False)
            aid2_and_groundtruth = ibs.get_annot_groundtruth(aid2, noself=False)
            trivial_merge = len(aid1_and_groundtruth) == 1 and len(aid2_and_groundtruth) == 1
            if not trivial_merge:
                if on_nontrivial_merge is None:
                    raise Exception('no function is set up to handle nontrivial merges!')
                else:
                    on_nontrivial_merge(ibs, aid1, aid2)
            status =  _set_annot_name_rowids(aid1_and_groundtruth, [nid2] * len(aid1_and_groundtruth))
        elif isunknown2 and not isunknown1:
            print('...match unknown2 into known1')
            status =  _set_annot_name_rowids([aid2], [nid1])
        elif isunknown1 and not isunknown2:
            print('...match unknown1 into known2')
            status =  _set_annot_name_rowids([aid1], [nid2])
        else:
            raise AssertionError('impossible state')
    return status


@register_ibs_method
def set_annot_pair_as_negative_match(ibs, aid1, aid2, dryrun=False, on_nontrivial_split=None):
    """
    TODO: ELEVATE THIS FUNCTION

    Args:
        ibs (IBEISController):  ibeis controller object
        aid1 (int):  annotation id
        aid2 (int):  annotation id
        dryrun (bool):

    CommandLine:
        python -m ibeis.annotmatch_funcs --test-set_annot_pair_as_negative_match

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1, aid2 = ibs.get_valid_aids()[0:2]
        >>> dryrun = True
        >>> # execute function
        >>> result = set_annot_pair_as_negative_match(ibs, aid1, aid2, dryrun)
        >>> # verify results
        >>> print(result)
    """
    def _set_annot_name_rowids(aid_list, nid_list):
        print('... _set_annot_name_rowids(%r, %r)' % (aid_list, nid_list))
        if not dryrun:
            ibs.set_annot_name_rowids(aid_list, nid_list)
            ibs.set_annot_pair_as_reviewed(aid1, aid2)
    nid1, nid2 = ibs.get_annot_name_rowids([aid1, aid2])
    if nid1 == nid2:
        print('images are marked as having the same name... we must tread carefully')
        aid1_groundtruth = ibs.get_annot_groundtruth(aid1, noself=True)
        if len(aid1_groundtruth) == 1 and aid1_groundtruth == [aid2]:
            # this is the only safe case for same name split
            # Change so the names are not the same
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1], next_nids)
        else:
            if on_nontrivial_split is None:
                raise Exception('no function is set up to handle nontrivial splits!')
            else:
                on_nontrivial_split(ibs, aid1, aid2)
    else:
        isunknown1, isunknown2 = ibs.is_aid_unknown([aid1, aid2])
        if isunknown1 and isunknown2:
            print('...nonmatch unknown1 and unknown2 into 2 new names')
            next_nids = ibs.make_next_nids(num=2)
            status =  _set_annot_name_rowids([aid1, aid2], next_nids)
        elif not isunknown1 and not isunknown2:
            print('...nonmatch known1 and known2... nothing to do (yet)')
            ibs.set_annot_pair_as_reviewed(aid1, aid2)
            status = None
        elif isunknown2 and not isunknown1:
            print('...nonmatch unknown2 -> newname and known1')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid2], next_nids)
        elif isunknown1 and not isunknown2:
            print('...nonmatch unknown1 -> newname and known2')
            next_nids = ibs.make_next_nids(num=1)
            status =  _set_annot_name_rowids([aid1], next_nids)
        else:
            raise AssertionError('impossible state')
    return status


@register_ibs_method
def set_annot_pair_as_unknown_match(ibs, aid1, aid2, dryrun=False, on_nontrivial_merge=None):
    pass


@register_ibs_method
def get_annot_pair_timdelta(ibs, aid_list1, aid_list2):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list1 (int):  list of annotation ids
        aid_list2 (int):  list of annotation ids

    Returns:
        list: timedelta_list

    CommandLine:
        python -m ibeis.annotmatch_funcs --test-get_annot_pair_timdelta

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids(hasgt=True)
        >>> unixtimes = ibs.get_annot_image_unixtimes(aid_list)
        >>> aid_list = ut.compress(aid_list, np.array(unixtimes) != -1)
        >>> gt_aids_list = ibs.get_annot_groundtruth(aid_list, daid_list=aid_list)
        >>> aid_list1 = [aid for aid, gt_aids in zip(aid_list, gt_aids_list) if len(gt_aids) > 0][0:5]
        >>> aid_list2 = [gt_aids[0] for gt_aids in gt_aids_list if len(gt_aids) > 0][0:5]
        >>> timedelta_list = ibs.get_annot_pair_timdelta(aid_list1, aid_list2)
        >>> result = ut.numpy_str(timedelta_list, precision=2)
        >>> print(result)
        np.array([  7.57e+07,   7.57e+07,   2.41e+06,   1.98e+08,   9.69e+07], dtype=np.float64)

    """
    #unixtime_list1 = np.array(ibs.get_annot_image_unixtimes(aid_list1), dtype=np.float)
    #unixtime_list2 = np.array(ibs.get_annot_image_unixtimes(aid_list2), dtype=np.float)
    unixtime_list1 = ibs.get_annot_image_unixtimes_asfloat(aid_list1)
    unixtime_list2 = ibs.get_annot_image_unixtimes_asfloat(aid_list2)
    #unixtime_list1[unixtime_list1 == -1] = np.nan
    #unixtime_list2[unixtime_list2 == -1] = np.nan
    timedelta_list = np.abs(unixtime_list1 - unixtime_list2)
    return timedelta_list


# AUTOGENED CONSTANTS:


@register_ibs_method
def get_annot_has_reviewed_matching_aids(ibs, aid_list, eager=True, nInput=None):
    num_reviewed_list = ibs.get_annot_num_reviewed_matching_aids(aid_list)
    has_reviewed_list = [num_reviewed > 0 for num_reviewed in num_reviewed_list]
    return has_reviewed_list


@register_ibs_method
def get_annot_num_reviewed_matching_aids(ibs, aid1_list, eager=True, nInput=None):
    r"""
    Args:
        aid_list (int):  list of annotation ids
        eager (bool):
        nInput (None):

    Returns:
        list: num_annot_reviewed_list

    CommandLine:
        python -m ibeis.annotmatch_funcs --test-get_annot_num_reviewed_matching_aids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid1_list = ibs.get_valid_aids()
        >>> eager = True
        >>> nInput = None
        >>> # execute function
        >>> num_annot_reviewed_list = get_annot_num_reviewed_matching_aids(ibs, aid_list, eager, nInput)
        >>> # verify results
        >>> result = str(num_annot_reviewed_list)
        >>> print(result)
    """
    aids_list = ibs.get_annot_reviewed_matching_aids(aid1_list, eager=eager, nInput=nInput)
    num_annot_reviewed_list = list(map(len, aids_list))
    return num_annot_reviewed_list


@register_ibs_method
def get_annot_reviewed_matching_aids(ibs, aid_list, eager=True, nInput=None):
    """
    Returns a list of the aids that were reviewed as candidate matches to the input aid

    aid_list = ibs.get_valid_aids()
    """
    ANNOT_ROWID1 = 'annot_rowid1'
    ANNOT_ROWID2 = 'annot_rowid2'
    #params_iter = [(aid, aid) for aid in aid_list]
    #[(aid, aid) for aid in aid_list]
    #colnames = (ANNOT_ROWID1, ANNOT_ROWID2)
    #where_colnames = (ANNOT_ROWID1, ANNOT_ROWID2)
    params_iter = [(aid,) for aid in aid_list]
    colnames = (ANNOT_ROWID2,)
    andwhere_colnames = (ANNOT_ROWID1,)
    aids_list = ibs.db.get_where2(ibs.const.ANNOTMATCH_TABLE, colnames,
                                  params_iter,
                                  andwhere_colnames=andwhere_colnames,
                                  eager=eager, unpack_scalars=False,
                                  nInput=nInput)
    #logicop = 'OR'
    #aids_list = ibs.db.get_where3(
    #    const.ANNOTMATCH_TABLE, colnames, params_iter,
    #    where_colnames=where_colnames, logicop=logicop, eager=eager,
    #    unpack_scalars=False, nInput=nInput)
    return aids_list


@register_ibs_method
def get_annot_pair_truth(ibs, aid1_list, aid2_list):
    """
    CAREFUL: uses annot match table for truth, so only works if reviews have happend
    """
    annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
    annotmatch_truth_list = ibs.get_annotmatch_truth(annotmatch_rowid_list)
    return annotmatch_truth_list


@register_ibs_method
def get_annot_pair_is_reviewed(ibs, aid1_list, aid2_list):
    r"""
    Args:
        aid1_list (list):
        aid2_list (list):

    Returns:
        list: annotmatch_reviewed_list

    CommandLine:
        python -m ibeis.annotmatch_funcs --test-get_annot_pair_is_reviewed

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb2')
        >>> aid_list = ibs.get_valid_aids()
        >>> pairs = list(ut.product(aid_list, aid_list))
        >>> aid1_list = ut.get_list_column(pairs, 0)
        >>> aid2_list = ut.get_list_column(pairs, 1)
        >>> # execute function
        >>> annotmatch_reviewed_list = get_annot_pair_is_reviewed(ibs, aid1_list, aid2_list)
        >>> # verify results
        >>> reviewed_pairs = ut.compress(pairs, annotmatch_reviewed_list)
        >>> result = len(reviewed_pairs)
        >>> print(result)
        104
    """
    annotmatch_truth_list1 = ibs.get_annot_pair_truth(aid1_list, aid2_list)
    annotmatch_truth_list2 = ibs.get_annot_pair_truth(aid2_list, aid1_list)
    annotmatch_truth_list = ut.or_lists(
        ut.flag_not_None_items(annotmatch_truth_list1),
        ut.flag_not_None_items(annotmatch_truth_list2))
    #annotmatch_reviewed_list = [truth is not None for truth in annotmatch_truth_list]
    return annotmatch_truth_list


def review_tagged_splits():
    """

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-review_tagged_splits --show
        python -m ibeis.annotmatch_funcs --exec-review_tagged_splits --show --db

    Example:
        >>> from ibeis.gui.guiback import *  # NOQA
        >>> import numpy as np
        >>> #back = testdata_guiback(defaultdb='PZ_Master1', activate=False)
        >>> #ibs = back.ibs
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> # Find aids that still need splits
        >>> aid_pair_list = ibs.filter_aidpairs_by_tags(has_any='SplitCase')
        >>> truth_list = ibs.get_aidpair_truths(*zip(*aid_pair_list))
        >>> _aid_list = ut.compress(aid_pair_list, truth_list)
        >>> _nids_list = ibs.unflat_map(ibs.get_annot_name_rowids, _aid_list)
        >>> _nid_list = ut.get_list_column(_nids_list, 0)
        >>> import vtool as vt
        >>> split_nids, groupxs = vt.group_indices(np.array(_nid_list))
        >>> problem_aids_list = vt.apply_grouping(np.array(_aid_list), groupxs)
        >>> #
        >>> split_aids_list = ibs.get_name_aids(split_nids)
        >>> assert len(split_aids_list) > 0, 'SPLIT cases are finished'
        >>> problem_aids = problem_aids_list[0]
        >>> aid_list = split_aids_list[0]
        >>> #
        >>> print('Run splits for tagd problem cases %r' % (problem_aids))
        >>> #back.run_annot_splits(aid_list)
        >>> print('Review splits for tagd problem cases %r' % (problem_aids))
        >>> from ibeis.viz import viz_graph
        >>> nid = split_nids[0]
        >>> selected_aids = np.unique(problem_aids.ravel()).tolist()
        >>> selected_aids = [] if ut.get_argflag('--noselect') else  selected_aids
        >>> print('selected_aids = %r' % (selected_aids,))
        >>> selected_aids = []
        >>> aids = ibs.get_name_aids(nid)
        >>> self = viz_graph.make_name_graph_interaction(ibs, aids=aids,
        >>>                                              with_all=False,
        >>>                                              selected_aids=selected_aids,
        >>>                                              with_images=True,
        >>>                                              prog='neato', rankdir='LR',
        >>>                                              augment_graph=False,
        >>>                                              ensure_edges=problem_aids.tolist())
        >>> ut.show_if_requested()

        rowids = ibs.get_annotmatch_rowid_from_superkey(problem_aids.T[0], problem_aids.T[1])
        ibs.get_annotmatch_prop('SplitCase', rowids)
        #ibs.set_annotmatch_prop('SplitCase', rowids, [False])
    """
    pass


def review_subgraph(ibs, nid_list):
    r"""
    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-review_subgraph --show

    Example:
        >>> # SCRIPT
        >>> from ibeis.annotmatch_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> nid_list = ibs.get_valid_nids()[0:5]
        >>> result = review_subgraph(ibs, nid_list)
        >>> ut.show_if_requested()
    """
    from ibeis.viz import viz_graph
    self = viz_graph.make_name_graph_interaction(ibs, nid_list)
    return self


def review_tagged_joins():
    """

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-review_tagged_joins --show --db PZ_Master1
        python -m ibeis.annotmatch_funcs --exec-review_tagged_joins --show --db testdb1

    Example:
        >>> from ibeis.gui.guiback import *  # NOQA
        >>> import numpy as np
        >>> import vtool as vt
        >>> #back = testdata_guiback(defaultdb='PZ_Master1', activate=False)
        >>> #ibs = back.ibs
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> # Find aids that still need Joins
        >>> aid_pair_list = ibs.filter_aidpairs_by_tags(has_any='JoinCase')
        >>> if ibs.get_dbname() == 'testdb1':
        >>>     aid_pair_list = [[1, 2]]
        >>> truth_list_ = ibs.get_aidpair_truths(*zip(*aid_pair_list))
        >>> truth_list = truth_list_ != 1
        >>> _aid_list = ut.compress(aid_pair_list, truth_list)
        >>> _nids_list = np.array(ibs.unflat_map(ibs.get_annot_name_rowids, _aid_list))
        >>> edge_ids = vt.get_undirected_edge_ids(_nids_list)
        >>> edge_ids = np.array(edge_ids)
        >>> unique_edgeids, groupxs = vt.group_indices(edge_ids)
        >>> problem_aids_list = vt.apply_grouping(np.array(_aid_list), groupxs)
        >>> problem_nids_list = vt.apply_grouping(np.array(_nids_list), groupxs)
        >>> join_nids = [np.unique(nids.ravel()) for nids in problem_nids_list]
        >>> join_aids_list = ibs.unflat_map(ibs.get_name_aids, join_nids)
        >>> assert len(join_aids_list) > 0, 'JOIN cases are finished'
        >>> problem_aid_pairs = problem_aids_list[0]
        >>> aid_list = join_aids_list[0]
        >>> #
        >>> print('Run JOINS for taged problem cases %r' % (problem_aid_pairs))
        >>> #back.run_annot_splits(aid_list)
        >>> print('Review splits for tagd problem cases %r' % (problem_aid_pairs))
        >>> from ibeis.viz import viz_graph
        >>> nids = join_nids[0]
        >>> selected_aids = np.unique(problem_aid_pairs.ravel()).tolist()
        >>> ut.flatten(ibs.get_name_aids(nids))
        >>> aids = ibs.sample_annots_general(ut.flatten(ibs.get_name_aids(nids)), sample_per_name=4, verbose=True)
        >>> import itertools
        >>> aids = ut.unique(aids + selected_aids)
        >>> self = viz_graph.make_name_graph_interaction(ibs, aids=aids, selected_aids=selected_aids, with_all=False, invis_edges=list(itertools.combinations(selected_aids, 2)))
        >>> #self = viz_graph.make_name_graph_interaction(ibs, nids, selected_aids=selected_aids)
        >>> ut.show_if_requested()

        rowids = ibs.get_annotmatch_rowid_from_superkey(problem_aids.T[0], problem_aids.T[1])
        ibs.get_annotmatch_prop('SplitCase', rowids)
        #ibs.set_annotmatch_prop('SplitCase', rowids, [False])
    """
    pass


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.annotmatch_funcs
        python -m ibeis.annotmatch_funcs --allexamples
        python -m ibeis.annotmatch_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
