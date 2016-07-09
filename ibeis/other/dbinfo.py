# -*- coding: utf-8 -*-
"""
get_dbinfo is probably the only usefull funciton in here
# This is not the cleanest module
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
from ibeis import constants as const
import numpy as np
from collections import OrderedDict
from utool import util_latex
import functools
print, rrr, profile = ut.inject2(__name__, '[dbinfo]')


def print_qd_info(ibs, qaid_list, daid_list, verbose=False):
    """
    SeeAlso:
        ibs.print_annotconfig_stats(qaid_list, daid_list)

    information for a query/database aid configuration
    """
    bigstr = functools.partial(ut.truncate_str, maxlen=64, truncmsg=' ~TRUNC~ ')
    print('[qd_info] * dbname = %s' % ibs.get_dbname())
    print('[qd_info] * qaid_list = %s' % bigstr(str(qaid_list)))
    print('[qd_info] * daid_list = %s' % bigstr(str(daid_list)))
    print('[qd_info] * len(qaid_list) = %d' % len(qaid_list))
    print('[qd_info] * len(daid_list) = %d' % len(daid_list))
    print('[qd_info] * intersection = %r' % len(list(set(daid_list).intersection(set(qaid_list)))))
    if verbose:
        infokw = dict(with_contrib=False, with_agesex=False, with_header=False, verbose=False)
        d_info_str = get_dbinfo(ibs, aid_list=daid_list, tag='DataInfo', **infokw)['info_str2']
        q_info_str = get_dbinfo(ibs, aid_list=qaid_list, tag='QueryInfo', **infokw)['info_str2']
        print(q_info_str)
        print('\n')
        print(d_info_str)


def sight_resight_prob(N_range, nvisit1, nvisit2, resight):
    """
    https://en.wikipedia.org/wiki/Talk:Mark_and_recapture#Statistical_treatment
    http://stackoverflow.com/questions/31439875/infinite-summation-in-python/31442749
    """
    k, K, n  = resight, nvisit1, nvisit2
    from scipy.misc import comb
    N_range = np.array(N_range)

    def integers(start, blk_size=10000, pos=True, neg=False):
        x = np.arange(start, start + blk_size)
        while True:
            if pos:
                yield x
            if neg:
                yield -x - 1
            x += blk_size

    def converge_inf_sum(func, x_strm, eps=1e-5, axis=0):
        # Can still be very slow
        total = np.sum(func(x_strm.next()), axis=axis)
        #for x_blk in ut.ProgIter(x_strm, lbl='converging'):
        for x_blk in x_strm:
            diff = np.sum(func(x_blk), axis=axis)
            total += diff
            #error = abs(np.linalg.norm(diff))
            #print('error = %r' % (error,))
            if np.sqrt(diff.ravel().dot(diff.ravel())) <= eps:
                # Converged
                break
        return total

    numers = (comb(N_range - K, n - k) / comb(N_range, n))

    @ut.memoize
    def func(N_):
        return (comb(N_ - K, n - k) / comb(N_, n))

    denoms = []
    for N in ut.ProgIter(N_range, lbl='denoms'):
        x_strm = integers(start=(N + n - k), blk_size=100)
        denom = converge_inf_sum(func, x_strm, eps=1e-3)
        denoms.append(denom)

    #denom = sum([func(N_) for N_ in range(N_start, N_start * 2)])
    probs = numers / np.array(denoms)
    return probs


def sight_resight_count(nvisit1, nvisit2, resight):
    r"""
    Lincoln Petersen Index

    The Lincoln-Peterson index is a method used to estimate the total number of
    individuals in a population given two independent sets observations.  The
    likelihood of a population size is a hypergeometric distribution given by
    assuming a uniform sampling distribution.

    Args:
        nvisit1 (int): the number of individuals seen on visit 1.
        nvisit2 (int): be the number of individuals seen on visit 2.
        resight (int): the number of (matched) individuals seen on both visits.

    Returns:
        tuple: (pl_index, pl_error)

    LaTeX:
        \begin{equation}\label{eqn:lpifull}
            L(\poptotal \given \nvisit_1, \nvisit_2, \resight) =
            \frac{
                \binom{\nvisit_1}{\resight}
                \binom{\poptotal - \nvisit_1}{\nvisit_2 - \resight}
            }{
                \binom{\poptotal}{\nvisit_2}
            }
        \end{equation}
        Assuming that $T$ has a uniform prior distribution, the maximum
          likelihood estimation of population size given two visits to a
          location is:
        \begin{equation}\label{eqn:lpi}
            \poptotal \approx
            \frac{\nvisit_1 \nvisit_2}{\resight} \pm 1.96 \sqrt{\frac{{(\nvisit_1)}^2 (\nvisit_2) (\nvisit_2 - \resight)}{\resight^3}}
        \end{equation}

    References:
        https://en.wikipedia.org/wiki/Mark_and_recapture
        https://en.wikipedia.org/wiki/Talk:Mark_and_recapture#Statistical_treatment
        https://mail.google.com/mail/u/0/#search/lincoln+peterse+n/14c6b50227f5209f
        https://probabilityandstats.wordpress.com/tag/maximum-likelihood-estimate/
        http://math.arizona.edu/~jwatkins/o-mle.pdf

    CommandLine:
        python -m ibeis.other.dbinfo sight_resight_count --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> nvisit1 = 100
        >>> nvisit2 = 20
        >>> resight = 10
        >>> (pl_index, pl_error) = sight_resight_count(nvisit1, nvisit2, resight)
        >>> result = ('(pl_index, pl_error) = %s' % (ut.repr2((pl_index, pl_error)),))
        >>> pl_low = max(pl_index - pl_error, 1)
        >>> pl_high = pl_index + pl_error
        >>> print('pl_low = %r' % (pl_low,))
        >>> print('pl_high = %r' % (pl_high,))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> import scipy, scipy.stats
        >>> x = pl_index  # np.array([10, 11, 12])
        >>> k, N, K, n = resight, x, nvisit1, nvisit2
        >>> #k, M, n, N = k, N, k, n  # Wiki to SciPy notation
        >>> #prob = scipy.stats.hypergeom.cdf(k, N, K, n)
        >>> fig = pt.figure(1)
        >>> fig.clf()
        >>> N_range = np.arange(1, pl_high * 2)
        >>> # Something seems to be off
        >>> probs = sight_resight_prob(N_range, nvisit1, nvisit2, resight)
        >>> pl_prob = sight_resight_prob([pl_index], nvisit1, nvisit2, resight)[0]
        >>> pt.plot(N_range, probs, 'b-', label='probability of population size')
        >>> pt.plt.title('nvisit1=%r, nvisit2=%r, resight=%r' % (nvisit1, nvisit2, resight))
        >>> pt.plot(pl_index, pl_prob, 'rx', label='Lincoln Peterson Estimate')
        >>> pt.plot([pl_low, pl_high], [pl_prob, pl_prob], 'gx-', label='Lincoln Peterson Error Bar')
        >>> pt.legend()
        >>> ut.show_if_requested()

    """
    import math
    try:
        pl_index = int(math.ceil( (nvisit1 * nvisit2) / resight ))
        pl_error_num = float((nvisit1 ** 2) * nvisit2 * (nvisit2 - resight))
        pl_error_dom = float(resight ** 3)
        pl_error = int(math.ceil( 1.96 * math.sqrt(pl_error_num / pl_error_dom) ))
    except ZeroDivisionError:
        # pl_index = 'Undefined - Zero recaptured (k = 0)'
        pl_index = 0
        pl_error = 0
    return pl_index, pl_error


def estimate_ggr_count(ibs):
    """
    Example:
        >>> # DISABLE_DOCTEST GGR
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> dbdir = ut.truepath('~/lev/media/danger/GGR/GGR-IBEIS')
        >>> ibs = ibeis.opendb(dbdir='/home/joncrall/lev/media/danger/GGR/GGR-IBEIS')
    """
    import datetime
    day1 = datetime.date(2016, 1, 30)
    day2 = datetime.date(2016, 1, 31)

    filter_kw = {
        'multiple': False,
        'minqual': 'good',
        'is_known': True,
        'min_pername': 1,
        'view': ['right'],
    }
    print('\nOnly Singles:')
    filter_kw['multiple'] = False
    estimate_twoday_count(ibs, day1, day2, filter_kw)

    print('\nOnly Multis:')
    filter_kw['multiple'] = True
    estimate_twoday_count(ibs, day1, day2, filter_kw)

    print('\nIncluding Multiple:')
    filter_kw['multiple'] = None
    estimate_twoday_count(ibs, day1, day2, filter_kw)


def estimate_twoday_count(ibs, day1, day2, filter_kw):
    gid_list = ibs.get_valid_gids()
    images = ibs.get_image_lazydict(gid_list)
    datetimes = images['datetime']
    dates = [dt.date() for dt in datetimes]
    date2_gids = ut.group_items(gid_list, dates)
    date2_gids = ut.sort_dict(date2_gids)
    # date_hist = ut.map_dict_vals(len, date2_gids)
    # print('date_hist = %s' % (ut.repr2(date_hist, nl=2),))
    verbose = 0

    visit_dates = [day1, day2]
    visit_info_list_ = []
    for day in visit_dates:
        images = ibs.get_image_lazydict(date2_gids[day])

        aids = ut.flatten(images['aids'])
        aids = ibs.filter_annots_general(aids, filter_kw=filter_kw,
                                         verbose=verbose)
        nids = ibs.get_annot_name_rowids(aids)
        grouped_aids = ut.group_items(aids, nids)
        unique_nids = ut.unique(list(grouped_aids.keys()))

        if False:
            aids_list = ut.take(grouped_aids, unique_nids)
            for aids in aids_list:
                if len(aids) > 30:
                    break
            timedeltas_list = ibs.get_unflat_annots_timedelta_list(aids_list)
            # Do the five second rule
            marked_thresh = 5
            flags = []
            for nid, timedeltas in zip(unique_nids, timedeltas_list):
                flags.append(timedeltas.max() > marked_thresh)
            print('Unmarking %d names' % (len(flags) - sum(flags)))
            unique_nids = ut.compress(unique_nids, flags)
            grouped_aids = ut.dict_subset(grouped_aids, unique_nids)

        unique_aids = ut.flatten(list(grouped_aids.values()))
        info = {
            'unique_nids': unique_nids,
            'grouped_aids': grouped_aids,
            'unique_aids': unique_aids,
        }
        visit_info_list_.append(info)

    # Estimate statistics
    from ibeis.other import dbinfo
    aids_day1, aids_day2 = ut.take_column(visit_info_list_, 'unique_aids')
    nids_day1, nids_day2 = ut.take_column(visit_info_list_, 'unique_nids')
    resight_nids = ut.isect(nids_day1, nids_day2)
    nsight1 = len(nids_day1)
    nsight2 = len(nids_day2)
    resight = len(resight_nids)
    lp_index, lp_error = dbinfo.sight_resight_count(nsight1, nsight2, resight)

    if False:
        from ibeis.other import dbinfo
        print('DAY 1 STATS:')
        _ = dbinfo.get_dbinfo(ibs, aid_list=aids_day1)  # NOQA
        print('DAY 2 STATS:')
        _ = dbinfo.get_dbinfo(ibs, aid_list=aids_day2)  # NOQA
        print('COMBINED STATS:')
        _ = dbinfo.get_dbinfo(ibs, aid_list=aids_day1 + aids_day2)  # NOQA

    print('%d annots on day 1' % (len(aids_day1)) )
    print('%d annots on day 2' % (len(aids_day2)) )
    print('%d names on day 1' % (nsight1,))
    print('%d names on day 2' % (nsight2,))
    print('resight = %r' % (resight,))
    print('lp_index = %r ± %r' % (lp_index, lp_error))


def draw_twoday_count(ibs, visit_info_list_):
    import copy
    visit_info_list = copy.deepcopy(visit_info_list_)

    aids_day1, aids_day2 = ut.take_column(visit_info_list_, 'aids')
    nids_day1, nids_day2 = ut.take_column(visit_info_list_, 'unique_nids')
    resight_nids = ut.isect(nids_day1, nids_day2)

    if False:
        # HACK REMOVE DATA TO MAKE THIS FASTER
        num = 20
        for info in visit_info_list:
            non_resight_nids = list(set(info['unique_nids']) - set(resight_nids))
            sample_nids2 = non_resight_nids[0:num] + resight_nids[:num]
            info['grouped_aids'] = ut.dict_subset(info['grouped_aids'], sample_nids2)
            info['unique_nids'] = sample_nids2

    # Build a graph of matches
    if False:

        debug = False

        for info in visit_info_list:
            edges = []
            grouped_aids = info['grouped_aids']

            aids_list = list(grouped_aids.values())
            ams_list = ibs.get_annotmatch_rowids_in_cliques(aids_list)
            aids1_list = ibs.unflat_map(ibs.get_annotmatch_aid1, ams_list)
            aids2_list = ibs.unflat_map(ibs.get_annotmatch_aid2, ams_list)
            for ams, aids, aids1, aids2 in zip(ams_list, aids_list, aids1_list, aids2_list):
                edge_nodes = set(aids1 + aids2)
                ##if len(edge_nodes) != len(set(aids)):
                #    #print('--')
                #    #print('aids = %r' % (aids,))
                #    #print('edge_nodes = %r' % (edge_nodes,))
                bad_aids = edge_nodes - set(aids)
                if len(bad_aids) > 0:
                    print('bad_aids = %r' % (bad_aids,))
                unlinked_aids = set(aids) - edge_nodes
                mst_links = list(ut.itertwo(list(unlinked_aids) + list(edge_nodes)[:1]))
                bad_aids.add(None)
                user_links = [(u, v) for (u, v) in zip(aids1, aids2) if u not in bad_aids and v not in bad_aids]
                new_edges = mst_links + user_links
                new_edges = [(int(u), int(v)) for u, v in new_edges if u not in bad_aids and v not in bad_aids]
                edges += new_edges
            info['edges'] = edges

        # Add edges between days
        grouped_aids1, grouped_aids2 = ut.take_column(visit_info_list, 'grouped_aids')
        nids_day1, nids_day2 = ut.take_column(visit_info_list, 'unique_nids')
        resight_nids = ut.isect(nids_day1, nids_day2)

        resight_aids1 = ut.take(grouped_aids1, resight_nids)
        resight_aids2 = ut.take(grouped_aids2, resight_nids)
        #resight_aids3 = [list(aids1) + list(aids2) for aids1, aids2 in zip(resight_aids1, resight_aids2)]

        ams_list = ibs.get_annotmatch_rowids_between_groups(resight_aids1, resight_aids2)
        aids1_list = ibs.unflat_map(ibs.get_annotmatch_aid1, ams_list)
        aids2_list = ibs.unflat_map(ibs.get_annotmatch_aid2, ams_list)

        between_edges = []
        for ams, aids1, aids2, rawaids1, rawaids2 in zip(ams_list, aids1_list, aids2_list, resight_aids1, resight_aids2):
            link_aids = aids1 + aids2
            rawaids3 = rawaids1 + rawaids2
            badaids = ut.setdiff(link_aids, rawaids3)
            assert not badaids
            user_links = [(int(u), int(v)) for (u, v) in zip(aids1, aids2)
                          if u is not None and v is not None]
            # HACK THIS OFF
            user_links = []
            if len(user_links) == 0:
                # Hack in an edge
                between_edges += [(rawaids1[0], rawaids2[0])]
            else:
                between_edges += user_links

        assert np.all(0 == np.diff(np.array(ibs.unflat_map(ibs.get_annot_nids, between_edges)), axis=1))

        import plottool as pt
        import networkx as nx
        #pt.qt4ensure()
        #len(list(nx.connected_components(graph1)))
        #print(ut.graph_info(graph1))

        # Layout graph
        layoutkw = dict(
            prog='neato',
            draw_implicit=False, splines='line',
            #splines='curved',
            #splines='spline',
            #sep=10 / 72,
            #prog='dot', rankdir='TB',
        )

        def translate_graph_to_origin(graph):
            x, y, w, h = ut.get_graph_bounding_box(graph)
            ut.translate_graph(graph, (-x, -y))

        def stack_graphs(graph_list, vert=False, pad=None):
            graph_list_ = [g.copy() for g in graph_list]
            for g in graph_list_:
                translate_graph_to_origin(g)
            bbox_list = [ut.get_graph_bounding_box(g) for g in graph_list_]
            if vert:
                dim1 = 3
                dim2 = 2
            else:
                dim1 = 2
                dim2 = 3
            dim1_list = np.array([bbox[dim1] for bbox in bbox_list])
            dim2_list = np.array([bbox[dim2] for bbox in bbox_list])
            if pad is None:
                pad = np.mean(dim1_list) / 2
            offset1_list = ut.cumsum([0] + [d + pad for d in dim1_list[:-1]])
            max_dim2 = max(dim2_list)
            offset2_list = [(max_dim2 - d2) / 2 for d2 in dim2_list]
            if vert:
                t_xy_list = [(d2, d1) for d1, d2 in zip(offset1_list, offset2_list)]
            else:
                t_xy_list = [(d1, d2) for d1, d2 in zip(offset1_list, offset2_list)]

            for g, t_xy in zip(graph_list_, t_xy_list):
                ut.translate_graph(g, t_xy)
                ut.pin_nodes(g)

            new_graph = nx.compose_all(graph_list_)
            #pt.show_nx(new_graph, layout='custom', node_labels=False, as_directed=False)  # NOQA
            return new_graph

        # Construct graph
        for count, info in enumerate(visit_info_list):
            graph = nx.Graph()
            edges = [(int(u), int(v)) for u, v in info['edges']
                     if u is not None and v is not None]
            graph.add_edges_from(edges, attr_dict={'zorder': 10})
            nx.set_node_attributes(graph, 'zorder', 20)

            # Layout in neato
            _ = pt.nx_agraph_layout(graph, inplace=True, **layoutkw)  # NOQA

            # Extract compoments and then flatten in nid ordering
            ccs = list(nx.connected_components(graph))
            root_aids = []
            cc_graphs = []
            for cc_nodes in ccs:
                cc = graph.subgraph(cc_nodes)
                try:
                    root_aids.append(list(ut.nx_source_nodes(cc.to_directed()))[0])
                except nx.NetworkXUnfeasible:
                    root_aids.append(list(cc.nodes())[0])
                cc_graphs.append(cc)

            root_nids = ibs.get_annot_nids(root_aids)
            nid2_graph = dict(zip(root_nids, cc_graphs))

            resight_nids_ = set(resight_nids).intersection(set(root_nids))
            noresight_nids_ = set(root_nids) - resight_nids_

            n_graph_list = ut.take(nid2_graph, sorted(noresight_nids_))
            r_graph_list = ut.take(nid2_graph, sorted(resight_nids_))

            if len(n_graph_list) > 0:
                n_graph = nx.compose_all(n_graph_list)
                _ = pt.nx_agraph_layout(n_graph, inplace=True, **layoutkw)  # NOQA
                n_graphs = [n_graph]
            else:
                n_graphs = []

            r_graphs = [stack_graphs(chunk) for chunk in ut.ichunks(r_graph_list, 100)]
            if count == 0:
                new_graph = stack_graphs(n_graphs + r_graphs, vert=True)
            else:
                new_graph = stack_graphs(r_graphs[::-1] + n_graphs, vert=True)

            #pt.show_nx(new_graph, layout='custom', node_labels=False, as_directed=False)  # NOQA
            info['graph'] = new_graph

        graph1_, graph2_ = ut.take_column(visit_info_list, 'graph')
        if False:
            _ = pt.show_nx(graph1_, layout='custom', node_labels=False, as_directed=False)  # NOQA
            _ = pt.show_nx(graph2_, layout='custom', node_labels=False, as_directed=False)  # NOQA

        graph_list = [graph1_, graph2_]
        twoday_graph = stack_graphs(graph_list, vert=True, pad=None)
        ut.pin_nodes(twoday_graph)

        if debug:
            ut.nx_delete_None_edge_attr(twoday_graph)
            ut.nx_delete_None_node_attr(twoday_graph)
            print('twoday_graph(pre) info' + ut.repr3(ut.graph_info(twoday_graph), nl=2))

        # Hack, no idea why there are nodes that dont exist here
        between_edges_ = [edge for edge in between_edges
                          if twoday_graph.has_node(edge[0]) and twoday_graph.has_node(edge[1])]

        twoday_graph.add_edges_from(between_edges_, attr_dict={'alpha': .2, 'zorder': 0})
        ut.nx_ensure_agraph_color(twoday_graph)

        layoutkw['splines'] = 'line'
        layoutkw['prog'] = 'neato'
        agraph = pt.nx_agraph_layout(twoday_graph, inplace=True, return_agraph=True, **layoutkw)[-1]  # NOQA
        if False:
            fpath = ut.truepath('~/ggr_graph.png')
            agraph.draw(fpath)
            ut.startfile(fpath)

        if debug:
            print('twoday_graph(post) info' + ut.repr3(ut.graph_info(twoday_graph)))

        _ = pt.show_nx(twoday_graph, layout='custom', node_labels=False, as_directed=False)  # NOQA


def get_dbinfo(ibs, verbose=True,
               with_imgsize=False,
               with_bytes=False,
               with_contrib=False,
               with_agesex=False,
               with_header=True,
               short=False,
               tag='dbinfo',
               aid_list=None):
    """

    Returns dictionary of digestable database information
    Infostr is a string summary of all the stats. Prints infostr in addition to
    returning locals

    Args:
        ibs (IBEISController):
        verbose (bool):
        with_imgsize (bool):
        with_bytes (bool):

    Returns:
        dict:

    CommandLine:
        python -m ibeis.other.dbinfo --exec-get_dbinfo:0
        python -m ibeis.other.dbinfo --test-get_dbinfo:1
        python -m ibeis.other.dbinfo --test-get_dbinfo:0 --db NNP_Master3
        python -m ibeis.other.dbinfo --test-get_dbinfo:0 --db PZ_Master1
        python -m ibeis.other.dbinfo --test-get_dbinfo:0 --db GZ_ALL
        python -m ibeis.other.dbinfo --exec-get_dbinfo:0 --db PZ_ViewPoints
        python -m ibeis.other.dbinfo --exec-get_dbinfo:0 --db GZ_Master1

        python -m ibeis.other.dbinfo --exec-get_dbinfo:0 -a ctrl
        python -m ibeis.other.dbinfo --exec-get_dbinfo:0 -a default:minqual=ok,require_timestamp=True --dbdir ~/lev/media/danger/LEWA
        python -m ibeis.other.dbinfo --exec-get_dbinfo:0 -a default:minqual=ok,require_timestamp=True --dbdir ~/lev/media/danger/LEWA --loadbackup=0

        python -m ibeis.other.dbinfo --exec-get_dbinfo:0 -a default: --dbdir ~/lev/media/danger/LEWA
        python -m ibeis.other.dbinfo --exec-get_dbinfo:0 -a default: --dbdir ~/lev/media/danger/LEWA --loadbackup=0

    Example1:
        >>> # SCRIPT
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> defaultdb = 'testdb1'
        >>> ibs, aid_list = ibeis.testdata_aids(defaultdb, a='default:minqual=ok,view=primary,view_ext1=1')
        >>> kwargs = ut.get_kwdefaults(get_dbinfo)
        >>> kwargs['verbose'] = False
        >>> kwargs['aid_list'] = aid_list
        >>> kwargs = ut.parse_dict_from_argv(kwargs)
        >>> output = get_dbinfo(ibs, **kwargs)
        >>> result = (output['info_str'])
        >>> print(result)
        >>> #ibs = ibeis.opendb(defaultdb='testdb1')
        >>> # <HACK FOR FILTERING>
        >>> #from ibeis.expt import cfghelpers
        >>> #from ibeis.expt import annotation_configs
        >>> #from ibeis.init import filter_annots
        >>> #named_defaults_dict = ut.dict_take(annotation_configs.__dict__,
        >>> #                                   annotation_configs.TEST_NAMES)
        >>> #named_qcfg_defaults = dict(zip(annotation_configs.TEST_NAMES,
        >>> #                               ut.get_list_column(named_defaults_dict, 'qcfg')))
        >>> #acfg = cfghelpers.parse_argv_cfg(('--annot-filter', '-a'), named_defaults_dict=named_qcfg_defaults, default=None)[0]
        >>> #aid_list = ibs.get_valid_aids()
        >>> # </HACK FOR FILTERING>

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> verbose = True
        >>> short = True
        >>> #ibs = ibeis.opendb(db='GZ_ALL')
        >>> #ibs = ibeis.opendb(db='PZ_Master0')
        >>> ibs = ibeis.opendb('testdb1')
        >>> assert ibs.get_dbname() == 'testdb1', 'DO NOT DELETE CONTRIBUTORS OF OTHER DBS'
        >>> ibs.delete_contributors(ibs.get_valid_contrib_rowids())
        >>> ibs.delete_empty_nids()
        >>> #ibs = ibeis.opendb(db='PZ_MTEST')
        >>> output = get_dbinfo(ibs, with_contrib=False, verbose=False, short=True)
        >>> result = (output['info_str'])
        >>> print(result)
        +============================
        DB Info:  testdb1
        DB Notes: None
        DB NumContrib: 0
        ----------
        # Names                      = 7
        # Names (unassociated)       = 0
        # Names (singleton)          = 5
        # Names (multiton)           = 2
        ----------
        # Annots                     = 13
        # Annots (unknown)           = 4
        # Annots (singleton)         = 5
        # Annots (multiton)          = 4
        ----------
        # Img                        = 13
        L============================
    """
    # TODO Database size in bytes
    # TODO: occurrence, contributors, etc...

    # Basic variables
    request_annot_subset = False
    _input_aid_list = aid_list  # NOQA
    if aid_list is None:
        valid_aids = ibs.get_valid_aids()
        valid_nids = ibs.get_valid_nids()
        valid_gids = ibs.get_valid_gids()
    else:
        if isinstance(aid_list, str):
            # Hack to get experiment stats on aids
            acfg_name_list = [aid_list]
            print('Specified custom aids via acfgname %s' % (acfg_name_list,))
            from ibeis.expt import experiment_helpers
            acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(
                ibs, acfg_name_list)
            aid_list = sorted(list(set(ut.flatten(ut.flatten(expanded_aids_list)))))
            #aid_list =
        if verbose:
            print('Specified %d custom aids' % (len(aid_list,)))
        request_annot_subset = True
        valid_aids = aid_list
        valid_nids = list(
            set(ibs.get_annot_nids(aid_list, distinguish_unknowns=False)) -
            {const.UNKNOWN_NAME_ROWID}
        )
        valid_gids = list(set(ibs.get_annot_gids(aid_list)))
    #associated_nids = ibs.get_valid_nids(filter_empty=True)  # nids with at least one annotation

    # Image info
    if verbose:
        print('Checking Image Info')
    gx2_aids = ibs.get_image_aids(valid_gids)
    if request_annot_subset:
        # remove annots not in this subset
        valid_aids_set = set(valid_aids)
        gx2_aids = [list(set(aids).intersection(valid_aids_set)) for aids in gx2_aids]

    gx2_nAnnots = np.array(list(map(len, gx2_aids)))
    image_without_annots = len(np.where(gx2_nAnnots == 0)[0])
    gx2_nAnnots_stats  = ut.get_stats_str(gx2_nAnnots, newlines=True, use_median=True)
    image_reviewed_list = ibs.get_image_reviewed(valid_gids)

    # Name stats
    if verbose:
        print('Checking Name Info')
    nx2_aids = ibs.get_name_aids(valid_nids)
    if request_annot_subset:
        # remove annots not in this subset
        valid_aids_set = set(valid_aids)
        nx2_aids = [list(set(aids).intersection(valid_aids_set)) for aids in nx2_aids]
    associated_nids = ut.compress(valid_nids, list(map(len, nx2_aids)))

    ibs.check_name_mapping_consistency(nx2_aids)

    # Occurrence Info
    def compute_annot_occurrence_ids(ibs, aid_list):
        from ibeis.algo.preproc import preproc_occurrence
        gid_list = ibs.get_annot_gids(aid_list)
        gid2_aids = ut.group_items(aid_list, gid_list)
        flat_imgsetids, flat_gids = preproc_occurrence.ibeis_compute_occurrences(ibs, gid_list, seconds_thresh=4 * 60 * 60, verbose=False)
        occurid2_gids = ut.group_items(flat_gids, flat_imgsetids)
        occurid2_aids = {oid: ut.flatten(ut.take(gid2_aids, gids)) for oid, gids in occurid2_gids.items()}
        return occurid2_aids

    import utool
    with utool.embed_on_exception_context:
        occurid2_aids = compute_annot_occurrence_ids(ibs, valid_aids)
        occur_nids = ibs.unflat_map(ibs.get_annot_nids, occurid2_aids.values())
        occur_unique_nids = [ut.unique(nids) for nids in occur_nids]
        nid2_occurxs = ut.ddict(list)
        for occurx, nids in enumerate(occur_unique_nids):
            for nid in nids:
                nid2_occurxs[nid].append(occurx)

    nid2_occurx_single = {nid: occurxs for nid, occurxs in nid2_occurxs.items() if len(occurxs) <= 1}
    nid2_occurx_resight = {nid: occurxs for nid, occurxs in nid2_occurxs.items() if len(occurxs) > 1}
    singlesight_encounters = ibs.get_name_aids(nid2_occurx_single.keys())

    singlesight_annot_stats = ut.get_stats(list(map(len, singlesight_encounters)), use_median=True, use_sum=True)
    resight_name_stats = ut.get_stats(list(map(len, nid2_occurx_resight.values())), use_median=True, use_sum=True)

    # Encounter Info
    def break_annots_into_encounters(aids):
        from ibeis.algo.preproc import occurrence_blackbox
        import datetime
        thresh_sec = datetime.timedelta(minutes=30).seconds
        posixtimes = np.array(ibs.get_annot_image_unixtimes_asfloat(aids))
        #latlons = ibs.get_annot_image_gps(aids)
        labels = occurrence_blackbox.cluster_timespace2(posixtimes, None, thresh_sec=thresh_sec)
        return labels
        #ave_enc_time = [np.mean(times) for lbl, times in ut.group_items(posixtimes, labels).items()]
        #ut.square_pdist(ave_enc_time)

    try:
        am_rowids = ibs.get_annotmatch_rowids_between_groups([valid_aids], [valid_aids])[0]
        aid_pairs = ibs.filter_aidpairs_by_tags(min_num=0, am_rowids=am_rowids)
        undirected_tags = ibs.get_aidpair_tags(aid_pairs.T[0], aid_pairs.T[1], directed=False)
        tagged_pairs = list(zip(aid_pairs.tolist(), undirected_tags))
        tag_dict = ut.groupby_tags(tagged_pairs, undirected_tags)
        pair_tag_info = ut.map_dict_vals(len, tag_dict)

        reviewed_type_hist = ut.dict_hist(ibs.get_annot_pair_is_reviewed(aid_pairs.T[0], aid_pairs.T[1]))

        pair_tag_info['reviewed_type_hist'] = reviewed_type_hist
    except Exception:
        pair_tag_info = {}

    #print(ut.dict_str(pair_tag_info))

    # Annot Stats
    # TODO: number of images where chips cover entire image
    # TODO: total image coverage of annotation
    # TODO: total annotation overlap
    """
    ax2_unknown = ibs.is_aid_unknown(valid_aids)
    ax2_nid = ibs.get_annot_name_rowids(valid_aids)
    assert all([nid < 0 if unknown else nid > 0 for nid, unknown in
                zip(ax2_nid, ax2_unknown)]), 'bad annot nid'
    """
    #
    if verbose:
        print('Checking Annot Species')
    unknown_aids = ut.compress(valid_aids, ibs.is_aid_unknown(valid_aids))
    species_list = ibs.get_annot_species_texts(valid_aids)
    species2_aids = ut.group_items(valid_aids, species_list)
    species2_nAids = {key: len(val) for key, val in species2_aids.items()}

    if verbose:
        print('Checking Multiton/Singleton Species')
    nx2_nAnnots = np.array(list(map(len, nx2_aids)))
    # Seperate singleton / multitons
    multiton_nxs  = np.where(nx2_nAnnots > 1)[0]
    singleton_nxs = np.where(nx2_nAnnots == 1)[0]
    unassociated_nxs = np.where(nx2_nAnnots == 0)[0]
    assert len(np.intersect1d(singleton_nxs, multiton_nxs)) == 0, 'intersecting names'
    valid_nxs      = np.hstack([multiton_nxs, singleton_nxs])
    num_names_with_gt = len(multiton_nxs)

    # Annot Info
    if verbose:
        print('Checking Annot Info')
    multiton_aids_list = ut.take(nx2_aids, multiton_nxs)
    assert len(set(multiton_nxs)) == len(multiton_nxs)
    if len(multiton_aids_list) == 0:
        multiton_aids = np.array([], dtype=np.int)
    else:
        multiton_aids = np.hstack(multiton_aids_list)
        assert len(set(multiton_aids)) == len(multiton_aids), 'duplicate annot'
    singleton_aids = ut.take(nx2_aids, singleton_nxs)
    multiton_nid2_nannots = list(map(len, multiton_aids_list))

    # Image size stats
    if with_imgsize:
        if verbose:
            print('Checking ImageSize Info')
        gpath_list = ibs.get_image_paths(valid_gids)
        def wh_print_stats(wh_list):
            if len(wh_list) == 0:
                return '{empty}'
            wh_list = np.asarray(wh_list)
            stat_dict = OrderedDict(
                [( 'max', wh_list.max(0)),
                 ( 'min', wh_list.min(0)),
                 ('mean', wh_list.mean(0)),
                 ( 'std', wh_list.std(0))])
            def arr2str(var):
                return ('[' + (
                    ', '.join(list(map(lambda x: '%.1f' % x, var)))
                ) + ']')
            ret = (',\n    '.join([
                '%s:%s' % (key, arr2str(val))
                for key, val in stat_dict.items()
            ]))
            return '{\n    ' + ret + '\n}'

        print('reading image sizes')
        # Image size stats
        img_size_list  = ibs.get_image_sizes(valid_gids)
        img_size_stats  = wh_print_stats(img_size_list)

        # Chip size stats
        annotation_bbox_list = ibs.get_annot_bboxes(valid_aids)
        annotation_bbox_arr = np.array(annotation_bbox_list)
        if len(annotation_bbox_arr) == 0:
            annotation_size_list = []
        else:
            annotation_size_list = annotation_bbox_arr[:, 2:4]
        chip_size_stats = wh_print_stats(annotation_size_list)
        imgsize_stat_lines = [
            (' # Img in dir                 = %d' % len(gpath_list)),
            (' Image Size Stats  = %s' % (img_size_stats,)),
            (' * Chip Size Stats = %s' % (chip_size_stats,)),
        ]
    else:
        imgsize_stat_lines = []

    if verbose:
        print('Building Stats String')

    multiton_stats = ut.get_stats_str(multiton_nid2_nannots, newlines=True, use_median=True)

    # Time stats
    unixtime_list = ibs.get_image_unixtime(valid_gids)
    unixtime_list = ut.list_replace(unixtime_list, -1, float('nan'))
    #valid_unixtime_list = [time for time in unixtime_list if time != -1]
    #unixtime_statstr = ibs.get_image_time_statstr(valid_gids)
    if ut.get_argflag('--hackshow-unixtime'):
        show_time_distributions(ibs, unixtime_list)
        ut.show_if_requested()
    unixtime_statstr = ut.get_timestats_str(unixtime_list, newlines=True, full=True)

    # GPS stats
    gps_list_ = ibs.get_image_gps(valid_gids)
    gpsvalid_list = [gps != (-1, -1) for gps in gps_list_]
    gps_list  = ut.compress(gps_list_, gpsvalid_list)

    def get_annot_age_stats(aid_list):
        annot_age_months_est_min = ibs.get_annot_age_months_est_min(aid_list)
        annot_age_months_est_max = ibs.get_annot_age_months_est_max(aid_list)
        age_dict = ut.ddict((lambda : 0))
        for min_age, max_age in zip(annot_age_months_est_min, annot_age_months_est_max):
            if (min_age is None or min_age < 12) and max_age < 12:
                age_dict['Infant'] += 1
            elif 12 <= min_age and min_age < 36 and 12 <= max_age and max_age < 36:
                age_dict['Juvenile'] += 1
            elif 36 <= min_age and (36 <= max_age or max_age is None):
                age_dict['Adult'] += 1
            else:
                print('Found UNKNOWN Age: %r, %r' % (min_age, max_age, ))
                age_dict['UNKNOWN'] += 1
        return age_dict

    def get_annot_sex_stats(aid_list):
        annot_sextext_list = ibs.get_annot_sex_texts(aid_list)
        sextext2_aids = ut.group_items(aid_list, annot_sextext_list)
        sex_keys = list(ibs.const.SEX_TEXT_TO_INT.keys())
        assert set(sex_keys) >= set(annot_sextext_list), 'bad keys: ' + str(set(annot_sextext_list) - set(sex_keys))
        sextext2_nAnnots = ut.odict([(key, len(sextext2_aids.get(key, []))) for key in sex_keys])
        # Filter 0's
        sextext2_nAnnots = {key: val for key, val in six.iteritems(sextext2_nAnnots) if val != 0}
        return sextext2_nAnnots

    if verbose:
        print('Checking Other Annot Stats')

    qualtext2_nAnnots = ibs.get_annot_qual_stats(valid_aids)
    yawtext2_nAnnots = ibs.get_annot_yaw_stats(valid_aids)
    agetext2_nAnnots = get_annot_age_stats(valid_aids)
    sextext2_nAnnots = get_annot_sex_stats(valid_aids)

    if verbose:
        print('Checking Contrib Stats')

    # Contributor Statistics
    # hack remove colon for image alignment
    def fix_tag_list(tag_list):
        return [None if tag is None else tag.replace(':', ';') for tag in tag_list]
    image_contrib_tags = fix_tag_list(ibs.get_image_contributor_tag(valid_gids))
    annot_contrib_tags = fix_tag_list(ibs.get_annot_image_contributor_tag(valid_aids))
    contrib_tag_to_gids = ut.group_items(valid_gids, image_contrib_tags)
    contrib_tag_to_aids = ut.group_items(valid_aids, annot_contrib_tags)

    contrib_tag_to_qualstats = {key: ibs.get_annot_qual_stats(aids) for key, aids in six.iteritems(contrib_tag_to_aids)}
    contrib_tag_to_viewstats = {key: ibs.get_annot_yaw_stats(aids) for key, aids in six.iteritems(contrib_tag_to_aids)}

    contrib_tag_to_nImages = {key: len(val) for key, val in six.iteritems(contrib_tag_to_gids)}
    contrib_tag_to_nAnnots = {key: len(val) for key, val in six.iteritems(contrib_tag_to_aids)}

    if verbose:
        print('Summarizing')

    # Summarize stats
    num_names = len(valid_nids)
    num_names_unassociated = len(valid_nids) - len(associated_nids)
    num_names_singleton = len(singleton_nxs)
    num_names_multiton =  len(multiton_nxs)

    num_singleton_annots = len(singleton_aids)
    num_multiton_annots = len(multiton_aids)
    num_unknown_annots = len(unknown_aids)
    num_annots = len(valid_aids)

    if with_bytes:
        if verbose:
            print('Checking Disk Space')
        ibsdir_space   = ut.byte_str2(ut.get_disk_space(ibs.get_ibsdir()))
        dbdir_space    = ut.byte_str2(ut.get_disk_space(ibs.get_dbdir()))
        imgdir_space   = ut.byte_str2(ut.get_disk_space(ibs.get_imgdir()))
        cachedir_space = ut.byte_str2(ut.get_disk_space(ibs.get_cachedir()))

    if True:
        if verbose:
            print('Check asserts')
        try:
            bad_aids = np.intersect1d(multiton_aids, unknown_aids)
            _num_names_total_check = num_names_singleton + num_names_unassociated + num_names_multiton
            _num_annots_total_check = num_unknown_annots + num_singleton_annots + num_multiton_annots
            assert len(bad_aids) == 0, 'intersecting multiton aids and unknown aids'
            assert _num_names_total_check == num_names, 'inconsistent num names'
            #if not request_annot_subset:
            # dont check this if you have an annot subset
            assert _num_annots_total_check == num_annots, 'inconsistent num annots'
        except Exception as ex:
            ut.printex(ex, keys=[
                '_num_names_total_check',
                'num_names',
                '_num_annots_total_check',
                'num_annots',
                'num_names_singleton',
                'num_names_multiton',
                'num_unknown_annots',
                'num_multiton_annots',
                'num_singleton_annots',
            ])
            raise

    # Get contributor statistics
    contrib_rowids = ibs.get_valid_contrib_rowids()
    num_contributors = len(contrib_rowids)

    # print
    num_tabs = 5

    def align2(str_):
        return ut.align(str_, ':', ' :')

    def align_dict2(dict_):
        str_ = ut.dict_str(dict_)
        return align2(str_)

    header_block_lines = (
        [('+============================'), ] + (
            [
                ('+ singleton := single sighting'),
                ('+ multiton  := multiple sightings'),
                ('--' * num_tabs),
            ] if not short and with_header else []
        )
    )

    source_block_lines = [
        ('DB Info:  ' + ibs.get_dbname()),
        ('DB Notes: ' + ibs.get_dbnotes()),
        ('DB NumContrib: %d' % num_contributors),
    ]

    bytes_block_lines = [
        ('--' * num_tabs),
        ('DB Bytes: '),
        ('     +- dbdir nBytes:         ' + dbdir_space),
        ('     |  +- _ibsdb nBytes:     ' + ibsdir_space),
        ('     |  |  +-imgdir nBytes:   ' + imgdir_space),
        ('     |  |  +-cachedir nBytes: ' + cachedir_space),
    ] if with_bytes else []

    name_block_lines = [
        ('--' * num_tabs),
        ('# Names                      = %d' % num_names),
        ('# Names (unassociated)       = %d' % num_names_unassociated),
        ('# Names (singleton)          = %d' % num_names_singleton),
        ('# Names (multiton)           = %d' % num_names_multiton),
    ]

    subset_str = '        ' if not request_annot_subset else '(SUBSET)'

    annot_block_lines = [
        ('--' * num_tabs),
        ('# Annots %s            = %d' % (subset_str, num_annots,)),
        ('# Annots (unknown)           = %d' % num_unknown_annots),
        ('# Annots (singleton)         = %d' % num_singleton_annots),
        ('# Annots (multiton)          = %d' % num_multiton_annots),
    ]

    annot_per_basic_block_lines = [
        ('--' * num_tabs),
        ('# Annots per Name (multiton) = %s' % (align2(multiton_stats),)),
        ('# Annots per Image           = %s' % (align2(gx2_nAnnots_stats),)),
        ('# Annots per Species         = %s' % (align_dict2(species2_nAids),)),
    ] if not short else []

    occurrence_block_lines = [
        ('--' * num_tabs),
        ('# Occurrence Per Name (Resights) = %s' % (align_dict2(resight_name_stats),)),
        ('# Annots per Encounter (Singlesights) = %s' % (align_dict2(singlesight_annot_stats),)),
        ('# Pair Tag Info (annots) = %s' % (align_dict2(pair_tag_info),)),
    ] if not short else []

    annot_per_qualview_block_lines = [
        None if short else '# Annots per Viewpoint = %s' % align_dict2(yawtext2_nAnnots),
        None if short else '# Annots per Quality = %s' % align_dict2(qualtext2_nAnnots),
    ]

    annot_per_agesex_block_lines = [
        '# Annots per Age = %s' % align_dict2(agetext2_nAnnots),
        '# Annots per Sex = %s' % align_dict2(sextext2_nAnnots),
    ] if not short  and with_agesex else []

    contrib_block_lines = [
        '# Images per contributor       = ' + align_dict2(contrib_tag_to_nImages),
        '# Annots per contributor       = ' + align_dict2(contrib_tag_to_nAnnots),
        '# Quality per contributor      = ' + ut.dict_str(contrib_tag_to_qualstats, sorted_=True),
        '# Viewpoint per contributor    = ' + ut.dict_str(contrib_tag_to_viewstats, sorted_=True),
    ] if with_contrib else []

    img_block_lines = [
        ('--' * num_tabs),
        ('# Img                        = %d' % len(valid_gids)),
        None if short else ('# Img reviewed               = %d' % sum(image_reviewed_list)),
        None if short else ('# Img with gps               = %d' % len(gps_list)),
        #('# Img with timestamp         = %d' % len(valid_unixtime_list)),
        None if short else ('Img Time Stats               = %s' % (align2(unixtime_statstr),)),
    ]

    info_str_lines = (
        header_block_lines +
        bytes_block_lines +
        source_block_lines +
        name_block_lines +
        annot_block_lines +
        annot_per_basic_block_lines +
        occurrence_block_lines +
        annot_per_qualview_block_lines +
        annot_per_agesex_block_lines +
        img_block_lines +
        contrib_block_lines +
        imgsize_stat_lines +
        [('L============================'), ]
    )
    info_str = '\n'.join(ut.filter_Nones(info_str_lines))
    info_str2 = ut.indent(info_str, '[{tag}]'.format(tag=tag))
    if verbose:
        print(info_str2)
    locals_ = locals()
    return locals_


def hackshow_names(ibs, aid_list, fnum=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):

    CommandLine:
        python -m ibeis.other.dbinfo --exec-hackshow_names --show
        python -m ibeis.other.dbinfo --exec-hackshow_names --show --db PZ_Master1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = hackshow_names(ibs, aid_list)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    import plottool as pt
    import vtool as vt
    grouped_aids, nid_list = ibs.group_annots_by_name(aid_list)
    grouped_aids = [aids for aids in grouped_aids if len(aids) > 1]
    unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, grouped_aids)
    yaws_list = ibs.unflat_map(ibs.get_annot_yaws, grouped_aids)
    #markers_list = [[(1, 2, yaw * 360 / (np.pi * 2)) for yaw in yaws] for yaws in yaws_list]

    unixtime_list = ut.flatten(unixtimes_list)
    timemax = np.nanmax(unixtime_list)
    timemin = np.nanmin(unixtime_list)
    timerange = timemax - timemin
    unixtimes_list = [((unixtimes[:] - timemin) / timerange) for unixtimes in unixtimes_list]
    for unixtimes in unixtimes_list:
        num_nan = sum(np.isnan(unixtimes))
        unixtimes[np.isnan(unixtimes)] = np.linspace(-1, -.5, num_nan)
    #ydata_list = [np.arange(len(aids)) for aids in grouped_aids]
    sortx_list = vt.argsort_groups(unixtimes_list, reverse=False)
    #markers_list = ut.list_ziptake(markers_list, sortx_list)
    yaws_list = ut.list_ziptake(yaws_list, sortx_list)
    ydatas_list = vt.ziptake(unixtimes_list, sortx_list)
    #ydatas_list = sortx_list
    #ydatas_list = vt.argsort_groups(unixtimes_list, reverse=False)

    # Sort by num members
    #ydatas_list = ut.take(ydatas_list, np.argsort(list(map(len, ydatas_list))))
    xdatas_list = [np.zeros(len(ydatas)) + count for count, ydatas in enumerate(ydatas_list)]
    #markers = ut.flatten(markers_list)
    #yaws = np.array(ut.flatten(yaws_list))
    y_data = np.array(ut.flatten(ydatas_list))
    x_data = np.array(ut.flatten(xdatas_list))
    fnum = pt.ensure_fnum(fnum)
    pt.figure(fnum=fnum)
    ax = pt.gca()

    #unique_yaws, groupxs = vt.group_indices(yaws)

    ax.scatter(x_data, y_data, color=[1, 0, 0], s=1, marker='.')
    #pt.draw_stems(x_data, y_data, marker=markers, setlims=True, linestyle='')
    pt.dark_background()
    ax = pt.gca()
    ax.set_xlim(min(x_data) - .1, max(x_data) + .1)
    ax.set_ylim(min(y_data) - .1, max(y_data) + .1)


def show_image_time_distributions(ibs, gid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):

    CommandLine:
        python -m ibeis.other.dbinfo --exec-show_image_time_distributions --show
        python -m ibeis.other.dbinfo --exec-show_image_time_distributions --show --db lynx

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aids = ibeis.testdata_aids(ibs=ibs)
        >>> gid_list = ut.unique_unordered(ibs.get_annot_gids(aids))
        >>> result = show_image_time_distributions(ibs, gid_list)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    unixtime_list = ibs.get_image_unixtime(gid_list)
    unixtime_list = np.array(unixtime_list, dtype=np.float)
    unixtime_list = ut.list_replace(unixtime_list, -1, float('nan'))
    show_time_distributions(ibs, unixtime_list)


def show_time_distributions(ibs, unixtime_list):
    r"""
    """
    #import vtool as vt
    import plottool as pt
    unixtime_list = np.array(unixtime_list)
    num_nan = np.isnan(unixtime_list).sum()
    num_total = len(unixtime_list)
    unixtime_list = unixtime_list[~np.isnan(unixtime_list)]
    if False:
        from matplotlib import dates as mpldates
        #data_list = list(map(ut.unixtime_to_datetimeobj, unixtime_list))
        n, bins, patches = pt.plt.hist(unixtime_list, 365)
        #n_ = list(map(ut.unixtime_to_datetimeobj, n))
        #bins_ = list(map(ut.unixtime_to_datetimeobj, bins))
        pt.plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
        ax = pt.gca()
        #ax.xaxis.set_major_locator(mpldates.YearLocator())
        #hfmt = mpldates.DateFormatter('%y/%m/%d')
        #ax.xaxis.set_major_formatter(hfmt)
        mpldates.num2date(unixtime_list)
        #pt.gcf().autofmt_xdate()
        #y = pt.plt.normpdf( bins, unixtime_list.mean(), unixtime_list.std())
        #ax.set_xticks(bins_)
        #l = pt.plt.plot(bins_, y, 'k--', linewidth=1.5)
    else:
        pt.draw_time_distribution(unixtime_list)
        #pt.draw_histogram()
        ax = pt.gca()
        ax.set_xlabel('Date')
        ax.set_title('Timestamp distribution of %s. #nan=%d/%d' % (
            ibs.get_dbname_alias(),
            num_nan, num_total))
        pt.gcf().autofmt_xdate()

        icon = ibs.get_database_icon()
        if icon is not None:
            #import matplotlib as mpl
            #import vtool as vt
            ax = pt.gca()
            # Overlay a species icon
            # http://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
            #icon = vt.convert_image_list_colorspace([icon], 'RGB', 'BGR')[0]
            pt.overlay_icon(icon, coords=(0, 1), bbox_alignment=(0, 1))
            #imagebox = mpl.offsetbox.OffsetImage(icon, zoom=1.0)
            ##xy = [ax.get_xlim()[0] + 5, ax.get_ylim()[1]]
            ##ax.set_xlim(1, 100)
            ##ax.set_ylim(0, 100)
            ##x = np.array(ax.get_xlim()).sum() / 2
            ##y = np.array(ax.get_ylim()).sum() / 2
            ##xy = [x, y]
            ##print('xy = %r' % (xy,))
            ##x = np.nanmin(unixtime_list)
            ##xy = [x, y]
            ##print('xy = %r' % (xy,))
            ##ax.get_ylim()[0]]
            #xy = [ax.get_xlim()[0], ax.get_ylim()[1]]
            #ab = mpl.offsetbox.AnnotationBbox(
            #    imagebox, xy, xycoords='data',
            #    xybox=(-0., 0.),
            #    boxcoords="offset points",
            #    box_alignment=(0, 1), pad=0.0)
            #ax.add_artist(ab)

    if ut.get_argflag('--contextadjust'):
        #pt.adjust_subplots2(left=.08, bottom=.1, top=.9, wspace=.3, hspace=.1)
        pt.adjust_subplots2(use_argv=True)


def latex_dbstats(ibs_list, **kwargs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.dbinfo --exec-latex_dbstats --dblist testdb1
        python -m ibeis.other.dbinfo --exec-latex_dbstats --dblist testdb1 --show
        python -m ibeis.other.dbinfo --exec-latex_dbstats --dblist PZ_Master0 testdb1 --show
        python -m ibeis.other.dbinfo --exec-latex_dbstats --dblist PZ_Master0 PZ_MTEST GZ_ALL --show
        python -m ibeis.other.dbinfo --test-latex_dbstats --dblist GZ_ALL NNP_MasterGIRM_core --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> db_list = ut.get_argval('--dblist', type_=list, default=['testdb1'])
        >>> ibs_list = [ibeis.opendb(db=db) for db in db_list]
        >>> tabular_str = latex_dbstats(ibs_list)
        >>> tabular_cmd = ut.latex_newcommand(ut.latex_sanitize_command_name('DatabaseInfo'), tabular_str)
        >>> ut.copy_text_to_clipboard(tabular_cmd)
        >>> write_fpath = ut.get_argval('--write', type_=str, default=None)
        >>> if write_fpath is not None:
        >>>     fpath = ut.truepath(write_fpath)
        >>>     text = ut.readfrom(fpath)
        >>>     new_text = ut.replace_between_tags(text, tabular_cmd, '% <DBINFO>', '% </DBINFO>')
        >>>     ut.writeto(fpath, new_text)
        >>> ut.print_code(tabular_cmd, 'latex')
        >>> ut.quit_if_noshow()
        >>> ut.render_latex_text('\\noindent \n' + tabular_str)
    """

    import ibeis
    # Parse for aids test data
    aids_list = [ibeis.testdata_aids(ibs=ibs) for ibs in ibs_list]

    #dbinfo_list = [get_dbinfo(ibs, with_contrib=False, verbose=False) for ibs in ibs_list]
    dbinfo_list = [get_dbinfo(ibs, with_contrib=False, verbose=False, aid_list=aids)
                   for ibs, aids in zip(ibs_list, aids_list)]

    #title = db_name + ' database statistics'
    title = 'Database statistics'
    stat_title = '# Annotations per name (multiton)'

    #col_lbls = [
    #    'multiton',
    #    #'singleton',
    #    'total',
    #    'multiton',
    #    'singleton',
    #    'total',
    #]
    key_to_col_lbls = {
        'num_names_multiton':   'multiton',
        'num_names_singleton':  'singleton',
        'num_names':            'total',

        'num_multiton_annots':  'multiton',
        'num_singleton_annots': 'singleton',
        'num_unknown_annots':   'unknown',
        'num_annots':           'total',
    }
    # Structure of columns / multicolumns
    multi_col_keys = [
        ('# Names', (
            'num_names_multiton',
            #'num_names_singleton',
            'num_names',
        )),

        ('# Annots', (
            'num_multiton_annots',
            'num_singleton_annots',
            #'num_unknown_annots',
            'num_annots')),
    ]
    #multicol_lbls = [('# Names', 3), ('# Annots', 3)]
    multicol_lbls = [(mcolname, len(mcols)) for mcolname, mcols in multi_col_keys]

    # Flatten column labels
    col_keys = ut.flatten(ut.get_list_column(multi_col_keys, 1))
    col_lbls = ut.dict_take(key_to_col_lbls, col_keys)

    row_lbls   = []
    row_values = []

    #stat_col_lbls = ['max', 'min', 'mean', 'std', 'nMin', 'nMax']
    stat_col_lbls = ['max', 'min', 'mean', 'std', 'med']
    #stat_row_lbls = ['# Annot per Name (multiton)']
    stat_row_lbls = []
    stat_row_values = []

    SINGLE_TABLE = False
    EXTRA = True

    for ibs, dbinfo_locals in zip(ibs_list, dbinfo_list):
        row_ = ut.dict_take(dbinfo_locals, col_keys)
        dbname = ibs.get_dbname_alias()
        row_lbls.append(dbname)
        multiton_annot_stats = ut.get_stats(dbinfo_locals['multiton_nid2_nannots'], use_median=True)
        stat_rows = ut.dict_take(multiton_annot_stats, stat_col_lbls)
        if SINGLE_TABLE:
            row_.extend(stat_rows)
        else:
            stat_row_lbls.append(dbname)
            stat_row_values.append(stat_rows)

        row_values.append(row_)

    CENTERLINE = False
    AS_TABLE = True
    tablekw = dict(
        astable=AS_TABLE, centerline=CENTERLINE, FORCE_INT=False, precision=2,
        col_sep='', multicol_sep='|',
        **kwargs)

    if EXTRA:
        extra_keys = [
            #'species2_nAids',
            'qualtext2_nAnnots',
            'yawtext2_nAnnots',
        ]
        extra_titles = {
            'species2_nAids': 'Annotations per species.',
            'qualtext2_nAnnots': 'Annotations per quality.',
            'yawtext2_nAnnots': 'Annotations per viewpoint.',
        }
        extra_collbls = ut.ddict(list)
        extra_rowvalues = ut.ddict(list)
        extra_tables = ut.ddict(list)

        for ibs, dbinfo_locals in zip(ibs_list, dbinfo_list):
            for key in extra_keys:
                extra_collbls[key] = ut.unique_ordered(extra_collbls[key] + list(dbinfo_locals[key].keys()))

        extra_collbls['qualtext2_nAnnots'] = ['excellent', 'good', 'ok', 'poor', 'junk', 'UNKNOWN']
        #extra_collbls['yawtext2_nAnnots'] = ['backleft', 'left', 'frontleft', 'front', 'frontright', 'right', 'backright', 'back', None]
        extra_collbls['yawtext2_nAnnots'] = ['BL', 'L', 'FL', 'F', 'FR', 'R', 'BR', 'B', None]

        for ibs, dbinfo_locals in zip(ibs_list, dbinfo_list):
            for key in extra_keys:
                extra_rowvalues[key].append(ut.dict_take(dbinfo_locals[key], extra_collbls[key], 0))

        qualalias = {'UNKNOWN': None}

        extra_collbls['yawtext2_nAnnots'] = [ibs.const.YAWALIAS.get(val, val) for val in extra_collbls['yawtext2_nAnnots']]
        extra_collbls['qualtext2_nAnnots'] = [qualalias.get(val, val) for val in extra_collbls['qualtext2_nAnnots']]

        for key in extra_keys:
            extra_tables[key] = ut.util_latex.make_score_tabular(
                row_lbls, extra_collbls[key], extra_rowvalues[key],
                title=extra_titles[key], col_align='r', table_position='[h!]', **tablekw)

    #tabular_str = util_latex.tabular_join(tabular_body_list)
    if SINGLE_TABLE:
        col_lbls += stat_col_lbls
        multicol_lbls += [(stat_title, len(stat_col_lbls))]

    count_tabular_str = ut.util_latex.make_score_tabular(
        row_lbls, col_lbls, row_values, title=title, multicol_lbls=multicol_lbls, table_position='[ht!]', **tablekw)

    #print(row_lbls)

    if SINGLE_TABLE:
        tabular_str = count_tabular_str
    else:
        stat_tabular_str = ut.util_latex.make_score_tabular(
            stat_row_lbls, stat_col_lbls, stat_row_values, title=stat_title,
            col_align='r', table_position='[h!]', **tablekw)

        # Make a table of statistics
        if tablekw['astable']:
            tablesep = '\n%--\n'
        else:
            tablesep = '\\\\\n%--\n'
        if EXTRA:
            tabular_str = tablesep.join([count_tabular_str, stat_tabular_str] + ut.dict_take(extra_tables, extra_keys))
        else:
            tabular_str = tablesep.join([count_tabular_str, stat_tabular_str])

    return tabular_str


def get_short_infostr(ibs):
    """ Returns printable database information

    Args:
        ibs (IBEISController):  ibeis controller object

    Returns:
        str: infostr

    CommandLine:
        python -m ibeis.other.dbinfo --test-get_short_infostr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> infostr = get_short_infostr(ibs)
        >>> result = str(infostr)
        >>> print(result)
        dbname = 'testdb1'
        num_images = 13
        num_annotations = 13
        num_names = 7
    """
    dbname = ibs.get_dbname()
    #workdir = ut.unixpath(ibs.get_workdir())
    num_images = ibs.get_num_images()
    num_annotations = ibs.get_num_annotations()
    num_names = ibs.get_num_names()
    #workdir = %r
    infostr = ut.codeblock('''
    dbname = %s
    num_images = %r
    num_annotations = %r
    num_names = %r
    ''' % (ut.repr2(dbname), num_images, num_annotations, num_names))
    return infostr


def test_name_consistency(ibs):
    """
    Example:
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='PZ_Master0')
        >>> #ibs = ibeis.opendb(db='GZ_ALL')

    """
    from ibeis import ibsfuncs
    import utool as ut
    max_ = -1
    #max_ = 10
    valid_aids = ibs.get_valid_aids()[0:max_]
    valid_nids = ibs.get_valid_nids()[0:max_]
    ax2_nid = ibs.get_annot_name_rowids(valid_aids)
    nx2_aids = ibs.get_name_aids(valid_nids)

    print('len(valid_aids) = %r' % (len(valid_aids),))
    print('len(valid_nids) = %r' % (len(valid_nids),))
    print('len(ax2_nid) = %r' % (len(ax2_nid),))
    print('len(nx2_aids) = %r' % (len(nx2_aids),))

    # annots are grouped by names, so mapping aid back to nid should
    # result in each list having the same value
    _nids_list = ibsfuncs.unflat_map(ibs.get_annot_name_rowids, nx2_aids)
    print(_nids_list[-20:])
    print(nx2_aids[-20:])
    assert all(map(ut.allsame, _nids_list))


def print_feature_info(testres):
    """
    draws keypoint statistics for each test configuration

    Args:
        testres (ibeis.expt.test_result.TestResult): test result

    Ignore:
        import plottool as pt
        pt.qt4ensure()
        testres.draw_rank_cdf()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> ibs, testres = ibeis.testdata_expts(defaultdb='PZ_MTEST', a='timectrl', t='invar:AI=False')
        >>> (tex_nKpts, tex_kpts_stats, tex_scale_stats) = feature_info(ibs)
        >>> result = ('(tex_nKpts, tex_kpts_stats, tex_scale_stats) = %s' % (ut.repr2((tex_nKpts, tex_kpts_stats, tex_scale_stats)),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    import vtool as vt
    #ibs = testres.ibs
    def print_feat_stats(kpts, vecs):
        assert len(vecs) == len(kpts), 'disagreement'
        print('keypoints and vecs agree')
        flat_kpts = np.vstack(kpts)
        num_kpts = list(map(len, kpts))
        kpt_scale = vt.get_scales(flat_kpts)
        num_kpts_stats = ut.get_stats(num_kpts)
        scale_kpts_stats = ut.get_stats(kpt_scale)
        print('Number of ' + prefix + ' keypoints: ' + ut.repr3(num_kpts_stats, nl=0, precision=2))
        print('Scale of ' + prefix + ' keypoints: ' + ut.repr3(scale_kpts_stats, nl=0, precision=2))

    for cfgx in range(testres.nConfig):
        print('------------------')
        ut.colorprint(testres.cfgx2_lbl[cfgx], 'yellow')
        qreq_ = testres.cfgx2_qreq_[cfgx]
        depc = qreq_.ibs.depc_annot
        tablename = 'feat'
        prefix_list = ['query', 'data']
        config_pair = [qreq_.query_config2_, qreq_.data_config2_]
        aids_pair = [qreq_.qaids, qreq_.daids]
        for prefix, aids, config in zip(prefix_list, aids_pair, config_pair):
            config_ = depc._ensure_config(tablename, config)
            ut.colorprint(prefix + ' Config: ' + str(config_), 'blue')
            # Get keypoints and SIFT descriptors for this config
            kpts = depc.get(tablename, aids, 'kpts', config=config_)
            vecs = depc.get(tablename, aids, 'vecs', config=config_)
            # Check various stats of these pairs
            print_feat_stats(kpts, vecs)

    #kpts = np.vstack(cx2_kpts)
    #print('[dbinfo] --- LaTeX --- ')
    ##_printopts = np.get_printoptions()
    ##np.set_printoptions(precision=3)
    #scales = np.array(sorted(scales))
    #tex_scale_stats = util_latex.latex_get_stats(r'kpt scale', scales)
    #tex_nKpts       = util_latex.latex_scalar(r'\# kpts', len(kpts))
    #tex_kpts_stats  = util_latex.latex_get_stats(r'\# kpts/chip', cx2_nFeats)
    #print(tex_nKpts)
    #print(tex_kpts_stats)
    #print(tex_scale_stats)
    ##np.set_printoptions(**_printopts)
    #print('[dbinfo] ---/LaTeX --- ')
    #return (tex_nKpts, tex_kpts_stats, tex_scale_stats)


def cache_memory_stats(ibs, cid_list, fnum=None):
    print('[dev stats] cache_memory_stats()')
    #kpts_list = ibs.get_annot_kpts(cid_list)
    #desc_list = ibs.get_annot_vecs(cid_list)
    #nFeats_list = map(len, kpts_list)
    gx_list = np.unique(ibs.cx2_gx(cid_list))

    bytes_map = {
        'chip dbytes': [ut.file_bytes(fpath) for fpath in ibs.get_rchip_path(cid_list)],
        'img dbytes':  [ut.file_bytes(gpath) for gpath in ibs.gx2_gname(gx_list, full=True)],
        #'flann dbytes':  ut.file_bytes(flann_fpath),
    }

    byte_units = {
        'GB': 2 ** 30,
        'MB': 2 ** 20,
        'KB': 2 ** 10,
    }

    tabular_body_list = [
    ]

    convert_to = 'KB'
    for key, val in six.iteritems(bytes_map):
        key2 = key.replace('bytes', convert_to)
        if isinstance(val, list):
            val2 = [bytes_ / byte_units[convert_to] for bytes_ in val]
            tex_str = util_latex.latex_get_stats(key2, val2)
        else:
            val2 = val / byte_units[convert_to]
            tex_str = util_latex.latex_scalar(key2, val2)
        tabular_body_list.append(tex_str)

    tabular = util_latex.tabular_join(tabular_body_list)

    print(tabular)
    util_latex.render(tabular)

    if fnum is None:
        fnum = 0

    return fnum + 1


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.other.dbinfo
        python -m ibeis.other.dbinfo --allexamples
        python -m ibeis.other.dbinfo --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
