# -*- coding: utf-8 -*-
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
from wbia.algo.graph.state import POSTV, NEGTV

(print, rrr, profile) = ut.inject2(__name__)


def sight_resight_prob(N_range, nvisit1, nvisit2, resight):
    """
    https://en.wikipedia.org/wiki/Talk:Mark_and_recapture#Statistical_treatment
    http://stackoverflow.com/questions/31439875/infinite-summation-in-python/31442749
    """
    k, K, n = resight, nvisit1, nvisit2
    from scipy.special import comb

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
        # for x_blk in ut.ProgIter(x_strm, lbl='converging'):
        for x_blk in x_strm:
            diff = np.sum(func(x_blk), axis=axis)
            total += diff
            # error = abs(np.linalg.norm(diff))
            # print('error = %r' % (error,))
            if np.sqrt(diff.ravel().dot(diff.ravel())) <= eps:
                # Converged
                break
        return total

    numers = comb(N_range - K, n - k) / comb(N_range, n)

    @ut.memoize
    def func(N_):
        return comb(N_ - K, n - k) / comb(N_, n)

    denoms = []
    for N in ut.ProgIter(N_range, lbl='denoms'):
        x_strm = integers(start=(N + n - k), blk_size=100)
        denom = converge_inf_sum(func, x_strm, eps=1e-3)
        denoms.append(denom)

    # denom = sum([func(N_) for N_ in range(N_start, N_start * 2)])
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
        python -m wbia.other.dbinfo sight_resight_count --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> nvisit1 = 100
        >>> nvisit2 = 20
        >>> resight = 10
        >>> (pl_index, pl_error) = sight_resight_count(nvisit1, nvisit2, resight)
        >>> result = '(pl_index, pl_error) = %s' % ut.repr2((pl_index, pl_error))
        >>> pl_low = max(pl_index - pl_error, 1)
        >>> pl_high = pl_index + pl_error
        >>> print('pl_low = %r' % (pl_low,))
        >>> print('pl_high = %r' % (pl_high,))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
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
        >>> pt.plt.title('nvisit1=%r, nvisit2=%r, resight=%r' % (
        >>>     nvisit1, nvisit2, resight))
        >>> pt.plot(pl_index, pl_prob, 'rx', label='Lincoln Peterson Estimate')
        >>> pt.plot([pl_low, pl_high], [pl_prob, pl_prob], 'gx-',
        >>>         label='Lincoln Peterson Error Bar')
        >>> pt.legend()
        >>> ut.show_if_requested()
    """
    import math

    try:
        nvisit1 = float(nvisit1)
        nvisit2 = float(nvisit2)
        resight = float(resight)
        pl_index = int(math.ceil((nvisit1 * nvisit2) / resight))
        pl_error_num = float((nvisit1 ** 2) * nvisit2 * (nvisit2 - resight))
        pl_error_dom = float(resight ** 3)
        pl_error = int(math.ceil(1.96 * math.sqrt(pl_error_num / pl_error_dom)))
    except ZeroDivisionError:
        # pl_index = 'Undefined - Zero recaptured (k = 0)'
        pl_index = 0
        pl_error = 0
    return pl_index, pl_error


def dans_splits(ibs):
    """
    python -m wbia dans_splits --show

    Example:
        >>> # DISABLE_DOCTEST GGR
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> dbdir = '/media/danger/GGR/GGR-IBEIS'
        >>> dbdir = dbdir if ut.checkpath(dbdir) else ut.truepath('~/lev/media/danger/GGR/GGR-IBEIS')
        >>> ibs = wbia.opendb(dbdir=dbdir, allow_newdir=False)
        >>> import wbia.guitool as gt
        >>> gt.ensure_qtapp()
        >>> win = dans_splits(ibs)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> gt.qtapp_loop(qwin=win)
    """
    # pair = 9262, 932

    dans_aids = [
        26548,
        2190,
        9418,
        29965,
        14738,
        26600,
        3039,
        2742,
        8249,
        20154,
        8572,
        4504,
        34941,
        4040,
        7436,
        31866,
        28291,
        16009,
        7378,
        14453,
        2590,
        2738,
        22442,
        26483,
        21640,
        19003,
        13630,
        25395,
        20015,
        14948,
        21429,
        19740,
        7908,
        23583,
        14301,
        26912,
        30613,
        19719,
        21887,
        8838,
        16184,
        9181,
        8649,
        8276,
        14678,
        21950,
        4925,
        13766,
        12673,
        8417,
        2018,
        22434,
        21149,
        14884,
        5596,
        8276,
        14650,
        1355,
        21725,
        21889,
        26376,
        2867,
        6906,
        4890,
        21524,
        6690,
        14738,
        1823,
        35525,
        9045,
        31723,
        2406,
        5298,
        15627,
        31933,
        19535,
        9137,
        21002,
        2448,
        32454,
        12615,
        31755,
        20015,
        24573,
        32001,
        23637,
        3192,
        3197,
        8702,
        1240,
        5596,
        33473,
        23874,
        9558,
        9245,
        23570,
        33075,
        23721,
        24012,
        33405,
        23791,
        19498,
        33149,
        9558,
        4971,
        34183,
        24853,
        9321,
        23691,
        9723,
        9236,
        9723,
        21078,
        32300,
        8700,
        15334,
        6050,
        23277,
        31164,
        14103,
        21231,
        8007,
        10388,
        33387,
        4319,
        26880,
        8007,
        31164,
        32300,
        32140,
    ]

    is_hyrbid = [  # NOQA
        7123,
        7166,
        7157,
        7158,
    ]
    needs_mask = [26836, 29742]  # NOQA
    justfine = [19862]  # NOQA

    annots = ibs.annots(dans_aids)
    unique_nids = ut.unique(annots.nids)
    grouped_aids = ibs.get_name_aids(unique_nids)
    annot_groups = ibs._annot_groups(grouped_aids)

    split_props = {'splitcase', 'photobomb'}
    needs_tag = [
        len(split_props.intersection(ut.flatten(tags))) == 0
        for tags in annot_groups.match_tags
    ]
    num_needs_tag = sum(needs_tag)
    num_had_split = len(needs_tag) - num_needs_tag
    print('num_had_split = %r' % (num_had_split,))
    print('num_needs_tag = %r' % (num_needs_tag,))

    # all_annot_groups = ibs._annot_groups(ibs.group_annots_by_name(ibs.get_valid_aids())[0])
    # all_has_split = [len(split_props.intersection(ut.flatten(tags))) > 0 for tags in all_annot_groups.match_tags]
    # num_nondan = sum(all_has_split) - num_had_split
    # print('num_nondan = %r' % (num_nondan,))

    from wbia.algo.graph import graph_iden
    from wbia.viz import viz_graph2
    import wbia.guitool as gt
    import wbia.plottool as pt

    pt.qt4ensure()
    gt.ensure_qtapp()

    aids_list = ut.compress(grouped_aids, needs_tag)
    aids_list = [a for a in aids_list if len(a) > 1]
    print('len(aids_list) = %r' % (len(aids_list),))

    for aids in aids_list:
        infr = graph_iden.AnnotInference(ibs, aids)
        infr.initialize_graph()
        win = viz_graph2.AnnotGraphWidget(
            infr=infr, use_image=False, init_mode='rereview'
        )
        win.populate_edge_model()
        win.show()
        return win
    assert False


def fix_splits_interaction(ibs):
    """
    python -m wbia fix_splits_interaction --show

    Example:
        >>> # DISABLE_DOCTEST GGR
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> dbdir = '/media/danger/GGR/GGR-IBEIS'
        >>> dbdir = dbdir if ut.checkpath(dbdir) else ut.truepath('~/lev/media/danger/GGR/GGR-IBEIS')
        >>> ibs = wbia.opendb(dbdir=dbdir, allow_newdir=False)
        >>> import wbia.guitool as gt
        >>> gt.ensure_qtapp()
        >>> win = fix_splits_interaction(ibs)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> gt.qtapp_loop(qwin=win)
    """
    split_props = {'splitcase', 'photobomb'}
    all_annot_groups = ibs._annot_groups(
        ibs.group_annots_by_name(ibs.get_valid_aids())[0]
    )
    all_has_split = [
        len(split_props.intersection(ut.flatten(tags))) > 0
        for tags in all_annot_groups.match_tags
    ]
    tosplit_annots = ut.compress(all_annot_groups.annots_list, all_has_split)

    tosplit_annots = ut.take(tosplit_annots, ut.argsort(ut.lmap(len, tosplit_annots)))[
        ::-1
    ]
    if ut.get_argflag('--reverse'):
        tosplit_annots = tosplit_annots[::-1]
    print('len(tosplit_annots) = %r' % (len(tosplit_annots),))
    aids_list = [a.aids for a in tosplit_annots]

    from wbia.algo.graph import graph_iden
    from wbia.viz import viz_graph2
    import wbia.guitool as gt
    import wbia.plottool as pt

    pt.qt4ensure()
    gt.ensure_qtapp()

    for aids in ut.InteractiveIter(aids_list):
        infr = graph_iden.AnnotInference(ibs, aids)
        infr.initialize_graph()
        win = viz_graph2.AnnotGraphWidget(
            infr=infr, use_image=False, init_mode='rereview'
        )
        win.populate_edge_model()
        win.show()
    return win
    # assert False


def split_analysis(ibs):
    """
    CommandLine:
        python -m wbia.other.dbinfo split_analysis --show
        python -m wbia split_analysis --show
        python -m wbia split_analysis --show --good

    Ignore:
        # mount
        sshfs -o idmap=user lev:/ ~/lev

        # unmount
        fusermount -u ~/lev

    Example:
        >>> # DISABLE_DOCTEST GGR
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> dbdir = '/media/danger/GGR/GGR-IBEIS'
        >>> dbdir = dbdir if ut.checkpath(dbdir) else ut.truepath('~/lev/media/danger/GGR/GGR-IBEIS')
        >>> ibs = wbia.opendb(dbdir=dbdir, allow_newdir=False)
        >>> import wbia.guitool as gt
        >>> gt.ensure_qtapp()
        >>> win = split_analysis(ibs)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> gt.qtapp_loop(qwin=win)
        >>> #ut.show_if_requested()
    """
    # nid_list = ibs.get_valid_nids(filter_empty=True)
    import datetime

    day1 = datetime.date(2016, 1, 30)
    day2 = datetime.date(2016, 1, 31)

    filter_kw = {
        'multiple': None,
        # 'view': ['right'],
        # 'minqual': 'good',
        'is_known': True,
        'min_pername': 1,
    }
    aids1 = ibs.filter_annots_general(
        filter_kw=ut.dict_union(
            filter_kw,
            {
                'min_unixtime': ut.datetime_to_posixtime(ut.date_to_datetime(day1, 0.0)),
                'max_unixtime': ut.datetime_to_posixtime(ut.date_to_datetime(day1, 1.0)),
            },
        )
    )
    aids2 = ibs.filter_annots_general(
        filter_kw=ut.dict_union(
            filter_kw,
            {
                'min_unixtime': ut.datetime_to_posixtime(ut.date_to_datetime(day2, 0.0)),
                'max_unixtime': ut.datetime_to_posixtime(ut.date_to_datetime(day2, 1.0)),
            },
        )
    )
    all_aids = aids1 + aids2
    all_annots = ibs.annots(all_aids)
    print('%d annots on day 1' % (len(aids1)))
    print('%d annots on day 2' % (len(aids2)))
    print('%d annots overall' % (len(all_annots)))
    print('%d names overall' % (len(ut.unique(all_annots.nids))))

    nid_list, annots_list = all_annots.group(all_annots.nids)

    REVIEWED_EDGES = True
    if REVIEWED_EDGES:
        aids_list = [annots.aids for annots in annots_list]
        # aid_pairs = [annots.get_am_aidpairs() for annots in annots_list]  # Slower
        aid_pairs = ibs.get_unflat_am_aidpairs(aids_list)  # Faster
    else:
        # ALL EDGES
        aid_pairs = [annots.get_aidpairs() for annots in annots_list]

    speeds_list = ibs.unflat_map(ibs.get_annotpair_speeds, aid_pairs)
    import vtool as vt

    max_speeds = np.array([vt.safe_max(s, nans=False) for s in speeds_list])

    nan_idx = np.where(np.isnan(max_speeds))[0]
    inf_idx = np.where(np.isinf(max_speeds))[0]
    bad_idx = sorted(ut.unique(ut.flatten([inf_idx, nan_idx])))
    ok_idx = ut.index_complement(bad_idx, len(max_speeds))

    print('#nan_idx = %r' % (len(nan_idx),))
    print('#inf_idx = %r' % (len(inf_idx),))
    print('#ok_idx = %r' % (len(ok_idx),))

    ok_speeds = max_speeds[ok_idx]
    ok_nids = ut.take(nid_list, ok_idx)
    ok_annots = ut.take(annots_list, ok_idx)
    sortx = np.argsort(ok_speeds)[::-1]

    sorted_speeds = np.array(ut.take(ok_speeds, sortx))
    sorted_annots = np.array(ut.take(ok_annots, sortx))
    sorted_nids = np.array(ut.take(ok_nids, sortx))  # NOQA

    sorted_speeds = np.clip(sorted_speeds, 0, 100)

    # idx = vt.find_elbow_point(sorted_speeds)
    # EXCESSIVE_SPEED = sorted_speeds[idx]
    # http://www.infoplease.com/ipa/A0004737.html
    # http://www.speedofanimals.com/animals/zebra
    # ZEBRA_SPEED_MAX  = 64  # km/h
    # ZEBRA_SPEED_RUN  = 50  # km/h
    ZEBRA_SPEED_SLOW_RUN = 20  # km/h
    # ZEBRA_SPEED_FAST_WALK = 10  # km/h
    # ZEBRA_SPEED_WALK = 7  # km/h

    MAX_SPEED = ZEBRA_SPEED_SLOW_RUN
    # MAX_SPEED = ZEBRA_SPEED_WALK
    # MAX_SPEED = EXCESSIVE_SPEED

    flags = sorted_speeds > MAX_SPEED
    flagged_ok_annots = ut.compress(sorted_annots, flags)
    inf_annots = ut.take(annots_list, inf_idx)
    flagged_annots = inf_annots + flagged_ok_annots

    print('MAX_SPEED = %r km/h' % (MAX_SPEED,))
    print('%d annots with infinite speed' % (len(inf_annots),))
    print('%d annots with large speed' % (len(flagged_ok_annots),))
    print('Marking all pairs of annots above the threshold as non-matching')

    from wbia.algo.graph import graph_iden
    import networkx as nx

    progkw = dict(freq=1, bs=True, est_window=len(flagged_annots))

    bad_edges_list = []
    good_edges_list = []
    for annots in ut.ProgIter(flagged_annots, lbl='flag speeding names', **progkw):
        edge_to_speeds = annots.get_speeds()
        bad_edges = [edge for edge, speed in edge_to_speeds.items() if speed > MAX_SPEED]
        good_edges = [
            edge for edge, speed in edge_to_speeds.items() if speed <= MAX_SPEED
        ]
        bad_edges_list.append(bad_edges)
        good_edges_list.append(good_edges)
    all_bad_edges = ut.flatten(bad_edges_list)
    good_edges_list = ut.flatten(good_edges_list)
    print('num_bad_edges = %r' % (len(ut.flatten(bad_edges_list)),))
    print('num_bad_edges = %r' % (len(ut.flatten(good_edges_list)),))

    if 1:
        from wbia.viz import viz_graph2
        import wbia.guitool as gt

        gt.ensure_qtapp()

        if ut.get_argflag('--good'):
            print('Looking at GOOD (no speed problems) edges')
            aid_pairs = good_edges_list
        else:
            print('Looking at BAD (speed problems) edges')
            aid_pairs = all_bad_edges
        aids = sorted(list(set(ut.flatten(aid_pairs))))
        infr = graph_iden.AnnotInference(ibs, aids, verbose=False)
        infr.initialize_graph()

        # Use random scores to randomize sort order
        rng = np.random.RandomState(0)
        scores = (-rng.rand(len(aid_pairs)) * 10).tolist()
        infr.graph.add_edges_from(aid_pairs)

        if True:
            edge_sample_size = 250
            pop_nids = ut.unique(ibs.get_annot_nids(ut.unique(ut.flatten(aid_pairs))))
            sorted_pairs = ut.sortedby(aid_pairs, scores)[::-1][0:edge_sample_size]
            sorted_nids = ibs.get_annot_nids(ut.take_column(sorted_pairs, 0))
            sample_size = len(ut.unique(sorted_nids))
            am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(
                *zip(*sorted_pairs)
            )
            flags = ut.not_list(ut.flag_None_items(am_rowids))
            # am_rowids = ut.compress(am_rowids, flags)
            positive_tags = ['SplitCase', 'Photobomb']
            flags_list = [
                ut.replace_nones(ibs.get_annotmatch_prop(tag, am_rowids), 0)
                for tag in positive_tags
            ]
            print(
                'edge_case_hist: '
                + ut.repr3(
                    [
                        '%s %s' % (txt, sum(flags_))
                        for flags_, txt in zip(flags_list, positive_tags)
                    ]
                )
            )
            is_positive = ut.or_lists(*flags_list)
            num_positive = sum(
                ut.lmap(any, ut.group_items(is_positive, sorted_nids).values())
            )
            pop = len(pop_nids)
            print(
                'A positive is any edge flagged as a %s'
                % (ut.conj_phrase(positive_tags, 'or'),)
            )
            print('--- Sampling wrt edges ---')
            print('edge_sample_size  = %r' % (edge_sample_size,))
            print('edge_population_size = %r' % (len(aid_pairs),))
            print('num_positive_edges = %r' % (sum(is_positive)))
            print('--- Sampling wrt names ---')
            print('name_population_size = %r' % (pop,))
            vt.calc_error_bars_from_sample(
                sample_size, num_positive, pop, conf_level=0.95
            )

        nx.set_edge_attributes(
            infr.graph, name='score', values=dict(zip(aid_pairs, scores))
        )

        win = viz_graph2.AnnotGraphWidget(infr=infr, use_image=False, init_mode=None)
        win.populate_edge_model()
        win.show()
        return win
        # Make review interface for only bad edges

    infr_list = []
    iter_ = list(zip(flagged_annots, bad_edges_list))
    for annots, bad_edges in ut.ProgIter(iter_, lbl='creating inference', **progkw):
        aids = annots.aids
        nids = [1] * len(aids)
        infr = graph_iden.AnnotInference(ibs, aids, nids, verbose=False)
        infr.initialize_graph()
        infr.reset_feedback()
        infr_list.append(infr)

    # Check which ones are user defined as incorrect
    # num_positive = 0
    # for infr in infr_list:
    #    flag = np.any(infr.get_feedback_probs()[0] == 0)
    #    num_positive += flag
    # print('num_positive = %r' % (num_positive,))
    # pop = len(infr_list)
    # print('pop = %r' % (pop,))

    iter_ = list(zip(infr_list, bad_edges_list))
    for infr, bad_edges in ut.ProgIter(iter_, lbl='adding speed edges', **progkw):
        flipped_edges = []
        for aid1, aid2 in bad_edges:
            if infr.graph.has_edge(aid1, aid2):
                flipped_edges.append((aid1, aid2))
            infr.add_feedback((aid1, aid2), NEGTV)
        nx.set_edge_attributes(infr.graph, name='_speed_split', values='orig')
        nx.set_edge_attributes(
            infr.graph, name='_speed_split', values={edge: 'new' for edge in bad_edges}
        )
        nx.set_edge_attributes(
            infr.graph,
            name='_speed_split',
            values={edge: 'flip' for edge in flipped_edges},
        )

    # for infr in ut.ProgIter(infr_list, lbl='flagging speeding edges', **progkw):
    #    annots = ibs.annots(infr.aids)
    #    edge_to_speeds = annots.get_speeds()
    #    bad_edges = [edge for edge, speed in edge_to_speeds.items() if speed > MAX_SPEED]

    def inference_stats(infr_list_):
        relabel_stats = []
        for infr in infr_list_:
            num_ccs, num_inconsistent = infr.relabel_using_reviews()
            state_hist = ut.dict_hist(
                nx.get_edge_attributes(infr.graph, 'decision').values()
            )
            if POSTV not in state_hist:
                state_hist[POSTV] = 0
            hist = ut.dict_hist(
                nx.get_edge_attributes(infr.graph, '_speed_split').values()
            )

            subgraphs = infr.positive_connected_compoments()
            subgraph_sizes = [len(g) for g in subgraphs]

            info = ut.odict(
                [
                    ('num_nonmatch_edges', state_hist[NEGTV]),
                    ('num_match_edges', state_hist[POSTV]),
                    (
                        'frac_nonmatch_edges',
                        state_hist[NEGTV] / (state_hist[POSTV] + state_hist[NEGTV]),
                    ),
                    ('num_inconsistent', num_inconsistent),
                    ('num_ccs', num_ccs),
                    ('edges_flipped', hist.get('flip', 0)),
                    ('edges_unchanged', hist.get('orig', 0)),
                    ('bad_unreviewed_edges', hist.get('new', 0)),
                    ('orig_size', len(infr.graph)),
                    ('new_sizes', subgraph_sizes),
                ]
            )
            relabel_stats.append(info)
        return relabel_stats

    relabel_stats = inference_stats(infr_list)

    print('\nAll Split Info:')
    lines = []
    for key in relabel_stats[0].keys():
        data = ut.take_column(relabel_stats, key)
        if key == 'new_sizes':
            data = ut.flatten(data)
        lines.append(
            'stats(%s) = %s'
            % (key, ut.repr2(ut.get_stats(data, use_median=True), precision=2))
        )
    print('\n'.join(ut.align_lines(lines, '=')))

    num_incon_list = np.array(ut.take_column(relabel_stats, 'num_inconsistent'))
    can_split_flags = num_incon_list == 0
    print('Can trivially split %d / %d' % (sum(can_split_flags), len(can_split_flags)))

    splittable_infrs = ut.compress(infr_list, can_split_flags)

    relabel_stats = inference_stats(splittable_infrs)

    print('\nTrival Split Info:')
    lines = []
    for key in relabel_stats[0].keys():
        if key in ['num_inconsistent']:
            continue
        data = ut.take_column(relabel_stats, key)
        if key == 'new_sizes':
            data = ut.flatten(data)
        lines.append(
            'stats(%s) = %s'
            % (key, ut.repr2(ut.get_stats(data, use_median=True), precision=2))
        )
    print('\n'.join(ut.align_lines(lines, '=')))

    num_match_edges = np.array(ut.take_column(relabel_stats, 'num_match_edges'))
    num_nonmatch_edges = np.array(ut.take_column(relabel_stats, 'num_nonmatch_edges'))
    flags1 = np.logical_and(num_match_edges > num_nonmatch_edges, num_nonmatch_edges < 3)
    reasonable_infr = ut.compress(splittable_infrs, flags1)

    new_sizes_list = ut.take_column(relabel_stats, 'new_sizes')
    flags2 = [
        len(sizes) == 2 and sum(sizes) > 4 and (min(sizes) / max(sizes)) > 0.3
        for sizes in new_sizes_list
    ]
    reasonable_infr = ut.compress(splittable_infrs, flags2)
    print('#reasonable_infr = %r' % (len(reasonable_infr),))

    for infr in ut.InteractiveIter(reasonable_infr):
        annots = ibs.annots(infr.aids)
        edge_to_speeds = annots.get_speeds()
        print('max_speed = %r' % (max(edge_to_speeds.values())),)
        infr.initialize_visual_node_attrs()
        infr.show_graph(use_image=True, only_reviewed=True)

    rest = ~np.logical_or(flags1, flags2)
    nonreasonable_infr = ut.compress(splittable_infrs, rest)
    rng = np.random.RandomState(0)
    random_idx = ut.random_indexes(len(nonreasonable_infr) - 1, 15, rng=rng)
    random_infr = ut.take(nonreasonable_infr, random_idx)
    for infr in ut.InteractiveIter(random_infr):
        annots = ibs.annots(infr.aids)
        edge_to_speeds = annots.get_speeds()
        print('max_speed = %r' % (max(edge_to_speeds.values())),)
        infr.initialize_visual_node_attrs()
        infr.show_graph(use_image=True, only_reviewed=True)

    # import scipy.stats as st
    # conf_interval = .95
    # st.norm.cdf(conf_interval)
    # view-source:http://www.surveysystem.com/sscalc.htm
    # zval = 1.96  # 95 percent confidence
    # zValC = 3.8416  #
    # zValC = 6.6564

    # import statsmodels.stats.api as sms
    # es = sms.proportion_effectsize(0.5, 0.75)
    # sms.NormalIndPower().solve_power(es, power=0.9, alpha=0.05, ratio=1)

    pop = 279
    num_positive = 3
    sample_size = 15
    conf_level = 0.95
    # conf_level = .99
    vt.calc_error_bars_from_sample(sample_size, num_positive, pop, conf_level)
    print('---')
    vt.calc_error_bars_from_sample(sample_size + 38, num_positive, pop, conf_level)
    print('---')
    vt.calc_error_bars_from_sample(sample_size + 38 / 3, num_positive, pop, conf_level)
    print('---')

    vt.calc_error_bars_from_sample(15 + 38, num_positive=3, pop=675, conf_level=0.95)
    vt.calc_error_bars_from_sample(15, num_positive=3, pop=675, conf_level=0.95)

    pop = 279
    # err_frac = .05  # 5%
    err_frac = 0.10  # 10%
    conf_level = 0.95
    vt.calc_sample_from_error_bars(err_frac, pop, conf_level)

    pop = 675
    vt.calc_sample_from_error_bars(err_frac, pop, conf_level)
    vt.calc_sample_from_error_bars(0.05, pop, conf_level=0.95, prior=0.1)
    vt.calc_sample_from_error_bars(0.05, pop, conf_level=0.68, prior=0.2)
    vt.calc_sample_from_error_bars(0.10, pop, conf_level=0.68)

    vt.calc_error_bars_from_sample(100, num_positive=5, pop=675, conf_level=0.95)
    vt.calc_error_bars_from_sample(100, num_positive=5, pop=675, conf_level=0.68)

    # flagged_nids = [a.nids[0] for a in flagged_annots]
    # all_nids = ibs.get_valid_nids()
    # remain_nids = ut.setdiff(all_nids, flagged_nids)
    # nAids_list = np.array(ut.lmap(len, ibs.get_name_aids(all_nids)))
    # nAids_list = np.array(ut.lmap(len, ibs.get_name_aids(remain_nids)))

    # #graph = infr.graph

    # g2 = infr.graph.copy()
    # [ut.nx_delete_edge_attr(g2, a) for a in infr.visual_edge_attrs]
    # g2.edge


def estimate_ggr_count(ibs):
    """
    Example:
        >>> # DISABLE_DOCTEST GGR
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> dbdir = ut.truepath('~/lev/media/danger/GGR/GGR-IBEIS')
        >>> ibs = wbia.opendb(dbdir='/home/joncrall/lev/media/danger/GGR/GGR-IBEIS')
    """
    import datetime

    day1 = datetime.date(2016, 1, 30)
    day2 = datetime.date(2016, 1, 31)

    filter_kw = {
        'multiple': None,
        'minqual': 'good',
        'is_known': True,
        'min_pername': 1,
        'view': ['right'],
    }
    print('\nOnly Single-Animal-In-Annotation:')
    filter_kw['multiple'] = False
    estimate_twoday_count(ibs, day1, day2, filter_kw)

    print('\nOnly Multi-Animal-In-Annotation:')
    filter_kw['multiple'] = True
    estimate_twoday_count(ibs, day1, day2, filter_kw)

    print('\nUsing Both:')
    filter_kw['multiple'] = None
    return estimate_twoday_count(ibs, day1, day2, filter_kw)


def estimate_twoday_count(ibs, day1, day2, filter_kw):
    # gid_list = ibs.get_valid_gids()
    all_images = ibs.images()
    dates = [dt.date() for dt in all_images.datetime]
    date_to_images = all_images.group_items(dates)
    date_to_images = ut.sort_dict(date_to_images)
    # date_hist = ut.map_dict_vals(len, date2_gids)
    # print('date_hist = %s' % (ut.repr2(date_hist, nl=2),))
    verbose = 0

    visit_dates = [day1, day2]
    visit_info_list_ = []
    for day in visit_dates:
        images = date_to_images[day]
        aids = ut.flatten(images.aids)
        aids = ibs.filter_annots_general(aids, filter_kw=filter_kw, verbose=verbose)
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
    from wbia.other import dbinfo

    aids_day1, aids_day2 = ut.take_column(visit_info_list_, 'unique_aids')
    nids_day1, nids_day2 = ut.take_column(visit_info_list_, 'unique_nids')
    resight_nids = ut.isect(nids_day1, nids_day2)
    nsight1 = len(nids_day1)
    nsight2 = len(nids_day2)
    resight = len(resight_nids)
    lp_index, lp_error = dbinfo.sight_resight_count(nsight1, nsight2, resight)

    if False:
        from wbia.other import dbinfo

        print('DAY 1 STATS:')
        _ = dbinfo.get_dbinfo(ibs, aid_list=aids_day1)  # NOQA
        print('DAY 2 STATS:')
        _ = dbinfo.get_dbinfo(ibs, aid_list=aids_day2)  # NOQA
        print('COMBINED STATS:')
        _ = dbinfo.get_dbinfo(ibs, aid_list=aids_day1 + aids_day2)  # NOQA

    print('%d annots on day 1' % (len(aids_day1)))
    print('%d annots on day 2' % (len(aids_day2)))
    print('%d names on day 1' % (nsight1,))
    print('%d names on day 2' % (nsight2,))
    print('resight = %r' % (resight,))
    print('lp_index = %r ± %r' % (lp_index, lp_error))
    return nsight1, nsight2, resight, lp_index, lp_error


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
            for ams, aids, aids1, aids2 in zip(
                ams_list, aids_list, aids1_list, aids2_list
            ):
                edge_nodes = set(aids1 + aids2)
                # #if len(edge_nodes) != len(set(aids)):
                #    #print('--')
                #    #print('aids = %r' % (aids,))
                #    #print('edge_nodes = %r' % (edge_nodes,))
                bad_aids = edge_nodes - set(aids)
                if len(bad_aids) > 0:
                    print('bad_aids = %r' % (bad_aids,))
                unlinked_aids = set(aids) - edge_nodes
                mst_links = list(ut.itertwo(list(unlinked_aids) + list(edge_nodes)[:1]))
                bad_aids.add(None)
                user_links = [
                    (u, v)
                    for (u, v) in zip(aids1, aids2)
                    if u not in bad_aids and v not in bad_aids
                ]
                new_edges = mst_links + user_links
                new_edges = [
                    (int(u), int(v))
                    for u, v in new_edges
                    if u not in bad_aids and v not in bad_aids
                ]
                edges += new_edges
            info['edges'] = edges

        # Add edges between days
        grouped_aids1, grouped_aids2 = ut.take_column(visit_info_list, 'grouped_aids')
        nids_day1, nids_day2 = ut.take_column(visit_info_list, 'unique_nids')
        resight_nids = ut.isect(nids_day1, nids_day2)

        resight_aids1 = ut.take(grouped_aids1, resight_nids)
        resight_aids2 = ut.take(grouped_aids2, resight_nids)
        # resight_aids3 = [list(aids1) + list(aids2) for aids1, aids2 in zip(resight_aids1, resight_aids2)]

        ams_list = ibs.get_annotmatch_rowids_between_groups(resight_aids1, resight_aids2)
        aids1_list = ibs.unflat_map(ibs.get_annotmatch_aid1, ams_list)
        aids2_list = ibs.unflat_map(ibs.get_annotmatch_aid2, ams_list)

        between_edges = []
        for ams, aids1, aids2, rawaids1, rawaids2 in zip(
            ams_list, aids1_list, aids2_list, resight_aids1, resight_aids2
        ):
            link_aids = aids1 + aids2
            rawaids3 = rawaids1 + rawaids2
            badaids = ut.setdiff(link_aids, rawaids3)
            assert not badaids
            user_links = [
                (int(u), int(v))
                for (u, v) in zip(aids1, aids2)
                if u is not None and v is not None
            ]
            # HACK THIS OFF
            user_links = []
            if len(user_links) == 0:
                # Hack in an edge
                between_edges += [(rawaids1[0], rawaids2[0])]
            else:
                between_edges += user_links

        assert np.all(
            0
            == np.diff(
                np.array(ibs.unflat_map(ibs.get_annot_nids, between_edges)), axis=1
            )
        )

        import wbia.plottool as pt
        import networkx as nx

        # pt.qt4ensure()
        # len(list(nx.connected_components(graph1)))
        # print(ut.graph_info(graph1))

        # Layout graph
        layoutkw = dict(
            prog='neato',
            draw_implicit=False,
            splines='line',
            # splines='curved',
            # splines='spline',
            # sep=10 / 72,
            # prog='dot', rankdir='TB',
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
                nx.set_node_attributes(g, name='pin', values='true')

            new_graph = nx.compose_all(graph_list_)
            # pt.show_nx(new_graph, layout='custom', node_labels=False, as_directed=False)  # NOQA
            return new_graph

        # Construct graph
        for count, info in enumerate(visit_info_list):
            graph = nx.Graph()
            edges = [
                (int(u), int(v))
                for u, v in info['edges']
                if u is not None and v is not None
            ]
            graph.add_edges_from(edges, attr_dict={'zorder': 10})
            nx.set_node_attributes(graph, name='zorder', values=20)

            # Layout in neato
            _ = pt.nx_agraph_layout(graph, inplace=True, **layoutkw)  # NOQA

            # Extract components and then flatten in nid ordering
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

            # pt.show_nx(new_graph, layout='custom', node_labels=False, as_directed=False)  # NOQA
            info['graph'] = new_graph

        graph1_, graph2_ = ut.take_column(visit_info_list, 'graph')
        if False:
            pt.show_nx(graph1_, layout='custom', node_labels=False, as_directed=False)
            pt.show_nx(graph2_, layout='custom', node_labels=False, as_directed=False)

        graph_list = [graph1_, graph2_]
        twoday_graph = stack_graphs(graph_list, vert=True, pad=None)
        nx.set_node_attributes(twoday_graph, name='pin', values='true')

        if debug:
            ut.nx_delete_None_edge_attr(twoday_graph)
            ut.nx_delete_None_node_attr(twoday_graph)
            print('twoday_graph(pre) info' + ut.repr3(ut.graph_info(twoday_graph), nl=2))

        # Hack, no idea why there are nodes that dont exist here
        between_edges_ = [
            edge
            for edge in between_edges
            if twoday_graph.has_node(edge[0]) and twoday_graph.has_node(edge[1])
        ]

        twoday_graph.add_edges_from(between_edges_, attr_dict={'alpha': 0.2, 'zorder': 0})
        ut.nx_ensure_agraph_color(twoday_graph)

        layoutkw['splines'] = 'line'
        layoutkw['prog'] = 'neato'
        agraph = pt.nx_agraph_layout(
            twoday_graph, inplace=True, return_agraph=True, **layoutkw
        )[
            -1
        ]  # NOQA
        if False:
            fpath = ut.truepath('~/ggr_graph.png')
            agraph.draw(fpath)
            ut.startfile(fpath)

        if debug:
            print('twoday_graph(post) info' + ut.repr3(ut.graph_info(twoday_graph)))

        pt.show_nx(twoday_graph, layout='custom', node_labels=False, as_directed=False)


def cheetah_stats(ibs):
    filters = [
        dict(view=['right', 'frontright', 'backright'], minqual='good'),
        dict(view=['right', 'frontright', 'backright']),
    ]
    for filtkw in filters:
        annots = ibs.annots(ibs.filter_annots_general(**filtkw))
        unique_nids, grouped_annots = annots.group(annots.nids)
        annots_per_name = ut.lmap(len, grouped_annots)
        annots_per_name_freq = ut.dict_hist(annots_per_name)

        def bin_mapper(num):
            if num < 5:
                return (num, num + 1)
            else:
                for bin, mod in [(20, 5), (50, 10)]:
                    if num < bin:
                        low = (num // mod) * mod
                        high = low + mod
                        return (low, high)
                if num >= bin:
                    return (bin, None)
                else:
                    assert False, str(num)

        hist = ut.ddict(lambda: 0)
        for num in annots_per_name:
            hist[bin_mapper(num)] += 1
        hist = ut.sort_dict(hist)

        print('------------')
        print('filters = %s' % ut.repr4(filtkw))
        print('num_annots = %r' % (len(annots)))
        print('num_names = %r' % (len(unique_nids)))
        print('annots_per_name_freq = %s' % (ut.repr4(annots_per_name_freq)))
        print('annots_per_name_freq (ranges) = %s' % (ut.repr4(hist)))
        assert sum(hist.values()) == len(unique_nids)


def print_feature_info(testres):
    """
    draws keypoint statistics for each test configuration

    Args:
        testres (wbia.expt.test_result.TestResult): test result

    Ignore:
        import wbia.plottool as pt
        pt.qt4ensure()
        testres.draw_rank_cmc()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> ibs, testres = wbia.testdata_expts(defaultdb='PZ_MTEST', a='timectrl', t='invar:AI=False')
        >>> (tex_nKpts, tex_kpts_stats, tex_scale_stats) = feature_info(ibs)
        >>> result = ('(tex_nKpts, tex_kpts_stats, tex_scale_stats) = %s' % (ut.repr2((tex_nKpts, tex_kpts_stats, tex_scale_stats)),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import vtool as vt

    # ibs = testres.ibs
    def print_feat_stats(kpts, vecs):
        assert len(vecs) == len(kpts), 'disagreement'
        print('keypoints and vecs agree')
        flat_kpts = np.vstack(kpts)
        num_kpts = list(map(len, kpts))
        kpt_scale = vt.get_scales(flat_kpts)
        num_kpts_stats = ut.get_stats(num_kpts)
        scale_kpts_stats = ut.get_stats(kpt_scale)
        print(
            'Number of '
            + prefix
            + ' keypoints: '
            + ut.repr3(num_kpts_stats, nl=0, precision=2)
        )
        print(
            'Scale of '
            + prefix
            + ' keypoints: '
            + ut.repr3(scale_kpts_stats, nl=0, precision=2)
        )

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

    # kpts = np.vstack(cx2_kpts)
    # print('[dbinfo] --- LaTeX --- ')
    # # _printopts = np.get_printoptions()
    # # np.set_printoptions(precision=3)
    # scales = np.array(sorted(scales))
    # tex_scale_stats = util_latex.latex_get_stats(r'kpt scale', scales)
    # tex_nKpts       = util_latex.latex_scalar(r'\# kpts', len(kpts))
    # tex_kpts_stats  = util_latex.latex_get_stats(r'\# kpts/chip', cx2_nFeats)
    # print(tex_nKpts)
    # print(tex_kpts_stats)
    # print(tex_scale_stats)
    # # np.set_printoptions(**_printopts)
    # print('[dbinfo] ---/LaTeX --- ')
    # return (tex_nKpts, tex_kpts_stats, tex_scale_stats)


def tst_name_consistency(ibs):
    """
    Example:
        >>> import wbia
        >>> ibs = wbia.opendb(db='PZ_Master0')
        >>> #ibs = wbia.opendb(db='GZ_ALL')

    """
    from wbia.other import ibsfuncs
    import utool as ut

    max_ = -1
    # max_ = 10
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
