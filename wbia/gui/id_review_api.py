# -*- coding: utf-8 -*-
"""
CommandLine:
    python -m wbia.gui.inspect_gui --test-test_review_widget --show
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from functools import partial
from wbia.viz import viz_helpers as vh
import wbia.guitool as gt
import numpy as np
import utool as ut

(print, rrr, profile) = ut.inject2(__name__, '[id_review_api]')


MATCHED_STATUS_TEXT = 'Matched'
REVIEWED_STATUS_TEXT = 'Reviewed'


REVIEW_CFG_DEFAULTS = {
    'ranks_top': 5,
    'directed': False,
    'name_scoring': True,
    'filter_reviewed': True,
    'filter_photobombs': True,
    'filter_true_matches': True,
    'show_chips': True,
    'filter_duplicate_true_matches': False,
}


@profile
def get_review_edges(cm_list, ibs=None, review_cfg={}):
    r"""
    Needs to be moved to a better file. Maybe something to do with
    identification.

    Returns a list of matches that should be inspected
    This function is more lightweight than orgres or allres.
    Used in id_review_api and interact_qres2

    Args:
        cm_list (list): list of chip match objects
        ranks_top (int): put all ranks less than this number into the graph
        directed (bool):

    Returns:
        tuple: review_edges = (qaid_arr, daid_arr, score_arr, rank_arr)

    CommandLine:
        python -m wbia.gui.id_review_api get_review_edges:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia.gui.id_review_api import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> qreq_ = wbia.main_helpers.testdata_qreq_()
        >>> cm_list = qreq_.execute()
        >>> review_cfg = dict(ranks_top=5, directed=True, name_scoring=False,
        >>>                   filter_true_matches=True)
        >>> review_edges = get_review_edges(cm_list, ibs=ibs, review_cfg=review_cfg)
        >>> print(review_edges)

    Example1:
        >>> # UNSTABLE_DOCTEST
        >>> from wbia.gui.id_review_api import *  # NOQA
        >>> import wbia
        >>> cm_list, qreq_ = wbia.testdata_cmlist('PZ_MTEST', a='default:qsize=5,dsize=20')
        >>> review_cfg = dict(ranks_top=5, directed=True, name_scoring=False,
        >>>                   filter_reviewed=False, filter_true_matches=True)
        >>> review_edges = get_review_edges(cm_list, review_cfg=review_cfg, ibs=ibs)
        >>> print(review_edges)

    Example3:
        >>> # UNSTABLE_DOCTEST
        >>> from wbia.gui.id_review_api import *  # NOQA
        >>> import wbia
        >>> cm_list, qreq_ = wbia.testdata_cmlist('PZ_MTEST', a='default:qsize=1,dsize=100')
        >>> review_cfg = dict(ranks_top=1, directed=False, name_scoring=False,
        >>>                   filter_reviewed=False, filter_true_matches=True)
        >>> review_edges = get_review_edges(cm_list, review_cfg=review_cfg, ibs=ibs)
        >>> print(review_edges)

    Example4:
        >>> # UNSTABLE_DOCTEST
        >>> from wbia.gui.id_review_api import *  # NOQA
        >>> import wbia
        >>> cm_list, qreq_ = wbia.testdata_cmlist('PZ_MTEST', a='default:qsize=10,dsize=10')
        >>> ranks_top = 3
        >>> review_cfg = dict(ranks_top=3, directed=False, name_scoring=False,
        >>>                   filter_reviewed=False, filter_true_matches=True)
        >>> review_edges = get_review_edges(cm_list, review_cfg=review_cfg, ibs=ibs)
        >>> print(review_edges)
    """
    import vtool as vt
    from wbia.algo.hots import chip_match

    automatch_kw = REVIEW_CFG_DEFAULTS.copy()
    automatch_kw = ut.update_existing(automatch_kw, review_cfg)
    print('[resorg] get_review_edges(%s)' % (ut.repr2(automatch_kw)))
    print('[resorg] len(cm_list) = %d' % (len(cm_list)))
    qaids_stack = []
    daids_stack = []
    ranks_stack = []
    scores_stack = []

    # For each QueryResult, Extract inspectable candidate matches
    if isinstance(cm_list, dict):
        cm_list = list(cm_list.values())

    if len(cm_list) == 0:
        return ([], [], [], [])

    for cm in cm_list:
        if isinstance(cm, chip_match.ChipMatch):
            daids = cm.get_top_aids(ntop=automatch_kw['ranks_top'])
            scores = cm.get_top_scores(ntop=automatch_kw['ranks_top'])
            ranks = np.arange(len(daids))
            qaids = np.full(daids.shape, cm.qaid, dtype=daids.dtype)
        else:
            (qaids, daids, scores, ranks) = cm.get_match_tbldata(
                ranks_top=automatch_kw['ranks_top'],
                name_scoring=automatch_kw['name_scoring'],
                ibs=ibs,
            )
        qaids_stack.append(qaids)
        daids_stack.append(daids)
        scores_stack.append(scores)
        ranks_stack.append(ranks)

    # Stack them into a giant array
    qaid_arr = np.hstack(qaids_stack)
    daid_arr = np.hstack(daids_stack)
    score_arr = np.hstack(scores_stack)
    rank_arr = np.hstack(ranks_stack)

    # Sort by scores
    sortx = score_arr.argsort()[::-1]
    qaid_arr = qaid_arr[sortx]
    daid_arr = daid_arr[sortx]
    score_arr = score_arr[sortx]
    rank_arr = rank_arr[sortx]

    # IS_REVIEWED DOES NOT WORK
    if automatch_kw['filter_reviewed']:
        _is_reviewed = ibs.get_annot_pair_is_reviewed(
            qaid_arr.tolist(), daid_arr.tolist()
        )
        is_unreviewed = ~np.array(_is_reviewed, dtype=np.bool)
        qaid_arr = qaid_arr.compress(is_unreviewed)
        daid_arr = daid_arr.compress(is_unreviewed)
        score_arr = score_arr.compress(is_unreviewed)
        rank_arr = rank_arr.compress(is_unreviewed)

    # Remove directed edges
    if not automatch_kw['directed']:
        # nodes = np.unique(directed_edges.flatten())
        directed_edges = np.vstack((qaid_arr, daid_arr)).T
        # idx1, idx2 = vt.intersect2d_indices(directed_edges, directed_edges[:, ::-1])

        unique_rowx = vt.find_best_undirected_edge_indexes(directed_edges, score_arr)

        qaid_arr = qaid_arr.take(unique_rowx)
        daid_arr = daid_arr.take(unique_rowx)
        score_arr = score_arr.take(unique_rowx)
        rank_arr = rank_arr.take(unique_rowx)

    # Filter Double Name Matches
    if automatch_kw['filter_duplicate_true_matches']:
        # filter_dup_namepairs
        qnid_arr = ibs.get_annot_nids(qaid_arr)
        dnid_arr = ibs.get_annot_nids(daid_arr)
        if not automatch_kw['directed']:
            directed_name_edges = np.vstack((qnid_arr, dnid_arr)).T
            unique_rowx2 = vt.find_best_undirected_edge_indexes(
                directed_name_edges, score_arr
            )
        else:
            namepair_id_list = np.array(
                vt.compute_unique_data_ids_(list(zip(qnid_arr, dnid_arr)))
            )
            unique_namepair_ids, namepair_groupxs = vt.group_indices(namepair_id_list)
            score_namepair_groups = vt.apply_grouping(score_arr, namepair_groupxs)
            unique_rowx2 = np.array(
                sorted(
                    [
                        groupx[score_group.argmax()]
                        for groupx, score_group in zip(
                            namepair_groupxs, score_namepair_groups
                        )
                    ]
                ),
                dtype=np.int32,
            )
        qaid_arr = qaid_arr.take(unique_rowx2)
        daid_arr = daid_arr.take(unique_rowx2)
        score_arr = score_arr.take(unique_rowx2)
        rank_arr = rank_arr.take(unique_rowx2)

    # Filter all true matches
    if automatch_kw['filter_true_matches']:
        qnid_arr = ibs.get_annot_nids(qaid_arr)
        dnid_arr = ibs.get_annot_nids(daid_arr)
        valid_flags = qnid_arr != dnid_arr
        qaid_arr = qaid_arr.compress(valid_flags)
        daid_arr = daid_arr.compress(valid_flags)
        score_arr = score_arr.compress(valid_flags)
        rank_arr = rank_arr.compress(valid_flags)

    if automatch_kw['filter_photobombs']:
        unique_aids = ut.unique(ut.flatten([qaid_arr, daid_arr]))
        # grouped_aids, unique_nids = ibs.group_annots_by_name(unique_aids)
        invalid_nid_map = get_photobomber_map(ibs, qaid_arr)

        nid2_aids = ut.group_items(unique_aids, ibs.get_annot_nids(unique_aids))

        expanded_aid_map = ut.ddict(set)
        for nid1, other_nids in invalid_nid_map.items():
            for aid1 in nid2_aids[nid1]:
                for nid2 in other_nids:
                    for aid2 in nid2_aids[nid2]:
                        expanded_aid_map[aid1].add(aid2)
                        expanded_aid_map[aid2].add(aid1)

        valid_flags = [
            daid not in expanded_aid_map[qaid] for qaid, daid in zip(qaid_arr, daid_arr)
        ]
        qaid_arr = qaid_arr.compress(valid_flags)
        daid_arr = daid_arr.compress(valid_flags)
        score_arr = score_arr.compress(valid_flags)
        rank_arr = rank_arr.compress(valid_flags)

    review_edges = (qaid_arr, daid_arr, score_arr, rank_arr)
    return review_edges


def make_review_api(ibs, cm_list, review_cfg, qreq_=None):
    """
    Builds columns which are displayable in a ColumnListTableWidget

    CommandLine:
        python -m wbia.gui.id_review_api --test-test_review_widget --show
        python -m wbia.gui.id_review_api --test-make_review_api

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.gui.id_review_api import *  # NOQA
        >>> import wbia
        >>> import wbia.guitool as gt
        >>> from wbia.gui import id_review_api
        >>> cm_list, qreq_ = wbia.main_helpers.testdata_cmlist()
        >>> tblname = 'chipmatch'
        >>> name_scoring = False
        >>> ranks_top = 5
        >>> review_cfg = dict(ranks_top=ranks_top, name_scoring=name_scoring)
        >>> review_api = make_review_api(qreq_.ibs, cm_list, review_cfg, qreq_=qreq_)
        >>> print('review_api = %r' % (review_api,))
    """
    # TODO: Add in timedelta to column info
    if ut.VERBOSE:
        print('[inspect] make_review_api')

    review_edges = get_review_edges(cm_list, ibs=ibs, review_cfg=review_cfg)
    # Get extra info
    (qaids, daids, scores, ranks) = review_edges

    RES_THUMB_TEXT = 'ResThumb'  # NOQA
    QUERY_THUMB_TEXT = 'querythumb'
    MATCH_THUMB_TEXT = 'MatchThumb'

    col_name_list = [
        'result_index',
        'score',
        REVIEWED_STATUS_TEXT,
    ]

    if review_cfg.get('show_chips', True):
        col_name_list += [
            MATCHED_STATUS_TEXT,
            QUERY_THUMB_TEXT,
        ]

    col_name_list += [
        RES_THUMB_TEXT,
        'qaid',
        'aid',
        'rank',
        'timedelta',
        'dnGt',
        'qnGt',
        'tags',
        'qname',
        'name',
    ]

    col_types_dict = dict(
        [
            ('qaid', int),
            ('aid', int),
            ('dnGt', int),
            ('qnGt', int),
            ('timedelta', float),
            # ('review',     'BUTTON'),
            (MATCHED_STATUS_TEXT, str),
            (REVIEWED_STATUS_TEXT, str),
            (QUERY_THUMB_TEXT, 'PIXMAP'),
            (RES_THUMB_TEXT, 'PIXMAP'),
            ('qname', str),
            ('name', str),
            ('score', float),
            ('rank', int),
            ('truth', bool),
            ('opt', int),
            ('result_index', int),
        ]
    )
    timedelta_list = np.array(
        ut.take_column(ibs.get_unflat_annots_timedelta_list(list(zip(qaids, daids))), 0)
    )
    # TODO: make a display role
    # timediff_list = [ut.get_posix_timedelta_str(t, year=True, approx=True) for t in (timedelta_list * 60 * 60)]

    def get_pair_tags(edge):
        aid1, aid2 = edge
        assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
        assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
        am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey([aid1], [aid2])
        tag_text = ibs.get_annotmatch_tag_text(am_rowids)[0]
        if tag_text is None:
            tag_text = ''
        return str(tag_text)

    col_getter_dict = dict(
        [
            ('qaid', np.array(qaids)),
            ('aid', np.array(daids)),
            ('dnGt', ibs.get_annot_num_groundtruth),
            ('qnGt', ibs.get_annot_num_groundtruth),
            ('timedelta', np.array(timedelta_list)),
            # ('review',     lambda rowid: get_buttontup),
            (MATCHED_STATUS_TEXT, partial(get_match_status, ibs)),
            (REVIEWED_STATUS_TEXT, partial(get_reviewed_status, ibs)),
            (QUERY_THUMB_TEXT, ibs.get_annot_chip_thumbtup),
            (RES_THUMB_TEXT, ibs.get_annot_chip_thumbtup),
            ('qname', ibs.get_annot_names),
            ('name', ibs.get_annot_names),
            ('score', np.array(scores)),
            ('rank', np.array(ranks)),
            ('result_index', np.arange(len(ranks))),
            ('tags', get_pair_tags),
            # lambda aid_pair: ibs.get_annotmatch_tag_text(ibs.get_annotmatch_rowid_from_undirected_superkey(ut.ensure_iterable(aid_pair[0]), ut.ensure_iterable(aid_pair[1])))[0]),
            # ('truth',     truths),
            # ('opt',       opts),
        ]
    )

    # default is 100
    col_width_dict = {
        'score': 75,
        REVIEWED_STATUS_TEXT: 75,
        MATCHED_STATUS_TEXT: 75,
        'rank': 42,
        'qaid': 42,
        'aid': 42,
        'result_index': 42,
        'qname': 60,
        'name': 60,
        'dnGt': 50,
        'timedelta': 75,
        'tags': 75,
        'qnGt': 50,
    }

    USE_MATCH_THUMBS = 1
    if USE_MATCH_THUMBS:

        def get_match_thumbtup(
            ibs,
            qaid2_cm,
            qaids,
            daids,
            index,
            qreq_=None,
            thumbsize=(128, 128),
            match_thumbtup_cache={},
        ):
            daid = daids[index]
            qaid = qaids[index]
            cm = qaid2_cm[qaid]
            assert cm.qaid == qaid, 'aids do not aggree'

            OLD = False
            if OLD:
                fpath = ensure_match_img(
                    ibs, cm, daid, qreq_=qreq_, match_thumbtup_cache=match_thumbtup_cache,
                )
                if isinstance(thumbsize, int):
                    thumbsize = (thumbsize, thumbsize)
                thumbtup = (
                    ut.augpath(fpath, 'thumb_%d,%d' % thumbsize),
                    fpath,
                    thumbsize,
                    [],
                    [],
                )
                return thumbtup
            else:
                # Hacky new way of drawing
                fpath, func, func2 = make_ensure_match_img_nosql_func(qreq_, cm, daid)
                # match_thumbdir = ibs.get_match_thumbdir()
                # match_thumb_fname = get_match_thumb_fname(cm, daid, qreq_)
                # fpath = ut.unixjoin(match_thumbdir, match_thumb_fname)
                thumbdat = {
                    'fpath': fpath,
                    'thread_func': func,
                    'main_func': func2,
                    # 'args': (ibs, cm, daid),
                    # 'kwargs': dict(qreq_=qreq_,
                    #               match_thumbtup_cache=match_thumbtup_cache)
                }
                return thumbdat

        col_name_list.insert(col_name_list.index('qaid'), MATCH_THUMB_TEXT)
        col_types_dict[MATCH_THUMB_TEXT] = 'PIXMAP'
        # col_types_dict[MATCH_THUMB_TEXT] = CustomMatchThumbDelegate
        qaid2_cm = {cm.qaid: cm for cm in cm_list}
        get_match_thumbtup_ = partial(
            get_match_thumbtup,
            ibs,
            qaid2_cm,
            qaids,
            daids,
            qreq_=qreq_,
            match_thumbtup_cache={},
        )
        col_getter_dict[MATCH_THUMB_TEXT] = get_match_thumbtup_

    col_bgrole_dict = {
        MATCHED_STATUS_TEXT: partial(get_match_status_bgrole, ibs),
        REVIEWED_STATUS_TEXT: partial(get_reviewed_status_bgrole, ibs),
    }
    # TODO: remove ider dict.
    # it is massively unuseful
    col_ider_dict = {
        MATCHED_STATUS_TEXT: ('qaid', 'aid'),
        REVIEWED_STATUS_TEXT: ('qaid', 'aid'),
        'tags': ('qaid', 'aid'),
        QUERY_THUMB_TEXT: ('qaid'),
        RES_THUMB_TEXT: ('aid'),
        'dnGt': ('aid'),
        'qnGt': ('qaid'),
        'qname': ('qaid'),
        'name': ('aid'),
    }
    col_setter_dict = {'qname': ibs.set_annot_names, 'name': ibs.set_annot_names}
    editable_colnames = ['truth', 'notes', 'qname', 'name', 'opt']

    sortby = 'score'

    def get_thumb_size():
        return ibs.cfg.other_cfg.thumb_size

    col_display_role_func_dict = {
        'timedelta': ut.partial(ut.get_posix_timedelta_str, year=True, approx=2),
    }

    if not review_cfg.get('show_chips', True):
        del col_getter_dict[QUERY_THUMB_TEXT]
        del col_getter_dict[RES_THUMB_TEXT]
        del col_types_dict[RES_THUMB_TEXT]
        del col_types_dict[QUERY_THUMB_TEXT]
        del col_ider_dict[RES_THUMB_TEXT]
        del col_ider_dict[QUERY_THUMB_TEXT]
        # del col_bgrole_dict[RES_THUMB_TEXT]
        # del col_bgrole_dict[QUERY_THUMB_TEXT]

    # Insert info into dict
    review_api = gt.CustomAPI(
        col_name_list=col_name_list,
        col_types_dict=col_types_dict,
        col_getter_dict=col_getter_dict,
        col_bgrole_dict=col_bgrole_dict,
        col_ider_dict=col_ider_dict,
        col_setter_dict=col_setter_dict,
        editable_colnames=editable_colnames,
        col_display_role_func_dict=col_display_role_func_dict,
        sortby=sortby,
        get_thumb_size=get_thumb_size,
        sort_reverse=True,
        col_width_dict=col_width_dict,
    )
    # review_api.review_edges = review_edges
    return review_api


def get_match_status(ibs, aid_pair):
    """ Data role for status column """
    aid1, aid2 = aid_pair
    assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
    assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
    text = ibs.get_match_text(aid1, aid2)
    if text is None:
        raise AssertionError('impossible state id_review_api')
    return text


def get_reviewed_status(ibs, aid_pair):
    """ Data role for status column """
    aid1, aid2 = aid_pair
    assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
    assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)
    # FIXME: use new api
    state = ibs.get_annot_pair_is_reviewed([aid1], [aid2])[0]
    state_to_text = {
        None: 'Unreviewed',
        2: 'Auto-reviewed',
        1: 'User-reviewed',
    }
    default = '??? unknown mode %r' % (state,)
    text = state_to_text.get(state, default)
    return text


def get_match_status_bgrole(ibs, aid_pair):
    """ Background role for status column """
    aid1, aid2 = aid_pair
    truth = ibs.get_match_truth(aid1, aid2)
    # print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=0.35)
    return truth_color


def get_reviewed_status_bgrole(ibs, aid_pair):
    """ Background role for status column """
    aid1, aid2 = aid_pair
    truth = ibs.get_match_truth(aid1, aid2)
    annotmach_reviewed = ibs.get_annot_pair_is_reviewed([aid1], [aid2])[0]
    if annotmach_reviewed == 0 or annotmach_reviewed is None:
        lighten_amount = 0.9
    elif annotmach_reviewed == 2:
        lighten_amount = 0.7
    else:
        lighten_amount = 0.35
    truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=lighten_amount)
    # truth = ibs.get_match_truth(aid1, aid2)
    # print('get status bgrole: %r truth=%r' % (aid_pair, truth))
    # truth_color = vh.get_truth_color(truth, base255=True, lighten_amount=0.35)
    return truth_color


def get_match_thumb_fname(
    cm, daid, qreq_, view_orientation='vertical', draw_matches=True, draw_heatmask=False
):
    """
    CommandLine:
        python -m wbia.gui.id_review_api --exec-get_match_thumb_fname

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.gui.id_review_api import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm('PZ_MTEST')
        >>> thumbsize = (128, 128)
        >>> daid = cm.get_top_aids()[0]
        >>> match_thumb_fname = get_match_thumb_fname(cm, daid, qreq_)
        >>> result = match_thumb_fname
        >>> print(result)
        match_aids=1,1_cfgstr=ubpzwu5k54h6xbnr.jpg
    """
    # Make thumbnail name
    config_hash = ut.hashstr27(qreq_.get_cfgstr())
    qaid = cm.qaid
    args = (
        qaid,
        daid,
        config_hash,
        draw_matches,
        draw_heatmask,
        view_orientation,
    )
    match_thumb_fname = (
        'match_aids=%d,%d_cfgstr=%s_draw=%s_mask=%s_orientation=%s.jpg' % args
    )
    return match_thumb_fname


def ensure_match_img(ibs, cm, daid, qreq_=None, match_thumbtup_cache={}):
    r"""
    CommandLine:
        python -m wbia.gui.id_review_api --test-ensure_match_img --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.gui.id_review_api import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> cm, qreq_ = wbia.testdata_cm()
        >>> daid = cm.get_top_aids()[0]
        >>> match_thumbtup_cache = {}
        >>> # execute function
        >>> match_thumb_fpath_ = ensure_match_img(qreq_.ibs, cm, daid, qreq_,
        >>>                                       match_thumbtup_cache)
        >>> # verify results
        >>> result = str(match_thumb_fpath_)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> ut.startfile(match_thumb_fpath_, quote=True)
    """
    # from os.path import exists
    match_thumbdir = ibs.get_match_thumbdir()
    match_thumb_fname = get_match_thumb_fname(cm, daid, qreq_)
    match_thumb_fpath_ = ut.unixjoin(match_thumbdir, match_thumb_fname)
    # if exists(match_thumb_fpath_):
    #    return match_thumb_fpath_
    if match_thumb_fpath_ in match_thumbtup_cache:
        fpath = match_thumbtup_cache[match_thumb_fpath_]
    else:
        # TODO: just draw the image at the correct thumbnail size
        # TODO: draw without matplotlib?
        # with ut.Timer('render-1'):
        fpath = cm.imwrite_single_annotmatch(
            qreq_,
            daid,
            fpath=match_thumb_fpath_,
            saveax=True,
            fnum=32,
            notitle=True,
            verbose=False,
        )
        # with ut.Timer('render-2'):
        #    img = cm.render_single_annotmatch(qreq_, daid, fnum=32, notitle=True, dpi=30)
        #    cv2.imwrite(match_thumb_fpath_, img)
        #    fpath = match_thumb_fpath_
        # with ut.Timer('render-3'):
        # fpath = match_thumb_fpath_
        # render_config = {
        #    'dpi'              : 60,
        #    'draw_fmatches'    : True,
        #    #'vert'             : view_orientation == 'vertical',
        #    'show_aidstr'      : False,
        #    'show_name'        : False,
        #    'show_exemplar'    : False,
        #    'show_num_gt'      : False,
        #    'show_timedelta'   : False,
        #    'show_name_rank'   : False,
        #    'show_score'       : False,
        #    'show_annot_score' : False,
        #    'show_name_score'  : False,
        #    'draw_lbl'         : False,
        #    'draw_border'      : False,
        # }
        # cm.imwrite_single_annotmatch2(qreq_, daid, fpath, fnum=32, notitle=True, **render_config)
        # print('fpath = %r' % (fpath,))
        match_thumbtup_cache[match_thumb_fpath_] = fpath
    return fpath


def make_ensure_match_img_nosql_func(qreq_, cm, daid):
    r"""
    CommandLine:
        python -m wbia.gui.id_review_api --test-ensure_match_img --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.gui.id_review_api import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> cm, qreq_ = wbia.testdata_cm()
        >>> ibs = qreq_.ibs
        >>> daid = cm.get_top_aids()[0]
        >>> match_thumbtup_cache = {}
        >>> # execute function
        >>> match_thumb_fpath_ = ensure_match_img(qreq_.ibs, cm, daid, qreq_, match_thumbtup_cache)
        >>> # verify results
        >>> result = str(match_thumb_fpath_)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> ut.startfile(match_thumb_fpath_, quote=True)
    """
    # import wbia.viz
    from wbia.viz import viz_matches
    import cv2
    import io
    import wbia.plottool as pt
    import vtool as vt
    import matplotlib as mpl

    if cm.__class__.__name__ == 'PairwiseMatch':
        # HACK DO THIS THE VTOOL WAY
        match = cm
        ibs = qreq_  # VERY HACK
        match_thumbdir = ibs.get_match_thumbdir()
        cfgstr = hash(match.config)  # HACK only works if config is already a hashdict
        match_thumb_fname = 'tmpmatch-%d-%d-%s.jpg' % (
            match.annot1['aid'],
            match.annot2['aid'],
            cfgstr,
        )
        fpath = ut.unixjoin(match_thumbdir, match_thumb_fname)

        def main_thread_load2():
            rchip1, kpts1 = ut.dict_take(match.annot1, ['rchip', 'kpts'])
            rchip2, kpts2 = ut.dict_take(match.annot2, ['rchip', 'kpts'])
            return (match,)

        def nosql_draw2(check_func, match):
            from matplotlib.backends.backend_agg import FigureCanvas

            try:
                from matplotlib.backends.backend_agg import Figure
            except ImportError:
                from matplotlib.figure import Figure

            was_interactive = mpl.is_interactive()
            if was_interactive:
                mpl.interactive(False)
            # fnum = 32
            fig = Figure()
            canvas = FigureCanvas(fig)  # NOQA
            # fig.clf()
            ax = fig.add_subplot(1, 1, 1)
            if check_func is not None and check_func():
                return
            ax, xywh1, xywh2 = match.show(ax=ax)
            if check_func is not None and check_func():
                return
            savekw = {
                # 'dpi' : 60,
                'dpi': 80,
            }
            axes_extents = pt.extract_axes_extents(fig)
            # assert len(axes_extents) == 1, 'more than one axes'
            extent = axes_extents[0]
            with io.BytesIO() as stream:
                # This call takes 23% - 15% of the time depending on settings
                fig.savefig(stream, bbox_inches=extent, **savekw)
                stream.seek(0)
                data = np.fromstring(stream.getvalue(), dtype=np.uint8)
            if check_func is not None and check_func():
                return
            pt.plt.close(fig)
            image = cv2.imdecode(data, 1)
            thumbsize = 221
            max_dsize = (thumbsize, thumbsize)
            dsize, sx, sy = vt.resized_clamped_thumb_dims(vt.get_size(image), max_dsize)
            if check_func is not None and check_func():
                return
            image = vt.resize(image, dsize)
            vt.imwrite(fpath, image)
            if check_func is not None and check_func():
                return
            # fig.savefig(fpath, bbox_inches=extent, **savekw)

        # match_thumbtup_cache[match_thumb_fpath_] = fpath
        return fpath, nosql_draw2, main_thread_load2

    aid1 = cm.qaid
    aid2 = daid

    ibs = qreq_.ibs
    resize_factor = 0.5

    match_thumbdir = ibs.get_match_thumbdir()
    match_thumb_fname = get_match_thumb_fname(cm, daid, qreq_)
    fpath = ut.unixjoin(match_thumbdir, match_thumb_fname)

    def main_thread_load():
        # This gets executed in the main thread and collects data
        # from sql
        rchip1_fpath, rchip2_fpath, kpts1, kpts2 = viz_matches._get_annot_pair_info(
            ibs, aid1, aid2, qreq_, draw_fmatches=True, as_fpath=True
        )
        return rchip1_fpath, rchip2_fpath, kpts1, kpts2

    def nosql_draw(check_func, rchip1_fpath, rchip2_fpath, kpts1, kpts2):
        # This gets executed in the child thread and does drawing async style
        # from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
        # from matplotlib.backends.backend_pdf import Figure
        # from matplotlib.backends.backend_svg import FigureCanvas
        # from matplotlib.backends.backend_svg import Figure
        from matplotlib.backends.backend_agg import FigureCanvas

        try:
            from matplotlib.backends.backend_agg import Figure
        except ImportError:
            from matplotlib.figure import Figure

        kpts1_ = vt.offset_kpts(kpts1, (0, 0), (resize_factor, resize_factor))
        kpts2_ = vt.offset_kpts(kpts2, (0, 0), (resize_factor, resize_factor))

        # from matplotlib.figure import Figure
        if check_func is not None and check_func():
            return

        rchip1 = vt.imread(rchip1_fpath)
        rchip1 = vt.resize_image_by_scale(rchip1, resize_factor)
        if check_func is not None and check_func():
            return
        rchip2 = vt.imread(rchip2_fpath)
        rchip2 = vt.resize_image_by_scale(rchip2, resize_factor)
        if check_func is not None and check_func():
            return

        try:
            idx = cm.daid2_idx[daid]
            fm = cm.fm_list[idx]
            fsv = None if cm.fsv_list is None else cm.fsv_list[idx]
            fs = None if fsv is None else fsv.prod(axis=1)
        except KeyError:
            fm = []
            fs = None
            fsv = None

        maxnum = 200
        if fs is not None and len(fs) > maxnum:
            # HACK TO ONLY SHOW TOP MATCHES
            sortx = fs.argsort()[::-1]
            fm = fm.take(sortx[:maxnum], axis=0)
            fs = fs.take(sortx[:maxnum], axis=0)

        was_interactive = mpl.is_interactive()
        if was_interactive:
            mpl.interactive(False)
        # fnum = 32
        fig = Figure()
        canvas = FigureCanvas(fig)  # NOQA
        # fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        if check_func is not None and check_func():
            return
        # fig = pt.plt.figure(fnum)
        # H1 = np.eye(3)
        # H2 = np.eye(3)
        # H1[0, 0] = .5
        # H1[1, 1] = .5
        # H2[0, 0] = .5
        # H2[1, 1] = .5
        ax, xywh1, xywh2 = pt.show_chipmatch2(
            rchip1, rchip2, kpts1_, kpts2_, fm, fs=fs, colorbar_=False, ax=ax
        )
        if check_func is not None and check_func():
            return
        savekw = {
            # 'dpi' : 60,
            'dpi': 80,
        }
        axes_extents = pt.extract_axes_extents(fig)
        # assert len(axes_extents) == 1, 'more than one axes'
        extent = axes_extents[0]
        with io.BytesIO() as stream:
            # This call takes 23% - 15% of the time depending on settings
            fig.savefig(stream, bbox_inches=extent, **savekw)
            stream.seek(0)
            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        if check_func is not None and check_func():
            return
        pt.plt.close(fig)
        image = cv2.imdecode(data, 1)
        thumbsize = 221
        max_dsize = (thumbsize, thumbsize)
        dsize, sx, sy = vt.resized_clamped_thumb_dims(vt.get_size(image), max_dsize)
        if check_func is not None and check_func():
            return
        image = vt.resize(image, dsize)
        vt.imwrite(fpath, image)
        if check_func is not None and check_func():
            return
        # fig.savefig(fpath, bbox_inches=extent, **savekw)

    # match_thumbtup_cache[match_thumb_fpath_] = fpath
    return fpath, nosql_draw, main_thread_load


def get_photobomber_map(ibs, aids, aid_to_nid=None):
    """
    Builds map of which names that photobomb other names.

    python -m wbia.gui.id_review_api --test-test_review_widget --show --db PZ_MTEST -a default:qindex=0

    >>> import wbia
    >>> dbdir = ut.truepath('~/lev/media/danger/GGR/GGR-IBEIS')
    >>> ibs = wbia.opendb(dbdir='/home/joncrall/lev/media/danger/GGR/GGR-IBEIS')
    >>> filter_kw = {
    >>>     'multiple': False,
    >>>     'minqual': 'good',
    >>>     'is_known': True,
    >>>     'min_pername': 2,
    >>>     'view': ['right'],
    >>> }
    >>> aids = ibs.filter_annots_general(ibs.get_valid_aids(), filter_kw=filter_kw)
    """
    ams_list = ibs.get_annotmatch_rowids_from_aid(aids)
    flags_list = ibs.unflat_map(
        ut.partial(ibs.get_annotmatch_prop, 'Photobomb'), ams_list
    )
    pb_ams = ut.zipcompress(ams_list, flags_list)
    has_pb_ams = [len(ams) > 0 for ams in pb_ams]
    pb_ams_ = ut.compress(pb_ams, has_pb_ams)
    # aids_ = ut.compress(aids, has_pb_ams)
    pb_ams_flat = ut.flatten(pb_ams_)

    pb_aids1_ = ibs.get_annotmatch_aid1(pb_ams_flat)
    pb_aids2_ = ibs.get_annotmatch_aid2(pb_ams_flat)

    pb_aid_pairs_ = list(zip(pb_aids1_, pb_aids2_))
    if aid_to_nid is None:
        pb_nid_pairs_ = ibs.unflat_map(ibs.get_annot_nids, pb_aid_pairs_)
    else:
        pb_nid_pairs_ = ibs.unflat_map(ut.partial(ut.take, aid_to_nid), pb_aid_pairs_)

    # invalid_aid_map = ut.ddict(set)
    # for aid1, aid2 in pb_aid_pairs_:
    #    if aid1 != aid2:
    #        invalid_aid_map[aid1].add(aid2)
    #        invalid_aid_map[aid2].add(aid1)

    invalid_nid_map = ut.ddict(set)
    for nid1, nid2 in pb_nid_pairs_:
        if nid1 != nid2:
            invalid_nid_map[nid1].add(nid2)
            invalid_nid_map[nid2].add(nid1)

    return invalid_nid_map
