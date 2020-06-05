# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utool as ut
import vtool.keypoint as ktool
import wbia.plottool.draw_func2 as df2
from six.moves import zip, map
from wbia.plottool import plot_helpers as ph
from wbia.other import ibsfuncs
from wbia.control.accessor_decors import getter, getter_vector_output

(print, rrr, profile) = ut.inject2(__name__)


NO_LBL_OVERRIDE = ut.get_argval('--no-lbl-override', type_=bool, default=None)


FNUMS = dict(image=1, chip=2, res=3, inspect=4, special=5, name=6)

IN_IMAGE_OVERRIDE = ut.get_argval('--in-image-override', type_=bool, default=None)
SHOW_QUERY_OVERRIDE = ut.get_argval('--show-query-override', type_=bool, default=None)
NO_LBL_OVERRIDE = ut.get_argval('--no-lbl-override', type_=bool, default=None)

SIFT_OR_VECFIELD = ph.SIFT_OR_VECFIELD


def register_FNUMS(FNUMS_):
    # DEPREICATE
    global FNUMS
    FNUMS = FNUMS_


draw = ph.draw
get_square_row_cols = ph.get_square_row_cols
get_ibsdat = ph.get_plotdat
set_ibsdat = ph.set_plotdat


@getter_vector_output
def get_annot_kpts_in_imgspace(ibs, aid_list, config2_=None, ensure=True):
    """ Transforms keypoints so they are plotable in imagespace """
    bbox_list = ibs.get_annot_bboxes(aid_list)
    theta_list = ibs.get_annot_thetas(aid_list)
    try:
        chipsz_list = ibs.get_annot_chip_sizes(aid_list, ensure=ensure)
    except AssertionError as ex:
        ut.printex(ex, '[!ibs.get_annot_kpts_in_imgspace]')
        print('[!ibs.get_annot_kpts_in_imgspace] aid_list = %r' % (aid_list,))
        raise
    kpts_list = ibs.get_annot_kpts(aid_list, ensure=ensure, config2_=config2_)
    imgkpts_list = [
        ktool.transform_kpts_to_imgspace(kpts, bbox, theta, chipsz)
        for kpts, bbox, theta, chipsz in zip(
            kpts_list, bbox_list, theta_list, chipsz_list
        )
    ]
    return imgkpts_list


@getter_vector_output
def get_chips(ibs, aid_list, in_image=False, config2_=None, as_fpath=False):
    # print('config2_ = %r' % (config2_,))
    if as_fpath:
        if in_image:
            return ibs.get_annot_image_paths(aid_list)
        else:
            return ibs.get_annot_chip_fpath(aid_list, config2_=config2_)
    else:
        if in_image:
            return ibs.get_annot_images(aid_list)
        else:
            return ibs.get_annot_chips(aid_list, config2_=config2_)


@getter_vector_output
def get_kpts(
    ibs,
    aid_list,
    in_image=False,
    config2_=None,
    ensure=True,
    kpts_subset=None,
    kpts=None,
):
    if kpts is not None:
        return [kpts]
    if in_image:
        kpts_list = get_annot_kpts_in_imgspace(
            ibs, aid_list, ensure=ensure, config2_=config2_
        )
    else:
        kpts_list = ibs.get_annot_kpts(aid_list, ensure=ensure, config2_=config2_)
    if kpts_subset is not None:
        kpts_list = [
            ut.spaced_items(kpts_, kpts_subset, trunc=True) for kpts_ in kpts_list
        ]
    return kpts_list


@getter_vector_output
def get_bboxes(ibs, aid_list, offset_list=None):
    bbox_list = ibs.get_annot_bboxes(aid_list)
    if offset_list is not None:
        assert len(offset_list) == len(bbox_list)
        # convert (ofx, ofy) offsets to (ofx, ofy, 0, 0) numpy arrays
        np_offsts = [np.array(list(offst) + [0, 0]) for offst in offset_list]
        # add offsets to (x, y, w, h) bounding boxes
        bbox_list = [bbox + offst for bbox, offst in zip(bbox_list, np_offsts)]
    return bbox_list


def get_aidstrs(aid_list, **kwargs):
    if ut.isiterable(aid_list):
        return [ibsfuncs.aidstr(aid, **kwargs) for aid in aid_list]
    else:
        return ibsfuncs.aidstr(aid_list, **kwargs)


def get_nidstrs(nid_list, **kwargs):
    if ut.isiterable(nid_list):
        return ['nid%d' for nid in nid_list]
    else:
        return 'nid%d' % nid_list


def get_vsstr(qaid, aid):
    return 'qaid%d-vs-aid%d' % (qaid, aid)


def get_bbox_centers(bbox_list):
    center_pts = [((x + w / 2), (y + h / 2)) for (x, y, w, h) in bbox_list]
    center_pts = np.array(center_pts)
    return center_pts


def is_unknown(ibs, nid_list):
    # this func seems unused
    return [not isinstance(nid, ut.VALID_INT_TYPES) and len(nid) == 0 for nid in nid_list]


def get_truth_color(truth, base255=False, lighten_amount=None):
    import wbia.constants as const

    truth_colors = {
        const.EVIDENCE_DECISION.NEGATIVE: df2.FALSE_RED,
        const.EVIDENCE_DECISION.POSITIVE: df2.TRUE_BLUE,
        const.EVIDENCE_DECISION.INCOMPARABLE: df2.YELLOW,
        const.EVIDENCE_DECISION.UNKNOWN: df2.UNKNOWN_PURP,
        const.EVIDENCE_DECISION.UNREVIEWED: df2.UNKNOWN_PURP,
    }
    color = truth_colors[truth]
    if lighten_amount is not None:
        # print('color = %r, lighten_amount=%r' % (color, lighten_amount))
        color = df2.lighten_rgb(color, lighten_amount)
        # print('color = %r' % (color))
    if base255:
        color = df2.to_base255(color)
    return color


def get_timedelta_str(ibs, aid1, aid2):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid1 (int):  annotation id
        aid2 (int):  annotation id

    Returns:
        str: timedelta_str

    CommandLine:
        python -m wbia.viz.viz_helpers --test-get_timedelta_str

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.viz.viz_helpers import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> aid1, aid2 = 1, 8
        >>> timedelta_str = get_timedelta_str(ibs, aid1, aid2)
        >>> result = str(timedelta_str)
        >>> print(result)
        td(2 hours 28 minutes 22 seconds)

        td(+2:28:22)
        td(02:28:22)
    """
    gid1, gid2 = ibs.get_annot_gids([aid1, aid2])
    unixtime1, unixtime2 = ibs.get_image_unixtime([gid1, gid2])
    if -1 in [unixtime1, unixtime2]:
        timedelta_str_ = 'NA'
    else:
        unixtime_diff = unixtime2 - unixtime1
        # timedelta_str_ = ut.get_posix_timedelta_str(unixtime_diff)
        timedelta_str_ = ut.get_unix_timedelta_str(unixtime_diff)
    # timedelta_str = 'timedelta(%s)' % (timedelta_str_)
    timedelta_str = 'td(%s)' % (timedelta_str_)
    return timedelta_str


def get_annot_texts(ibs, aid_list, **kwargs):
    """ Add each type of text_list to the strings list

    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: annotation_text_list

    CommandLine:
        python -m wbia.viz.viz_helpers --test-get_annot_texts

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.viz.viz_helpers import *  # NOQA
        >>> import wbia
        >>> import collections
        >>> ibs = wbia.opendb('testdb1')
        >>> # Default all kwargs to true
        >>> class KwargsProxy(object):
        ...    def get(self, a, b):
        ...        return True
        >>> kwargs_proxy = KwargsProxy()
        >>> aid_list = ibs.get_valid_aids()[::3]
        >>> # execute function
        >>> annotation_text_list = get_annot_texts(ibs, aid_list, kwargs_proxy=kwargs_proxy)
        >>> # verify results
        >>> result = ut.repr2(annotation_text_list, nl=1)
        >>> print(result)
        [
            'aid1, gname=easy1.JPG, name=____, nid=-1, , nGt=0, quality=UNKNOWN, view=left',
            'aid4, gname=hard1.JPG, name=____, nid=-4, , nGt=0, quality=UNKNOWN, view=left',
            'aid7, gname=jeff.png, name=jeff, nid=3, EX, nGt=0, quality=UNKNOWN, view=unknown',
            'aid10, gname=occl2.JPG, name=occl, nid=5, EX, nGt=0, quality=UNKNOWN, view=left',
            'aid13, gname=zebra.jpg, name=zebra, nid=7, EX, nGt=0, quality=UNKNOWN, view=unknown',
        ]
    """
    # HACK FOR TEST
    if 'kwargs_proxy' in kwargs:
        kwargs = kwargs['kwargs_proxy']
    try:
        ibsfuncs.assert_valid_aids(ibs, aid_list)
        assert ut.isiterable(aid_list), 'input must be iterable'
        assert all(
            [isinstance(aid, ut.VALID_INT_TYPES) for aid in aid_list]
        ), 'invalid input'
    except AssertionError as ex:
        ut.printex(ex, 'invalid input', 'viz', key_list=['aid_list'])
        raise
    texts_list = []  # list of lists of texts
    if kwargs.get('show_aidstr', True):
        aidstr_list = get_aidstrs(aid_list)
        texts_list.append(aidstr_list)
    if kwargs.get('show_gname', False):
        gname_list = ibs.get_annot_image_names(aid_list)
        texts_list.append(['gname=%s' % gname for gname in gname_list])
    if kwargs.get('show_name', True):
        name_list = ibs.get_annot_names(aid_list)
        texts_list.append(['name=%s' % name for name in name_list])
    if kwargs.get('show_nid', False):
        nid_list = ibs.get_annot_name_rowids(aid_list)
        texts_list.append(['nid=%d' % nid for nid in nid_list])
    if kwargs.get('show_exemplar', True):
        flag_list = ibs.get_annot_exemplar_flags(aid_list)
        texts_list.append(['EX' if flag else '' for flag in flag_list])
    if kwargs.get('show_num_gt', True):
        # FIXME: This should be num_groundtruth with respect to the currently
        # allowed annotations
        nGt_list = ibs.get_annot_num_groundtruth(aid_list)
        texts_list.append(['nGt=%r' % nGt for nGt in nGt_list])
    if kwargs.get('show_quality_text', False):
        qualtext_list = ibs.get_annot_quality_texts(aid_list)
        texts_list.append(list(map(lambda text: 'quality=%s' % text, qualtext_list)))
    if kwargs.get('show_viewcode', False):
        # FIXME: This should be num_groundtruth with respect to the currently
        # allowed annotations
        viewcode_list = ibs.get_annot_viewpoint_code(aid_list)
        texts_list.append(list(map(lambda text: 'view=%s' % text, viewcode_list)))
    # zip them up to get a tuple for each chip and join the fields
    if len(texts_list) > 0:
        annotation_text_list = [', '.join(tup) for tup in zip(*texts_list)]
    else:
        # no texts were specified return empty string for each input
        annotation_text_list = [''] * len(aid_list)
    return annotation_text_list


@getter
def get_image_titles(ibs, gid_list):
    gname_list = ibs.get_image_gnames(gid_list)
    title_list = [
        'gname=%r, gid=%r ' % (str(gname), gid)
        for gid, gname in zip(gid_list, gname_list)
    ]
    return title_list


def get_annot_text(ibs, aid_list, draw_lbls):
    if draw_lbls:
        text_list = ibs.get_annot_names(aid_list)
    else:
        text_list = ut.alloc_nones(len(aid_list))
    return text_list


def get_query_text(ibs, cm, aid2, truth, **kwargs):
    """
    returns title based on the query chip and result

    Args:
        ibs (IBEISController):  wbia controller object
        cm (ChipMatch):  object of feature correspondences and scores
        aid2 (int):  annotation id
        truth (int): 0, 1, 2

    Kwargs:
        qaid, score, rawscore, aid2_raw_rank, show_truth, name_score,
        name_rank, show_name_score, show_name_rank, show_timedelta

    Returns:
        str: query_text

    CommandLine:
        python -m wbia.viz.viz_helpers --exec-get_query_text

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_helpers import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm()
        >>> aid2 = cm.get_top_aids()[0]
        >>> truth = 1
        >>> query_text = get_query_text(ibs, cm, aid2, truth)
        >>> result = ('query_text = %s' % (str(query_text),))
        >>> print(result)
    """
    text_list = []
    if cm is not None:
        qaid = cm.qaid
        score = cm.get_annot_scores([aid2])[0]
        rawscore = cm.get_annot_scores([aid2])[0]
        aid2_raw_rank = cm.get_annot_ranks([aid2])[0]
    else:
        qaid = kwargs.get('qaid', None)
        score = kwargs.get('score', None)
        rawscore = kwargs.get('rawscore', None)
        aid2_raw_rank = kwargs.get('aid2_raw_rank', None)
    if kwargs.get('show_truth', False):
        truth_str = '*%s*' % ibs.const.EVIDENCE_DECISION.INT_TO_NICE.get(truth, None)
        text_list.append(truth_str)
    if kwargs.get('show_rank', aid2_raw_rank is not None or cm is not None):
        try:
            # aid2_raw_rank = cm.get_annot_ranks([aid2])[0]
            aid2_rank = aid2_raw_rank + 1 if aid2_raw_rank is not None else None
            rank_str = 'rank=%s' % str(aid2_rank)
        except Exception as ex:
            ut.printex(ex)
            # ut.embed()
            raise
        text_list.append(rank_str)
    if kwargs.get('show_rawscore', rawscore is not None or cm is not None):
        rawscore_str = 'rawscore=' + ut.num_fmt(rawscore)
        if len(text_list) > 0:
            rawscore_str = '\n' + rawscore_str
        text_list.append(rawscore_str)
    if kwargs.get('show_score', score is not None or cm is not None):
        score_str = 'score=' + ut.num_fmt(score)
        if len(text_list) > 0:
            score_str = '\n' + score_str
        text_list.append(score_str)
    name_score = kwargs.get('name_score', None)
    name_rank = kwargs.get('name_rank', None)
    if kwargs.get('show_name_score', True):
        if name_score is not None:
            text_list.append('name_score=' + ut.num_fmt(name_score))
    if kwargs.get('show_name_rank', True):
        if name_rank is not None:
            # Make display one based
            text_list.append('name_rank=#%s' % (str(name_rank + 1),))
    # with ut.embed_on_exception_context:
    if kwargs.get('show_timedelta', True):
        assert qaid is not None, 'qaid cannot be None'
        # TODO: fixme
        if isinstance(aid2, list):
            aid2_ = aid2[0]
        else:
            aid2_ = aid2
        timedelta_str = '\n' + get_timedelta_str(ibs, qaid, aid2_)
        text_list.append(timedelta_str)
    query_text = ', '.join(text_list)
    return query_text


# ==========================#
#  --- TESTING FUNCS ---   #
# ==========================#


def show_keypoint_gradient_orientations(
    ibs, aid, fx, fnum=None, pnum=None, config2_=None
):
    # Draw the gradient vectors of a patch overlaying the keypoint
    if fnum is None:
        fnum = df2.next_fnum()
    rchip = ibs.get_annot_chips(aid, config2_=config2_)
    kp = ibs.get_annot_kpts(aid, config2_=config2_)[fx]
    sift = ibs.get_annot_vecs(aid, config2_=config2_)[fx]
    fig = df2.draw_keypoint_gradient_orientations(
        rchip, kp, sift=sift, mode='vec', fnum=fnum, pnum=pnum
    )
    fig.canvas.draw()
    fig.show()
    df2.set_title('Gradient orientation\n %s, fx=%d' % (get_aidstrs(aid), fx))


def kp_info(kp):
    kpts = np.array([kp])
    xy_str = ktool.get_xy_strs(kpts)[0]
    shape_str = ktool.get_shape_strs(kpts)[0]
    ori_ = ktool.get_oris(kpts)[0]
    ori_str = 'ori=%.2f' % ori_
    scale = ktool.get_scales(kpts)[0]
    return xy_str, shape_str, scale, ori_str


# ----


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.viz_helpers
        python -m wbia.viz.viz_helpers --allexamples
        python -m wbia.viz.viz_helpers --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
