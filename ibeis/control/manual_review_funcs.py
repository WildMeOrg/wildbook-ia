# -*- coding: utf-8 -*-
"""
python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'ibeis.templates.template_generator')"
sh Tgen.sh --key review --invert --Tcfg with_getters=True with_setters=False --modfname manual_review_funcs

# TODO: Fix this name it is too special case
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
from six.moves import range, zip, map  # NOQA
#import numpy as np
#import vtool as vt
import numpy as np
from ibeis import constants as const
from ibeis.control import accessor_decors, controller_inject  # NOQA
import utool as ut
from ibeis.control.controller_inject import make_ibs_register_decorator
print, rrr, profile = ut.inject2(__name__)


VERBOSE_SQL    = ut.get_argflag(('--print-sql', '--verbose-sql', '--verb-sql', '--verbsql'))
CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)

REVIEW_ROWID     = 'review_rowid'
REVIEW_AID1      = 'annot_1_rowid'
REVIEW_AID2      = 'annot_2_rowid'
REVIEW_COUNT     = 'review_count'
REVIEW_DECISION  = 'review_decision'
REVIEW_TIMESTAMP = 'review_time_posix'
REVIEW_IDENTITY  = 'review_identity'
REVIEW_TAGS      = 'review_tags'


@register_ibs_method
@accessor_decors.ider
@register_api('/api/review/', methods=['GET'])
def _get_all_review_rowids(ibs):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    #all_known_review_rowids = ibs._get_all_known_lblannot_rowids(const.review_KEY)
    all_known_review_rowids = ibs.staging.get_all_rowids(const.REVIEW_TABLE)
    return all_known_review_rowids


@register_ibs_method
def get_review_rowid_from_superkey(ibs, aid_1_list, aid_2_list, count_list, eager=False, nInput=None):
    """ Returns review_rowid_list

    Args:
        superkey lists: review_rowid_list, aid_list

    Returns:
        review_rowid_list
    """
    colnames = (REVIEW_ROWID,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid_1_list, aid_2_list, count_list)
    andwhere_colnames = [REVIEW_AID1, REVIEW_AID2, REVIEW_COUNT]
    review_rowid_list = list(ibs.staging.get_where_eq(
        const.REVIEW_TABLE, colnames, params_iter, andwhere_colnames, eager=eager, nInput=nInput))
    return review_rowid_list


@register_ibs_method
@accessor_decors.adder
@register_api('/api/review/', methods=['POST'])
def add_review(ibs, aid_1_list, aid_2_list, decision_list, identity_list=None,
               tags_list=None):
    r"""
    Adds a list of reviews.

    Returns:
        list: review_id_list - review rowids

    RESTful:
        Method: POST
        URL:    /api/review/

    CommandLine:
        python -m ibeis.control.manual_review_funcs --test-add_review
    """
    # Get current review counts from database
    diff_list =  - np.array(aid_2_list)
    assert np.all(diff_list != 0), 'Cannot add a review state between an aid and itself'

    # Order aid_1_list and aid_2_list pairs so that aid_1_list is always lower
    pairs_list = zip(aid_1_list, aid_2_list)
    sorted_pairs_list = list(map(sorted, pairs_list))
    aid_1_list = [ pair[0] for pair in sorted_pairs_list ]
    aid_2_list = [ pair[1] for pair in sorted_pairs_list ]

    current_counts_list = ibs.get_review_counts_from_tuple(aid_1_list, aid_2_list)
    current_counts_dict = {
        (aid1, aid2) : 0 if count_list_ is None else max(max(count_list_), len(count_list_))
        for aid1, aid2, count_list_ in zip(aid_1_list, aid_2_list, current_counts_list)
    }
    count_list = []
    for aid1, aid2 in zip(aid_1_list, aid_2_list):
        key = (aid1, aid2)
        current_counts_dict[key] += 1
        count_list.append(current_counts_dict[key])

    if identity_list is None:
        # identity_list = [ut.get_computer_name()] * len(aid_1_list)
        identity_list = [''] * len(aid_1_list)
    if tags_list is None:
        tag_str_list = [''] * len(aid_1_list)
    else:
        tag_str_list = [';'.join(map(str, tag_list)) for tag_list in tags_list]

    superkey_paramx = (0, 1, 2, )
    # TODO Allow for better ensure=False without using partial
    # Just autogenerate these functions
    colnames = [REVIEW_AID1, REVIEW_AID2, REVIEW_COUNT, REVIEW_DECISION, REVIEW_IDENTITY, REVIEW_TAGS]
    params_iter = list(zip(aid_1_list, aid_2_list, count_list, decision_list, identity_list, tag_str_list))
    review_rowid_list = ibs.staging.add_cleanly(const.REVIEW_TABLE, colnames, params_iter,
                                                ibs.get_review_rowid_from_superkey, superkey_paramx)
    return review_rowid_list


@register_ibs_method
@accessor_decors.deleter
#@cache_invalidator(const.REVIEW_TABLE)
@register_api('/api/review/', methods=['DELETE'])
def delete_review(ibs, review_rowid_list):
    r"""
    deletes reviews from the database

    RESTful:
        Method: DELETE
        URL:    /api/review/
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d reviews' % len(review_rowid_list))
    ibs.staging.delete_rowids(const.REVIEW_TABLE, review_rowid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/rowids/tuple/', methods=['GET'], __api_plural_check__=False)
def get_review_rowids_from_aid_tuple(ibs, aid_1_list, aid_2_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_rowid_list - review rowid list of lists

    RESTful:
        Method: GET
        URL:    /api/review/rowid/tuple/
    """
    colnames = (REVIEW_ROWID,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid_1_list, aid_2_list)
    andwhere_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_rowids_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE, colnames, params_iter, andwhere_colnames,
        eager=eager, nInput=nInput, __reject_multiple_records__=False)
    return review_rowids_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/count/', methods=['GET'], __api_plural_check__=False)
def get_review_count(ibs, review_rowid_list):
    review_count_list = ibs.staging.get(const.REVIEW_TABLE, (REVIEW_COUNT,), review_rowid_list)
    return review_count_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/counts/tuple/', methods=['GET'], __api_plural_check__=False)
def get_review_counts_from_tuple(ibs, aid_1_list, aid_2_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_counts_list - review counts

    RESTful:
        Method: GET
        URL:    /api/review/counts/tuple/
    """
    colnames = (REVIEW_COUNT,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid_1_list, aid_2_list)
    andwhere_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_counts_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE, colnames, params_iter, andwhere_colnames,
        eager=eager, nInput=nInput, __reject_multiple_records__=False)
    return review_counts_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/decision/', methods=['GET'], __api_plural_check__=False)
def get_review_decision(ibs, review_rowid_list):
    review_decision_list = ibs.staging.get(const.REVIEW_TABLE, (REVIEW_DECISION,), review_rowid_list)
    return review_decision_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/decision/str/', methods=['GET'], __api_plural_check__=False)
def get_review_decision_str(ibs, review_rowid_list):
    review_decision_list = ibs.get_review_decision(review_rowid_list)
    review_decision_str_list = [
        const.REVIEW_INT_TO_TEXT.get(review_decision)
        for review_decision in review_decision_list
    ]
    return review_decision_str_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/decisions/only/', methods=['GET'], __api_plural_check__=False)
def get_review_decisions_from_only(ibs, aid_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_tuple_decisions_list - review decisions

    RESTful:
        Method: GET
        URL:    /api/review/identities/only/
    """
    colnames = (REVIEW_AID1, REVIEW_AID2, REVIEW_DECISION,)
    params_iter = [ (aid, ) for aid in aid_list ]
    where_clause = '%s=?' % (REVIEW_AID1)
    review_tuple_decisions_list = ibs.staging.get_where(const.REVIEW_TABLE, colnames,
                                                        params_iter, where_clause,
                                                        __reject_multiple_records__=False)
    return review_tuple_decisions_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/decisions/single/', methods=['GET'], __api_plural_check__=False)
def get_review_decisions_from_single(ibs, aid_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_tuple_decisions_list - review decisions

    RESTful:
        Method: GET
        URL:    /api/review/identities/single/
    """
    colnames = (REVIEW_AID1, REVIEW_AID2, REVIEW_DECISION,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid_list, aid_list, )
    andwhere_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_tuple_decisions_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE, colnames, params_iter, andwhere_colnames,
        eager=eager, nInput=nInput, op='OR', __reject_multiple_records__=False)
    return review_tuple_decisions_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/decisions/tuple/', methods=['GET'], __api_plural_check__=False)
def get_review_decisions_from_tuple(ibs, aid_1_list, aid_2_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_decisions_list - review decisions

    RESTful:
        Method: GET
        URL:    /api/review/identities/tuple/
    """
    colnames = (REVIEW_DECISION,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid_1_list, aid_2_list)
    andwhere_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_decisions_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE, colnames, params_iter, andwhere_colnames,
        eager=eager, nInput=nInput, __reject_multiple_records__=False)
    return review_decisions_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/decisions/str/tuple/', methods=['GET'], __api_plural_check__=False)
def get_review_decisions_str_from_tuple(ibs, aid_1_list, aid_2_list, **kwargs):
    r"""
    Returns:
        list_ (list): review_decisions_list - review decisions

    RESTful:
        Method: GET
        URL:    /api/review/identities/str/tuple/
    """
    review_decisions_list = ibs.get_review_decisions_from_tuple(aid_1_list, aid_2_list, **kwargs)
    review_decision_str_list = [
        [
            const.REVIEW_INT_TO_TEXT.get(review_decision)
            for review_decision in review_decision_list
        ]
        for review_decision_list in review_decisions_list
    ]
    return review_decision_str_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/identity/', methods=['GET'], __api_plural_check__=False)
def get_review_identity(ibs, review_rowid_list):
    review_identity_list = ibs.staging.get(const.REVIEW_TABLE, (REVIEW_IDENTITY,), review_rowid_list)
    return review_identity_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/identities/tuple/', methods=['GET'], __api_plural_check__=False)
def get_review_identities_from_tuple(ibs, aid_1_list, aid_2_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_identities_list - review identities

    RESTful:
        Method: GET
        URL:    /api/review/identities/tuple/
    """
    colnames = (REVIEW_IDENTITY,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid_1_list, aid_2_list)
    andwhere_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_identities_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE, colnames, params_iter, andwhere_colnames,
        eager=eager, nInput=nInput, __reject_multiple_records__=False)
    return review_identities_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/time/posix/', methods=['GET'], __api_plural_check__=False)
def get_review_posix_time(ibs, review_rowid_list):
    review_posix_time_list = ibs.staging.get(const.REVIEW_TABLE, (REVIEW_TIMESTAMP,), review_rowid_list)
    return review_posix_time_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/times/posix/tuple/', methods=['GET'], __api_plural_check__=False)
def get_review_posix_times_from_tuple(ibs, aid_1_list, aid_2_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): identity_list - review posix times

    RESTful:
        Method: GET
        URL:    /api/review/time/posix/tuple/
    """
    colnames = (REVIEW_TIMESTAMP,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid_1_list, aid_2_list)
    andwhere_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_posix_times_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE, colnames, params_iter, andwhere_colnames,
        eager=eager, nInput=nInput, __reject_multiple_records__=False)
    return review_posix_times_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/tags/', methods=['GET'], __api_plural_check__=False)
def get_review_tags(ibs, review_rowid_list):
    review_tag_str_list = ibs.staging.get(const.REVIEW_TABLE, (REVIEW_TAGS,), review_rowid_list)
    review_tags_list = [
        None if review_tag_str is None or len(review_tag_str) == 0 else review_tag_str.split(';')
        for review_tag_str in review_tag_str_list
    ]
    return review_tags_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/review/tags/tuple/', methods=['GET'], __api_plural_check__=False)
def get_review_tags_from_tuple(ibs, aid_1_list, aid_2_list, eager=True, nInput=None):
    r"""
    Returns:
        list_ (list): review_tags_list - review tags (list of strings)

    RESTful:
        Method: GET
        URL:    /api/review/tags/tuple/
    """
    colnames = (REVIEW_TAGS,)
    # FIXME: col_rowid is not correct
    params_iter = zip(aid_1_list, aid_2_list)
    andwhere_colnames = [REVIEW_AID1, REVIEW_AID2]
    review_tag_strs_list = ibs.staging.get_where_eq(
        const.REVIEW_TABLE, colnames, params_iter, andwhere_colnames,
        eager=eager, nInput=nInput, __reject_multiple_records__=False)
    review_tags_list = [
        []
        if review_tag_str_list is None else
        [
            None if review_tag_str is None or len(review_tag_str) == 0 else review_tag_str.split(';')
            for review_tag_str in review_tag_str_list
        ]
        for review_tag_str_list in review_tag_strs_list
    ]
    return review_tags_list


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.control.manual_review_funcs
        python -m ibeis.control.manual_review_funcs --allexamples
        python -m ibeis.control.manual_review_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
