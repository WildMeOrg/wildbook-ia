# -*- coding: utf-8 -*-
"""
Autogenerated IBEISController functions

TemplateInfo:
    autogen_time = 13:31:28 2015/04/28
    autogen_key = annotgroup

ToRegenerate:
    python -m wbia.templates.template_generator --key annotgroup --Tcfg with_web_api=True with_api_cache=False with_deleters=True no_extern_deleters=True --diff
    python -m wbia.templates.template_generator --key annotgroup --Tcfg with_web_api=True with_api_cache=False with_deleters=True no_extern_deleters=True --write
"""
import logging

import utool as ut

from wbia import constants as const
from wbia.control import accessor_decors  # NOQA
from wbia.control import controller_inject

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger('wbia')

# Create dectorator to inject functions in this module into the IBEISController
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


register_api = controller_inject.get_wbia_flask_api(__name__)


def testdata_ibs(defaultdb='testdb1'):
    import wbia

    ibs = wbia.opendb(defaultdb=defaultdb)
    config2_ = None  # qreq_.qparams
    return ibs, config2_


# AUTOGENED CONSTANTS:
GAR_ROWID = 'gar_rowid'
ANNOTGROUP_NOTE = 'annotgroup_note'
ANNOTGROUP_ROWID = 'annotgroup_rowid'
ANNOTGROUP_TEXT = 'annotgroup_text'
ANNOTGROUP_UUID = 'annotgroup_uuid'
ANNOT_ROWID = 'annot_rowid'


@register_ibs_method
# @register_api('/api/annotgroup/', methods=['GET'])
def _get_all_annotgroup_rowids(ibs):
    """all_annotgroup_rowids <- annotgroup.get_all_rowids()

    Returns:
        list_ (list): unfiltered annotgroup_rowids

    TemplateInfo:
        Tider_all_rowids
        tbl = annotgroup

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_annotgroup_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> ibs._get_all_annotgroup_rowids()
    """
    all_annotgroup_rowids = ibs.db.get_all_rowids(const.ANNOTGROUP_TABLE)
    return all_annotgroup_rowids


@register_ibs_method
# @register_api('/api/annotgroup/', methods=['POST'])
def add_annotgroup(ibs, annotgroup_uuid_list, annotgroup_text_list, annotgroup_note_list):
    """
    Returns:
        returns annotgroup_rowid_list of added (or already existing annotgroups)

    TemplateInfo:
        Tadder_native
        tbl = annotgroup
    """
    # WORK IN PROGRESS
    colnames = (ANNOTGROUP_UUID, ANNOTGROUP_TEXT, ANNOTGROUP_NOTE)

    params_iter = (
        (annotgroup_uuid, annotgroup_text, annotgroup_note)
        for (annotgroup_uuid, annotgroup_text, annotgroup_note) in zip(
            annotgroup_uuid_list, annotgroup_text_list, annotgroup_note_list
        )
    )
    get_rowid_from_superkey = ibs.get_annotgroup_rowid_from_superkey
    # FIXME: encode superkey paramx
    superkey_paramx = (1,)
    annotgroup_rowid_list = ibs.db.add_cleanly(
        const.ANNOTGROUP_TABLE,
        colnames,
        params_iter,
        get_rowid_from_superkey,
        superkey_paramx,
    )
    return annotgroup_rowid_list


@register_ibs_method
# @register_api('/api/annotgroup/', methods=['DELETE'])
def delete_annotgroup(ibs, annotgroup_rowid_list, config2_=None):
    """annotgroup.delete(annotgroup_rowid_list)

    delete annotgroup rows

    Args:
        annotgroup_rowid_list

    Returns:
        int: num_deleted

    TemplateInfo:
        Tdeleter_native_tbl
        tbl = annotgroup

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_annotgroup_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> annotgroup_rowid_list = ibs._get_all_annotgroup_rowids()[:2]
        >>> num_deleted = ibs.delete_annotgroup(annotgroup_rowid_list)
        >>> print('num_deleted = %r' % (num_deleted,))
    """
    # from wbia.algo.preproc import preproc_annotgroup
    # NO EXTERN IMPORT
    if ut.VERBOSE:
        logger.info('[ibs] deleting %d annotgroup rows' % len(annotgroup_rowid_list))
    # Prepare: Delete externally stored data (if any)
    # preproc_annotgroup.on_delete(ibs, annotgroup_rowid_list, config2_=config2_)
    # NO EXTERN DELETE
    # Finalize: Delete self
    ibs.db.delete_rowids(const.ANNOTGROUP_TABLE, annotgroup_rowid_list)
    num_deleted = len(ut.filter_Nones(annotgroup_rowid_list))
    return num_deleted


@register_ibs_method
@accessor_decors.getter_1toM
# @register_api('/api/annotgroup/gar/rowids/', methods=['GET'])
def get_annotgroup_gar_rowids(ibs, annotgroup_rowid_list, eager=True, nInput=None):
    """
    Auto-docstr for 'get_annotgroup_gar_rowids'

    RESTful:
        Method: GET
        URL:    /api/annotgroup/gar/rowids/
    """
    colnames = (GAR_ROWID,)
    gar_rowid_list = ibs.db.get(
        const.GA_RELATION_TABLE,
        colnames,
        annotgroup_rowid_list,
        id_colname=ANNOTGROUP_ROWID,
        unpack_scalars=False,
    )
    return gar_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annotgroup/note/', methods=['GET'])
def get_annotgroup_note(ibs, annotgroup_rowid_list, eager=True, nInput=None):
    """annotgroup_note_list <- annotgroup.annotgroup_note[annotgroup_rowid_list]

    gets data from the "native" column "annotgroup_note" in the "annotgroup" table

    Args:
        annotgroup_rowid_list (list):

    Returns:
        list: annotgroup_note_list

    TemplateInfo:
        Tgetter_table_column
        col = annotgroup_note
        tbl = annotgroup

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_annotgroup_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> annotgroup_rowid_list = ibs._get_all_annotgroup_rowids()
        >>> eager = True
        >>> annotgroup_note_list = ibs.get_annotgroup_note(annotgroup_rowid_list, eager=eager)
        >>> assert len(annotgroup_rowid_list) == len(annotgroup_note_list)
    """
    id_iter = annotgroup_rowid_list
    colnames = (ANNOTGROUP_NOTE,)
    annotgroup_note_list = ibs.db.get(
        const.ANNOTGROUP_TABLE,
        colnames,
        id_iter,
        id_colname='rowid',
        eager=eager,
        nInput=nInput,
    )
    return annotgroup_note_list


@register_ibs_method
def get_annotgroup_rowid_from_superkey(
    ibs, annotgroup_text_list, eager=True, nInput=None
):
    """annotgroup_rowid_list <- annotgroup[annotgroup_text_list]

    Args:
        superkey lists: annotgroup_text_list

    Returns:
        annotgroup_rowid_list

    TemplateInfo:
        Tgetter_native_rowid_from_superkey
        tbl = annotgroup
    """
    colnames = (ANNOTGROUP_ROWID,)
    # FIXME: col_rowid is not correct
    params_iter = zip(annotgroup_text_list)
    andwhere_colnames = [ANNOTGROUP_TEXT]
    annotgroup_rowid_list = ibs.db.get_where_eq(
        const.ANNOTGROUP_TABLE,
        colnames,
        params_iter,
        andwhere_colnames,
        eager=eager,
        nInput=nInput,
    )
    return annotgroup_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annotgroup/text/', methods=['GET'])
def get_annotgroup_text(ibs, annotgroup_rowid_list, eager=True, nInput=None):
    """annotgroup_text_list <- annotgroup.annotgroup_text[annotgroup_rowid_list]

    gets data from the "native" column "annotgroup_text" in the "annotgroup" table

    Args:
        annotgroup_rowid_list (list):

    Returns:
        list: annotgroup_text_list

    TemplateInfo:
        Tgetter_table_column
        col = annotgroup_text
        tbl = annotgroup

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_annotgroup_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> annotgroup_rowid_list = ibs._get_all_annotgroup_rowids()
        >>> eager = True
        >>> annotgroup_text_list = ibs.get_annotgroup_text(annotgroup_rowid_list, eager=eager)
        >>> assert len(annotgroup_rowid_list) == len(annotgroup_text_list)
    """
    id_iter = annotgroup_rowid_list
    colnames = (ANNOTGROUP_TEXT,)
    annotgroup_text_list = ibs.db.get(
        const.ANNOTGROUP_TABLE,
        colnames,
        id_iter,
        id_colname='rowid',
        eager=eager,
        nInput=nInput,
    )
    return annotgroup_text_list


@register_ibs_method
@accessor_decors.getter_1to1
# @register_api('/api/annotgroup/uuid/', methods=['GET'])
def get_annotgroup_uuid(ibs, annotgroup_rowid_list, eager=True, nInput=None):
    """annotgroup_uuid_list <- annotgroup.annotgroup_uuid[annotgroup_rowid_list]

    gets data from the "native" column "annotgroup_uuid" in the "annotgroup" table

    Args:
        annotgroup_rowid_list (list):

    Returns:
        list: annotgroup_uuid_list

    TemplateInfo:
        Tgetter_table_column
        col = annotgroup_uuid
        tbl = annotgroup

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_annotgroup_funcs import *  # NOQA
        >>> ibs, config2_ = testdata_ibs()
        >>> annotgroup_rowid_list = ibs._get_all_annotgroup_rowids()
        >>> eager = True
        >>> annotgroup_uuid_list = ibs.get_annotgroup_uuid(annotgroup_rowid_list, eager=eager)
        >>> assert len(annotgroup_rowid_list) == len(annotgroup_uuid_list)
    """
    id_iter = annotgroup_rowid_list
    colnames = (ANNOTGROUP_UUID,)
    annotgroup_uuid_list = ibs.db.get(
        const.ANNOTGROUP_TABLE,
        colnames,
        id_iter,
        id_colname='rowid',
        eager=eager,
        nInput=nInput,
    )
    return annotgroup_uuid_list


@register_ibs_method
@accessor_decors.setter
# @register_api('/api/annotgroup/note/', methods=['PUT'])
def set_annotgroup_note(
    ibs, annotgroup_rowid_list, annotgroup_note_list, duplicate_behavior='error'
):
    """annotgroup_note_list -> annotgroup.annotgroup_note[annotgroup_rowid_list]

    Args:
        annotgroup_rowid_list
        annotgroup_note_list

    TemplateInfo:
        Tsetter_native_column
        tbl = annotgroup
        col = annotgroup_note
    """
    id_iter = annotgroup_rowid_list
    colnames = (ANNOTGROUP_NOTE,)
    ibs.db.set(
        const.ANNOTGROUP_TABLE,
        colnames,
        annotgroup_note_list,
        id_iter,
        duplicate_behavior=duplicate_behavior,
    )


@register_ibs_method
@accessor_decors.setter
# @register_api('/api/annotgroup/uuid/', methods=['PUT'])
def set_annotgroup_uuid(
    ibs, annotgroup_rowid_list, annotgroup_uuid_list, duplicate_behavior='error'
):
    """annotgroup_uuid_list -> annotgroup.annotgroup_uuid[annotgroup_rowid_list]

    Args:
        annotgroup_rowid_list
        annotgroup_uuid_list

    TemplateInfo:
        Tsetter_native_column
        tbl = annotgroup
        col = annotgroup_uuid
    """
    id_iter = annotgroup_rowid_list
    colnames = (ANNOTGROUP_UUID,)
    ibs.db.set(
        const.ANNOTGROUP_TABLE,
        colnames,
        annotgroup_uuid_list,
        id_iter,
        duplicate_behavior=duplicate_behavior,
    )
