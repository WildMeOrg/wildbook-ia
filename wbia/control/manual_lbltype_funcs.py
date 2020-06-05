# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia import constants as const
from wbia.control.accessor_decors import adder, getter_1to1, ider
import utool as ut
from wbia.control.controller_inject import make_ibs_register_decorator

print, rrr, profile = ut.inject2(__name__)


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


@register_ibs_method
@ider
def _get_all_known_lblannot_rowids(ibs, _lbltype):
    """
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names) """
    all_known_lblannot_rowids = ibs.db.get_all_rowids_where(
        const.LBLANNOT_TABLE, 'lbltype_rowid=?', (ibs.lbltype_ids[_lbltype],)
    )
    return all_known_lblannot_rowids


@register_ibs_method
@adder
def add_lbltype(ibs, text_list, default_list):
    """ Adds a label type and its default value
    Should only be called at the begining of the program.
    """
    params_iter = zip(text_list, default_list)
    colnames = (
        'lbltype_text',
        'lbltype_default',
    )
    get_rowid_from_superkey = ibs.get_lbltype_rowid_from_text
    lbltype_rowid_list = ibs.db.add_cleanly(
        const.LBLTYPE_TABLE, colnames, params_iter, get_rowid_from_superkey
    )
    return lbltype_rowid_list


#
# GETTERS::LBLTYPE


@register_ibs_method
@getter_1to1
def get_lbltype_rowid_from_text(ibs, text_list):
    """
    Returns:
        lbltype_rowid (list): lbltype_rowid where the lbltype_text is given
    """
    # FIXME: MAKE SQL-METHOD FOR NON-ROWID GETTERS
    # FIXME: Use unique SUPERKEYS instead of specifying id_colname
    lbltype_rowid = ibs.db.get(
        const.LBLTYPE_TABLE, ('lbltype_rowid',), text_list, id_colname='lbltype_text'
    )
    return lbltype_rowid


@register_ibs_method
@getter_1to1
def get_lbltype_default(ibs, lbltype_rowid_list):
    lbltype_default_list = ibs.db.get(
        const.LBLTYPE_TABLE, ('lbltype_default',), lbltype_rowid_list
    )
    return lbltype_default_list


@register_ibs_method
@getter_1to1
def get_lbltype_text(ibs, lbltype_rowid_list):
    lbltype_text_list = ibs.db.get(
        const.LBLTYPE_TABLE, ('lbltype_text',), lbltype_rowid_list
    )
    return lbltype_text_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.control.manual_lbltype_funcs
        python -m wbia.control.manual_lbltype_funcs --allexamples
        python -m wbia.control.manual_lbltype_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
