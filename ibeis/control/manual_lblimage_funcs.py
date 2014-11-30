from __future__ import absolute_import, division, print_function
import uuid
import six  # NOQA
from six.moves import range
from ibeis import constants as const
from ibeis.control.accessor_decors import (
    adder, getter_1to1, default_decorator, getter_1toM)
import utool as ut
from ibeis import ibsfuncs
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_lblimg]')


CLASS_INJECT_KEY = ('IBEISController', 'lblimage')

# Create dectorator to inject these functions into the IBEISController
register_ibs_aliased_method   = ut.make_class_method_decorator(CLASS_INJECT_KEY)
register_ibs_unaliased_method = ut.make_class_method_decorator(CLASS_INJECT_KEY)


def register_ibs_method(func):
    aliastup = (func, 'manual_' + ut.get_funcname(func))
    register_ibs_unaliased_method(func)
    register_ibs_aliased_method(aliastup)
    return func

r"""
Vim add decorator
%s/^\n^@\([^r]\)/\r\r@register_ibs_method\r@\1/gc
%s/\n\n\n\n/\r\r\r/gc
"""
# ADDERS::IMAGE->ENCOUNTER

#
# GETTERS::GLR


@register_ibs_method
@getter_1to1
def get_glr_confidence(ibs, glrid_list):
    """
    Returns:
        list_ (list):  confidence in an image relationship """
    glr_confidence_list = ibs.db.get(const.GL_RELATION_TABLE, ('glr_confidence',), glrid_list)
    return glr_confidence_list


@register_ibs_method
@getter_1to1
def get_glr_lblimage_rowids(ibs, glrid_list):
    """ get the lblimage_rowid belonging to each relationship """
    lblimage_rowids_list = ibs.db.get(const.GL_RELATION_TABLE, ('lblimage_rowid',), glrid_list)
    return lblimage_rowids_list


@register_ibs_method
@getter_1to1
def get_glr_image_rowids(ibs, glrid_list):
    """ get the image_rowid belonging to each relationship """
    image_rowids_list = ibs.db.get(const.GL_RELATION_TABLE, ('image_rowid',), glrid_list)
    return image_rowids_list


@register_ibs_method
@getter_1to1
def get_glr_config_rowid(ibs, glrid_list):
    """
    Returns:
        list_ (list):  config_rowid in an image relationship """
    config_rowid_list = ibs.db.get(const.GL_RELATION_TABLE, ('config_rowid',), glrid_list)
    return config_rowid_list

# ADDERS::LBLIMAGE


@register_ibs_method
@adder
def add_lblimages(ibs, lbltype_rowid_list, value_list, note_list=None, lblimage_uuid_list=None):
    """ Adds new lblimages (labels of imageations)
    creates a new uuid for any new pair(type, value)
    #TODO: reverse order of rowid_list value_list in input
    """
    if note_list is None:
        note_list = [''] * len(value_list)
    # Get random uuids
    if lblimage_uuid_list is None:
        lblimage_uuid_list = [uuid.uuid4() for _ in range(len(value_list))]
    colnames = ['lblimage_uuid', 'lbltype_rowid', 'lblimage_value', 'lblimage_note']
    params_iter = list(zip(lblimage_uuid_list, lbltype_rowid_list, value_list, note_list))
    get_rowid_from_superkey = ibs.get_lblimage_rowid_from_superkey
    superkey_paramx = (1, 2)
    lblimage_rowid_list = ibs.db.add_cleanly(const.LBLIMAGE_TABLE, colnames, params_iter,
                                             get_rowid_from_superkey, superkey_paramx)
    return lblimage_rowid_list


#
# GETTERS::LBLIMAGE_TABLE


@register_ibs_method
@getter_1to1
def get_lblimage_rowid_from_superkey(ibs, lbltype_rowid_list, value_list):
    """
    Returns:
        list_ (list):  lblimage_rowid_list from the superkey (lbltype, value)
    """
    colnames = ('lblimage_rowid',)
    params_iter = zip(lbltype_rowid_list, value_list)
    where_clause = 'lbltype_rowid=? AND lblimage_value=?'
    lblimage_rowid_list = ibs.db.get_where(const.LBLIMAGE_TABLE, colnames, params_iter, where_clause)
    return lblimage_rowid_list


@register_ibs_method
@getter_1to1
def get_lblimage_rowid_from_uuid(ibs, lblimage_uuid_list):
    """
    Returns:
        list_ (list):  lblimage_rowid_list from the superkey (lbltype, value)
    """
    colnames = ('lblimage_rowid',)
    params_iter = lblimage_uuid_list
    id_colname = 'lblimage_uuid'
    lblimage_rowid_list = ibs.db.get(const.LBLIMAGE_TABLE, colnames, params_iter, id_colname=id_colname)
    return lblimage_rowid_list


@register_ibs_method
@getter_1to1
def get_lblimage_uuids(ibs, lblimage_rowid_list):
    lblimageuuid_list = ibs.db.get(const.LBLIMAGE_TABLE, ('lblimage_uuid',), lblimage_rowid_list)
    return lblimageuuid_list


@register_ibs_method
@getter_1to1
def get_lblimage_lbltypes_rowids(ibs, lblimage_rowid_list):
    lbltype_rowid_list = ibs.db.get(const.LBLIMAGE_TABLE, ('lbltype_rowid',), lblimage_rowid_list)
    return lbltype_rowid_list


@register_ibs_method
@getter_1to1
def get_lblimage_notes(ibs, lblimage_rowid_list):
    lblimagenotes_list = ibs.db.get(const.LBLIMAGE_TABLE, ('lblimage_note',), lblimage_rowid_list)
    return lblimagenotes_list


@register_ibs_method
@getter_1to1
def get_lblimage_values(ibs, lblimage_rowid_list, _lbltype=None):
    """
    Returns:
        list_ (list): text lblimages """
    #TODO: Remove keyword argument
    #ibsfuncs.assert_lblimage_rowids_are_type(ibs, lblimage_rowid_list,  ibs.lbltype_ids[_lbltype])
    lblimage_value_list = ibs.db.get(const.LBLIMAGE_TABLE, ('lblimage_value',), lblimage_rowid_list)
    return lblimage_value_list


@register_ibs_method
@default_decorator
def get_lblimage_gids(ibs, lblimage_rowid_list):
    #verbose = len(lblimage_rowid_list) > 20
    # TODO: Optimize IF POSSIBLE
    # FIXME: SLOW
    #if verbose:
    #    print(ut.get_caller_name(N=list(range(0, 20))))
    where_clause = 'lblimage_rowid=?'
    params_iter = [(lblimage_rowid,) for lblimage_rowid in lblimage_rowid_list]
    gids_list = ibs.db.get_where(const.GL_RELATION_TABLE, ('image_rowid',), params_iter,
                                 where_clause, unpack_scalars=False)
    return gids_list


# ADDERS::GLR


@register_ibs_method
@adder
def add_image_relationship_one(ibs, gid_list, lblimage_rowid_list, config_rowid_list=None,
                                glr_confidence_list=None):
    """ Adds a relationship between images and lblimages
        (imageations and labels of imageations) """
    if config_rowid_list is None:
        config_rowid_list = [ibs.MANUAL_CONFIGID] * len(gid_list)
    if glr_confidence_list is None:
        glr_confidence_list = [0.0] * len(gid_list)
    colnames = ('image_rowid', 'lblimage_rowid', 'config_rowid', 'glr_confidence',)
    params_iter = list(zip(gid_list, lblimage_rowid_list, config_rowid_list, glr_confidence_list))
    get_rowid_from_superkey = ibs.get_glrid_from_superkey
    superkey_paramx = (0, 1, 2)  # TODO HAVE SQL GIVE YOU THESE NUMBERS
    glrid_list = ibs.db.add_cleanly(const.GL_RELATION_TABLE, colnames, params_iter,
                                    get_rowid_from_superkey, superkey_paramx)
    return glrid_list


@register_ibs_method
@adder
def add_image_relationship(ibs, gid_list, eid_list):
    """ Adds a relationship between an image and and encounter """
    colnames = ('image_rowid', 'encounter_rowid',)
    params_iter = list(zip(gid_list, eid_list))
    get_rowid_from_superkey = ibs.get_egr_rowid_from_superkey
    superkey_paramx = (0, 1)
    egrid_list = ibs.db.add_cleanly(const.EG_RELATION_TABLE, colnames, params_iter,
                                    get_rowid_from_superkey, superkey_paramx)
    return egrid_list


@register_ibs_method
@getter_1to1
def get_glrid_from_superkey(ibs, gid_list, lblimage_rowid_list, config_rowid_list):
    """
    Args:
        gid_list (list): list of image row-ids
        lblimage_rowid_list (list): list of lblimage row-ids
        config_rowid_list (list): list of config row-ids
    Returns:
        glrid_list (list): image-label relationship id list
    """
    colnames = ('image_rowid',)
    params_iter = zip(gid_list, lblimage_rowid_list, config_rowid_list)
    where_clause = 'image_rowid=? AND lblimage_rowid=? AND config_rowid=?'
    glrid_list = ibs.db.get_where(const.GL_RELATION_TABLE, colnames, params_iter, where_clause)
    return glrid_list


@register_ibs_method
@getter_1toM
def get_image_glrids(ibs, gid_list, configid=None):
    """ FIXME: __name__
    Get all the relationship ids belonging to the input images
    if lblimage lbltype is specified the relationship ids are filtered to
    be only of a specific lbltype/category/type
    """
    if configid is None:
        configid = ibs.MANUAL_CONFIGID
    params_iter = ((gid, configid) for gid in gid_list)
    where_clause = 'image_rowid=? AND config_rowid=?'
    glrids_list = ibs.db.get_where(const.GL_RELATION_TABLE, ('glr_rowid',), params_iter,
                                   where_clause=where_clause, unpack_scalars=False)
    # assert all([x > 0 for x in map(len, alrids_list)]), 'annotations must have at least one relationship'
    return glrids_list


@register_ibs_method
@getter_1toM
def get_annot_alrids(ibs, aid_list, configid=None):
    """ FIXME: __name__
    Get all the relationship ids belonging to the input annotations
    if lblannot lbltype is specified the relationship ids are filtered to
    be only of a specific lbltype/category/type
    """
    if configid is None:
        alrids_list = ibs.db.get(const.AL_RELATION_TABLE, ('alr_rowid',), aid_list,
                                 id_colname='annot_rowid', unpack_scalars=False)
    else:
        params_iter = ((aid, configid) for aid in aid_list)
        where_clause = 'annot_rowid=? AND config_rowid=?'
        alrids_list = ibs.db.get_where(const.AL_RELATION_TABLE, ('alr_rowid',), params_iter,
                                       where_clause=where_clause, unpack_scalars=False)
    # assert all([x > 0 for x in map(len, alrids_list)]), 'annotations must have at least one relationship'
    return alrids_list


@register_ibs_method
@getter_1toM
def get_annot_alrids_oftype(ibs, aid_list, lbltype_rowid, configid=None):
    """
    Get all the relationship ids belonging to the input annotations where the
    relationship ids are filtered to be only of a specific lbltype/category/type
    """
    alrids_list = ibs.get_annot_alrids(aid_list, configid=configid)
    # Get lblannot_rowid of each relationship
    lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
    # Get the type of each lblannot
    lbltype_rowids_list = ibsfuncs.unflat_map(ibs.get_lblannot_lbltypes_rowids, lblannot_rowids_list)
    # only want the nids of individuals, not species, for example
    valids_list = [[typeid == lbltype_rowid for typeid in rowids] for rowids in lbltype_rowids_list]
    alrids_list = [ut.filter_items(alrids, valids) for alrids, valids in zip(alrids_list, valids_list)]
    if configid is None:
        def resolution_func_first(alrid_list):
            return [ alrid_list[0] ]

        def resolution_func_lowest_config(alrid_list):
            config_rowid_list = ibs.get_alr_config(alrid_list)
            temp = sorted(list(zip(config_rowid_list, alrid_list)))
            return [ temp[0][0] ]

        alrids_list = [
            resolution_func_first(alrid_list)
            if len(alrid_list) > 1 else
            alrid_list
            for alrid_list in alrids_list
        ]
    assert all([len(alrid_list) < 2 for alrid_list in alrids_list]),\
        ("More than one type per lbltype.  ALRIDS: " + str(alrids_list) +
         ", ROW: " + str(lbltype_rowid) + ", KEYS:" + str(ibs.lbltype_ids))
    return alrids_list


@register_ibs_method
@getter_1toM
def get_annot_lblannot_rowids(ibs, aid_list):
    """
    Returns:
        list_ (list): the name id of each annotation. """
    # Get all the annotation lblannot relationships
    # filter out only the ones which specify names
    alrids_list = ibs.get_annot_alrids(aid_list)
    lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
    return lblannot_rowids_list


@register_ibs_method
@getter_1toM
def get_annot_lblannot_rowids_oftype(ibs, aid_list, _lbltype=None):
    """
    Returns:
        list_ (list): the name id of each annotation. """
    # Get all the annotation lblannot relationships
    # filter out only the ones which specify names
    assert _lbltype is not None, 'should be using lbltype_rowids anyway'
    alrids_list = ibs.get_annot_alrids_oftype(aid_list, ibs.lbltype_ids[_lbltype])
    lblannot_rowids_list = ibsfuncs.unflat_map(ibs.get_alr_lblannot_rowids, alrids_list)
    return lblannot_rowids_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.manual_lblimage_funcs
        python -m ibeis.control.manual_lblimage_funcs --allexamples
        python -m ibeis.control.manual_lblimage_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
