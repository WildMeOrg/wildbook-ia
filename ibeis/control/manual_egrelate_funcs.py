"""
CommandLine:
    # Autogenerate Encounter Functions
    # key should be the table name
    # the write flag makes a file, but dont use that
    python -m ibeis.templates.template_generator --key encounter_image_relationship --onlyfn
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants as const
from ibeis.control import accessor_decors
from ibeis.control.controller_inject import make_ibs_register_decorator
import utool as ut
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[manual_egr]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


@register_ibs_method
def delete_empty_eids(ibs):
    """ Removes encounters without images

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.control.manual_egrelate_funcs --test-delete_empty_eids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_egrelate_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> # execute function
        >>> result = ibs.delete_empty_eids()
        >>> # verify results
        >>> print(result)
    """
    eid_list = ibs.get_valid_eids(min_num_gids=0)
    nGids_list = ibs.get_encounter_num_gids(eid_list)
    is_invalid = [nGids == 0 for nGids in nGids_list]
    invalid_eids = ut.filter_items(eid_list, is_invalid)
    ibs.delete_encounters(invalid_eids)


@register_ibs_method
@accessor_decors.getter_1to1
def get_image_egrids(ibs, gid_list):
    """
    Returns:
        list_ (list):  a list of encounter-image-relationship rowids for each imageid """
    # TODO: Group type
    params_iter = ((gid,) for gid in gid_list)
    where_clause = 'image_rowid=?'
    # list of relationships for each image
    egrids_list = ibs.db.get_where(const.EG_RELATION_TABLE, ('egr_rowid',), params_iter, where_clause, unpack_scalars=False)
    return egrids_list


@register_ibs_method
@accessor_decors.deleter
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_imgs_reviewed_str'])
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_names_with_exemplar_str'])
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_annotmatch_reviewed_str'])
def delete_egr_encounter_relations(ibs, eid_list):
    """ Removes relationship between input encounters and all images """
    ibs.db.delete(const.EG_RELATION_TABLE, eid_list, id_colname='encounter_rowid')


@register_ibs_method
@accessor_decors.deleter
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_imgs_reviewed_str'])
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_names_with_exemplar_str'])
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_annotmatch_reviewed_str'])
def delete_egr_image_relations(ibs, gid_list):
    """ Removes relationship between input images and all encounters """
    ibs.db.delete(const.EG_RELATION_TABLE, gid_list, id_colname='image_rowid')


@register_ibs_method
@accessor_decors.deleter
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['image_rowids'], rowidx=1)
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_imgs_reviewed_str'])
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_names_with_exemplar_str'])
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_annotmatch_reviewed_str'])
def unrelate_images_and_encounters(ibs, gid_list, eid_list):
    """
    Seems to unrelate specific image encounter pairs

    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):
        eid_list (list):

    Returns:
        list: gids_list

    CommandLine:
        python -m ibeis.control.manual_egrelate_funcs --test-unrelate_images_and_encounters
        python -c "import utool; print(utool.auto_docstr('ibeis.control.manual_egrelate_funcs', 'delete_egr_image_relations'))"

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_egrelate_funcs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> # Reset and compute encounters
        >>> ibs.delete_all_encounters()
        >>> ibs.compute_encounters()
        >>> eid_list = ibs.get_valid_eids()
        >>> gids_list = ibs.get_encounter_gids(eid_list)
        >>> assert len(eid_list) == 2
        >>> assert len(gids_list) == 2
        >>> assert len(gids_list[0]) == 7
        >>> assert len(gids_list[1]) == 6
        >>> # Add encounter 2 gids to encounter 1 so an image belongs to multiple encounters
        >>> enc2_gids = gids_list[1][0:1]
        >>> enc1_eids = eid_list[0:1]
        >>> ibs.add_image_relationship(enc2_gids, enc1_eids)
        >>> # Now delete the image from the encounter 2
        >>> enc2_eids = eid_list[1:2]
        >>> # execute function
        >>> ibs.unrelate_images_and_encounters(enc2_gids, enc2_eids)
        >>> # verify results
        >>> ibs.print_egpairs_table()
        >>> eid_list_ = ibs.get_valid_eids()
        >>> gids_list_ = ibs.get_encounter_gids(eid_list_)
        >>> result = str(gids_list_)
        >>> print(result)
        >>> # enc2_gids should now only be in encounter1
        >>> assert enc2_gids[0] in gids_list_[0]
        >>> assert enc2_gids[0] not in gids_list_[1]
    """
    # WHAT IS THIS FUNCTION? FIXME CALLS WEIRD FUNCTION
    if ut.VERBOSE:
        print('[ibs] deleting %r image\'s encounter ids' % len(gid_list))
    egrid_list = ut.flatten(ibs.get_encounter_egrids(eid_list=eid_list, gid_list=gid_list))
    ibs.db.delete_rowids(const.EG_RELATION_TABLE, egrid_list)


# GETTERS::EG_RELATION_TABLE


@register_ibs_method
@accessor_decors.getter_1to1
def get_egr_rowid_from_superkey(ibs, gid_list, eid_list):
    """
    Returns:
        egrid_list (list):  eg-relate-ids from info constrained to be unique (eid, gid) """
    colnames = ('image_rowid',)
    params_iter = zip(gid_list, eid_list)
    where_clause = 'image_rowid=? AND encounter_rowid=?'
    egrid_list = ibs.db.get_where(const.EG_RELATION_TABLE, colnames, params_iter, where_clause)
    return egrid_list


@register_ibs_method
@accessor_decors.adder
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['image_rowids'], rowidx=1)
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_imgs_reviewed_str'], rowidx=1)
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_names_with_exemplar_str'], rowidx=1)
@accessor_decors.cache_invalidator(const.ENCOUNTER_TABLE, ['percent_annotmatch_reviewed_str'], rowidx=1)
def add_image_relationship(ibs, gid_list, eid_list):
    """ Adds a relationship between an image and and encounter """
    colnames = ('image_rowid', 'encounter_rowid',)
    params_iter = list(zip(gid_list, eid_list))
    get_rowid_from_superkey = ibs.get_egr_rowid_from_superkey
    superkey_paramx = (0, 1)
    egrid_list = ibs.db.add_cleanly(const.EG_RELATION_TABLE, colnames, params_iter,
                                    get_rowid_from_superkey, superkey_paramx)
    return egrid_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.manual_egrelate_funcs
        python -m ibeis.control.manual_egrelate_funcs --allexamples
        python -m ibeis.control.manual_egrelate_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
