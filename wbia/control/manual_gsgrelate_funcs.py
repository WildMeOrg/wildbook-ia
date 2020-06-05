# -*- coding: utf-8 -*-
"""
CommandLine:
    # Autogenerate ImageSet Functions
    # key should be the table name
    # the write flag makes a file, but dont use that
    python -m wbia.templates.template_generator --key imageset_image_relationship --onlyfn
"""
from __future__ import absolute_import, division, print_function
from wbia import constants as const
from wbia.control import accessor_decors
from wbia.control.controller_inject import make_ibs_register_decorator
import utool as ut

print, rrr, profile = ut.inject2(__name__)


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


@register_ibs_method
def delete_empty_imgsetids(ibs):
    """ Removes imagesets without images

    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.control.manual_gsgrelate_funcs --test-delete_empty_imgsetids

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_gsgrelate_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> # execute function
        >>> result = ibs.delete_empty_imgsetids()
        >>> # verify results
        >>> print(result)
    """
    imgsetid_list = ibs.get_valid_imgsetids(min_num_gids=0)
    nGids_list = ibs.get_imageset_num_gids(imgsetid_list)
    is_invalid = [nGids == 0 for nGids in nGids_list]
    invalid_imgsetids = ut.compress(imgsetid_list, is_invalid)
    ibs.delete_imagesets(invalid_imgsetids)


@register_ibs_method
@accessor_decors.getter_1to1
def get_image_gsgrids(ibs, gid_list):
    """
    Returns:
        list_ (list):  a list of imageset-image-relationship rowids for each imageid """
    # TODO: Group type
    params_iter = ((gid,) for gid in gid_list)
    where_clause = 'image_rowid=?'
    # list of relationships for each image
    gsgrids_list = ibs.db.get_where(
        const.GSG_RELATION_TABLE,
        ('gsgr_rowid',),
        params_iter,
        where_clause,
        unpack_scalars=False,
    )
    return gsgrids_list


@register_ibs_method
@accessor_decors.deleter
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['percent_imgs_reviewed_str'])
@accessor_decors.cache_invalidator(
    const.IMAGESET_TABLE, ['percent_names_with_exemplar_str']
)
@accessor_decors.cache_invalidator(
    const.IMAGESET_TABLE, ['percent_annotmatch_reviewed_str']
)
def delete_gsgr_imageset_relations(ibs, imgsetid_list):
    """ Removes relationship between input imagesets and all images """
    ibs.db.delete(const.GSG_RELATION_TABLE, imgsetid_list, id_colname='imageset_rowid')


@register_ibs_method
@accessor_decors.deleter
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['percent_imgs_reviewed_str'])
@accessor_decors.cache_invalidator(
    const.IMAGESET_TABLE, ['percent_names_with_exemplar_str']
)
@accessor_decors.cache_invalidator(
    const.IMAGESET_TABLE, ['percent_annotmatch_reviewed_str']
)
def delete_gsgr_image_relations(ibs, gid_list):
    """ Removes relationship between input images and all imagesets """
    ibs.db.delete(const.GSG_RELATION_TABLE, gid_list, id_colname='image_rowid')


@register_ibs_method
@accessor_decors.deleter
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['image_rowids'], rowidx=1)
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['percent_imgs_reviewed_str'])
@accessor_decors.cache_invalidator(
    const.IMAGESET_TABLE, ['percent_names_with_exemplar_str']
)
@accessor_decors.cache_invalidator(
    const.IMAGESET_TABLE, ['percent_annotmatch_reviewed_str']
)
def unrelate_images_and_imagesets(ibs, gid_list, imgsetid_list):
    """
    Seems to unrelate specific image imageset pairs

    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):
        imgsetid_list (list):

    Returns:
        list: gids_list

    CommandLine:
        python -m wbia.control.manual_gsgrelate_funcs --test-unrelate_images_and_imagesets
        python -c "import utool; print(utool.auto_docstr('wbia.control.manual_gsgrelate_funcs', 'delete_gsgr_image_relations'))"

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.manual_gsgrelate_funcs import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> # Reset and compute imagesets
        >>> ibs.delete_all_imagesets()
        >>> ibs.compute_occurrences(config={'use_gps': False, 'seconds_thresh': 600})
        >>> imgsetid_list = ibs.get_valid_imgsetids()
        >>> gids_list = ibs.get_imageset_gids(imgsetid_list)
        >>> assert len(imgsetid_list) == 2, 'bad len %r' % (len(imgsetid_list),)
        >>> assert len(gids_list) == 2, 'bad len %r' % (len(gids_list),)
        >>> assert len(gids_list[0]) == 7, 'bad len %r' % (len(gids_list[0]),)
        >>> assert len(gids_list[1]) == 6, 'bad len %r' % (len(gids_list[1]),)
        >>> # Add imageset 2 gids to imageset 1 so an image belongs to multiple imagesets
        >>> imgset2_gids = gids_list[1][0:1]
        >>> imgset1_imgsetids = imgsetid_list[0:1]
        >>> ibs.add_image_relationship(imgset2_gids, imgset1_imgsetids)
        >>> # Now delete the image from the imageset 2
        >>> imgset2_imgsetids = imgsetid_list[1:2]
        >>> # execute function
        >>> ibs.unrelate_images_and_imagesets(imgset2_gids, imgset2_imgsetids)
        >>> # verify results
        >>> ibs.print_egpairs_table()
        >>> imgsetid_list_ = ibs.get_valid_imgsetids()
        >>> gids_list_ = ibs.get_imageset_gids(imgsetid_list_)
        >>> result = str(gids_list_)
        >>> print(result)
        >>> # imgset2_gids should now only be in imageset1
        >>> assert imgset2_gids[0] in gids_list_[0], 'imgset2_gids should now only be in imageset1'
        >>> assert imgset2_gids[0] not in gids_list_[1], 'imgset2_gids should now only be in imageset1'
    """
    # WHAT IS THIS FUNCTION? FIXME CALLS WEIRD FUNCTION
    if ut.VERBOSE:
        print("[ibs] deleting %r image's imageset ids" % len(gid_list))
    gsgrid_list = ut.flatten(
        ibs.get_imageset_gsgrids(imgsetid_list=imgsetid_list, gid_list=gid_list)
    )
    ibs.db.delete_rowids(const.GSG_RELATION_TABLE, gsgrid_list)


# GETTERS::GSG_RELATION_TABLE


@register_ibs_method
@accessor_decors.getter_1to1
def get_gsgr_rowid_from_superkey(ibs, gid_list, imgsetid_list):
    """
    Returns:
        gsgrid_list (list):  eg-relate-ids from info constrained to be unique (imgsetid, gid) """
    colnames = ('image_rowid',)
    params_iter = zip(gid_list, imgsetid_list)
    where_clause = 'image_rowid=? AND imageset_rowid=?'
    gsgrid_list = ibs.db.get_where(
        const.GSG_RELATION_TABLE, colnames, params_iter, where_clause
    )
    return gsgrid_list


@register_ibs_method
@accessor_decors.adder
@accessor_decors.cache_invalidator(const.IMAGESET_TABLE, ['image_rowids'], rowidx=1)
@accessor_decors.cache_invalidator(
    const.IMAGESET_TABLE, ['percent_imgs_reviewed_str'], rowidx=1
)
@accessor_decors.cache_invalidator(
    const.IMAGESET_TABLE, ['percent_names_with_exemplar_str'], rowidx=1
)
@accessor_decors.cache_invalidator(
    const.IMAGESET_TABLE, ['percent_annotmatch_reviewed_str'], rowidx=1
)
def add_image_relationship(ibs, gid_list, imgsetid_list):
    """ Adds a relationship between an image and and imageset """
    colnames = (
        'image_rowid',
        'imageset_rowid',
    )
    params_iter = list(zip(gid_list, imgsetid_list))
    get_rowid_from_superkey = ibs.get_gsgr_rowid_from_superkey
    superkey_paramx = (0, 1)
    gsgrid_list = ibs.db.add_cleanly(
        const.GSG_RELATION_TABLE,
        colnames,
        params_iter,
        get_rowid_from_superkey,
        superkey_paramx,
    )
    return gsgrid_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.control.manual_gsgrelate_funcs
        python -m wbia.control.manual_gsgrelate_funcs --allexamples
        python -m wbia.control.manual_gsgrelate_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
