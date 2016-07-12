# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
from ibeis import _ibeis_object
from ibeis.control.controller_inject import make_ibs_register_decorator
(print, rrr, profile) = ut.inject2(__name__, '[images]')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


@register_ibs_method
def images(ibs, gids=None):
    if gids is None:
        gids = ibs.get_valid_gids()
    return Images(gids, ibs)

BASE_TYPE = type


class ImageIBEISPropertyInjector(BASE_TYPE):
    def __init__(metaself, name, bases, dct):
        super(ImageIBEISPropertyInjector, metaself).__init__(name, bases, dct)
        metaself.rrr = rrr
        #misc = [ 'instancelist', 'gids_with_aids', 'lazydict', ]  #
        attrs = ['aids', 'aids_of_species', 'annot_uuids',
                 'annot_uuids_of_species', 'annotation_bboxes',
                 'annotation_thetas', 'contributor_rowid', 'contributor_tag',
                 'datetime', 'datetime_str', 'detect_confidence',
                 'detectpaths', 'enabled', 'exts', 'gid', 'glrids', 'gnames',
                 'gps', 'gsgrids', 'heights', 'imagesettext', 'imgset_uuids',
                 'imgsetids', 'lat', 'location_codes', 'lon', 'missing_uuid',
                 'name_uuids', 'nids', 'notes', 'num_annotations',
                 'orientation', 'orientation_str', 'party_rowids', 'party_tag',
                 'paths', 'reviewed', 'sizes', 'species_rowids',
                 'species_uuids', 'thumbpath', 'thumbtup', 'time_statstr',
                 'timedelta_posix', 'unixtime',
                 'uris',
                 'uris_original',
                 'uuids', 'widths']
        #inverse_attrs = [
        #     'gids_from_uuid',
        #]
        objname = 'image'
        _ibeis_object._inject_getter_attrs(metaself, objname, attrs, [])


@ut.reloadable_class
@six.add_metaclass(ImageIBEISPropertyInjector)
class Images(_ibeis_object.PrimaryObject):
    """
    Represents a group of annotations. Efficiently accesses properties from a
    database using lazy evaluation.

    CommandLine:
        python -m ibeis.images Images --show

    Example:
        >>> from ibeis.images import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> gids = ibs.get_valid_gids()
        >>> g = self = images = Images(gids, ibs)
        >>> print(g.widths)
        >>> print(g)
        <Images(num=13)>

    """
    def __init__(self, gids, ibs, config=None):
        super(Images, self).__init__(gids, ibs, config)

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.images
        python -m ibeis.images --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
