# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
from ibeis.control.controller_inject import make_ibs_register_decorator
(print, rrr, profile) = ut.inject2(__name__, '[images]')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


def _find_ibeis_attrs(ibs, objname, blacklist=[]):
    r"""
    Args:
        ibs (ibeis.IBEISController):  images analysis api

    CommandLine:
        python -m ibeis.images _find_ibeis_attrs --show

    Example:
        >>> from ibeis.images import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> objname = 'images'
        >>> blacklist = []
        >>> _find_ibeis_attrs(ibs, objname, blacklist)
    """
    import re
    getter_prefix = 'get_' + objname + '_'
    found_funcnames = ut.search_module(ibs, getter_prefix)
    pat = getter_prefix + ut.named_field('attr', '.*')
    for stopword in blacklist:
        found_funcnames = [fn for fn in found_funcnames if stopword not in fn]
    matched_attrs = [re.match(pat, fn).groupdict()['attr'] for fn in found_funcnames]


@register_ibs_method
def images(ibs, gids=None):
    from ibeis import images
    if gids is None:
        gids = ibs.get_valid_gids()
    return Images(gids, ibs)


BASE_TYPE = type

class ImageIBEISPropertyInjector(BASE_TYPE):
    def __init__(metaself, name, bases, dct):
        super(ImageIBEISPropertyInjector, metaself).__init__(name, bases, dct)
        metaself.rrr = rrr

        misc = [ 'instancelist', 'gids_with_aids', 'lazydict', ]
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

        inverse_attrs = [
             'gids_from_uuid',
        ]

        def make_injectable_method(ibs_attr):
            def _wrapped_method(self, *args, **kwargs):
                #print('**kwargs = %r' % (kwargs,))
                #print('*args = %r' % (args,))
                ibs_callable = getattr(self.ibs, ibs_attr)
                return ibs_callable(self.gids, *args, **kwargs)
            return _wrapped_method
        for attr in attrs:
            ibs_getter = make_injectable_method('get_image_' + attr)
            setattr(metaself, '_get_' + attr, ibs_getter)
            setattr(metaself, attr, property(ibs_getter))

import six

@ut.reloadable_class
@six.add_metaclass(ImageIBEISPropertyInjector)
class Images(ut.NiceRepr):
    """
    Represents a group of annotations. Efficiently accesses properties from a
    database using lazy evaluation.

    CommandLine:
        python -m ibeis.images Images --show

    Example:
        >>> from ibeis.images import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> gids = ibs.get_valid_aids()
        >>> g = self = images = Images(gids, ibs)
        >>> print(g.widths)
        >>> print(g)
        <Images(num=13)>

    """
    def __init__(self, gids, ibs):
        self.ibs = ibs
        self.gids = gids
        self._islist = True
        self.ibs = ibs
        #self._initialize_self()

    #def _initialize_self(self):
    #    misc = [ 'instancelist', 'gids_with_aids', 'lazydict', ]
    #    attrs = ['aids', 'aids_of_species', 'annot_uuids',
    #             'annot_uuids_of_species', 'annotation_bboxes',
    #             'annotation_thetas', 'contributor_rowid', 'contributor_tag',
    #             'datetime', 'datetime_str', 'detect_confidence',
    #             'detectpaths', 'enabled', 'exts', 'gid', 'glrids', 'gnames',
    #             'gps', 'gsgrids', 'heights', 'imagesettext', 'imgset_uuids',
    #             'imgsetids', 'lat', 'location_codes', 'lon', 'missing_uuid',
    #             'name_uuids', 'nids', 'notes', 'num_annotations',
    #             'orientation', 'orientation_str', 'party_rowids', 'party_tag',
    #             'paths', 'reviewed', 'sizes', 'species_rowids',
    #             'species_uuids', 'thumbpath', 'thumbtup', 'time_statstr',
    #             'timedelta_posix', 'unixtime', 'uris', 'uris_original',
    #             'uuids', 'widths']

    #    inverse_attrs = [
    #         'gids_from_uuid',
    #    ]

    #    from ibeis import images
    #    def make_injectable_method(self, ibs_callable):
    #        def _wrapped_method(self, *args, **kwargs):
    #            print('**kwargs = %r' % (kwargs,))
    #            print('*args = %r' % (args,))
    #            return ibs_callable(self.gids, *args, **kwargs)
    #        return _wrapped_method
    #    for attr in attrs:
    #        ibs_callable = getattr(self.ibs, 'get_image_' + attr)
    #        ibs_getter = make_injectable_method(self, ibs_callable)
    #        #setattr(images.Images, attr, property(ibs_getter))
    #        ut.inject_func_as_method(self, ibs_getter, method_name='_get_' + attr, allow_override=True)
    #        ut.inject_func_as_property(self, ibs_getter, method_name=attr, allow_override=True)

    # def filter(self, filterkw):
    #     pass

    # def filter_flags(self, filterkw):
    #     pass

    def take(self, idxs):
        return images.Images(ut.take(self.gids, idxs), self.ibs)

    def __iter__(self):
        return iter(self.gids)

    def chunks(self,  chunksize):
        from ibeis import images
        return (images.Images(gids, self.ibs) for gids in ut.ichunks(self, chunksize))

    def compress(self,  flags):
        from ibeis import images
        return images.Images(ut.compress(self.gids, flags), self.ibs)

    def groupby(self, labels):
        unique_labels, groupxs = ut.group_indices(labels)
        annot_groups = [self.take(idxs) for idxs in groupxs]
        return annot_groups

    def __nice__(self):
        if self._islist:
            # try:
            #     return '(%s)' % (self._get_hashid_uuid(),)
            # except Exception:
            return '(num=%r)' % (len(self.gids))
        else:
            return '(single)'


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
