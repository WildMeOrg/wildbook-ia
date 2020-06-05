# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
from wbia import _wbia_object
from wbia.control.controller_inject import make_ibs_register_decorator

(print, rrr, profile) = ut.inject2(__name__, '[images]')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)

BASE_TYPE = type

try:
    from wbia import _autogen_image_base

    IMAGE_BASE = _autogen_image_base._image_base_class
except ImportError:
    IMAGE_BASE = _wbia_object.ObjectList1D

try:
    from wbia import _autogen_imageset_base

    IMAGESET_BASE = _autogen_imageset_base._imageset_base_class
except ImportError:
    IMAGESET_BASE = _wbia_object.ObjectList1D


@register_ibs_method
def images(ibs, gids=None, uuids=None, **kwargs):
    """ Makes an Images object """
    if uuids is not None:
        assert gids is None, 'specify one primary key'
        gids = ibs.get_image_gids_from_uuid(uuids)
    if gids is None:
        gids = ibs.get_valid_gids()
    elif gids.__class__.__name__ == 'Images':
        return gids
    gids = ut.ensure_iterable(gids)
    return Images(gids, ibs, **kwargs)


@register_ibs_method
def imagesets(ibs, gsids=None, text=None):
    if text is not None:
        gsids = ibs.get_imageset_imgsetids_from_text(text)
    if gsids is None:
        gsids = ibs.get_valid_imgsetids()
    elif gsids.__class__.__name__ == 'ImageSets':
        return gsids
    gsids = ut.ensure_iterable(gsids)
    return ImageSets(gsids, ibs)


class ImageIBEISPropertyInjector(BASE_TYPE):
    def __init__(metaself, name, bases, dct):
        super(ImageIBEISPropertyInjector, metaself).__init__(name, bases, dct)
        metaself.rrr = rrr
        # misc = [ 'instancelist', 'gids_with_aids', 'lazydict', ]  #
        attrs = [
            'aids',
            'aids_of_species',
            'annot_uuids',
            'annot_uuids_of_species',
            'annotation_bboxes',
            'annotation_thetas',
            'contributor_rowid',
            'contributor_tag',
            'datetime',
            'datetime_str',
            'detect_confidence',
            'detectpaths',
            'enabled',
            'exts',
            'gid',
            'glrids',
            'gnames',
            'gps',
            'gps2',
            'gsgrids',
            'heights',
            'imagesettext',
            'imgset_uuids',
            'imgsetids',
            'lat',
            'location_codes',
            'lon',
            'missing_uuid',
            'name_uuids',
            'nids',
            'notes',
            'num_annotations',
            'orientation',
            'orientation_str',
            'party_rowids',
            'party_tag',
            'paths',
            'reviewed',
            'sizes',
            'species_rowids',
            'species_uuids',
            'thumbpath',
            'thumbtup',
            'time_statstr',
            'timedelta_posix',
            'unixtime',
            'unixtime_asfloat',
            'unixtime2',
            'uris',
            'uris_original',
            'uuids',
            'widths' 'imgdata',
        ]
        # inverse_attrs = [
        #     'gids_from_uuid',
        # ]
        objname = 'image'
        _wbia_object._inject_getter_attrs(metaself, objname, attrs, [])


# @ut.reloadable_class
@six.add_metaclass(ImageIBEISPropertyInjector)
class Images(IMAGE_BASE):
    """
    Represents a group of annotations. Efficiently accesses properties from a
    database using lazy evaluation.

    CommandLine:
        python -m wbia.images Images --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.images import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> gids = ibs.get_valid_gids()
        >>> g = self = images = Images(gids, ibs)
        >>> print(g.widths)
        >>> print(g)
        <Images(num=13)>
    """

    # def __init__(self, gids, ibs, config=None):
    #    super(Images, self).__init__(gids, ibs, config)

    @property
    def gids(self):
        return self._rowids

    @property
    def annots(self):
        return [self._ibs.annots(aids) for aids in self.aids]

    @property
    def _annot_groups(self):
        return self._ibs._annot_groups(self.annots)

    def remove_from_imageset(self, imageset_text):
        ibs = self._ibs
        if isinstance(imageset_text, six.string_types):
            gsid = ibs.get_imageset_imgsetids_from_text(imageset_text)
            gsids = [gsid] * len(self)
        else:
            gsids = ibs.get_imageset_imgsetids_from_text(imageset_text)
        ibs.unrelate_images_and_imagesets(self.gids, gsids)

    def append_to_imageset(self, imageset_text):
        ibs = self._ibs
        if isinstance(imageset_text, six.string_types):
            gsid = ibs.get_imageset_imgsetids_from_text(imageset_text)
            gsids = [gsid] * len(self)
        else:
            gsids = ibs.get_imageset_imgsetids_from_text(imageset_text)
        ibs.add_image_relationship(self.gids, gsids)

    def show(self, *args, **kwargs):
        if len(self) != 1:
            raise ValueError('Can only show one, got {}'.format(len(self)))
        from wbia.viz import viz_image

        for gid in self:
            return viz_image.show_image(self._ibs, gid, *args, **kwargs)


class ImageSetAttrInjector(BASE_TYPE):
    """
    Example:
        >>> # SCRIPT
        >>> from wbia import _wbia_object
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> objname = 'imageset'
        >>> blacklist = []
        >>> _wbia_object._find_wbia_attrs(ibs, objname, blacklist)
    """

    def __init__(metaself, name, bases, dct):
        super(ImageSetAttrInjector, metaself).__init__(name, bases, dct)
        metaself.rrr = rrr
        # misc = [ 'instancelist', 'gids_with_aids', 'lazydict', ]  #
        attrs = [
            'aids',
            'configid',
            'custom_filtered_aids',
            'duration',
            'end_time_posix',
            'fraction_annotmatch_reviewed',
            'fraction_imgs_reviewed',
            'fraction_names_with_exemplar',
            'gids',
            'gps_lats',
            'gps_lons',
            'gsgrids',
            'image_uuids',
            'imgsetids_from_text',
            'imgsetids_from_uuid',
            'isoccurrence',
            'name_uuids',
            'nids',
            'note',
            'notes',
            'num_aids',
            'num_annotmatch_reviewed',
            'num_annots_reviewed',
            'num_gids',
            'num_imgs_reviewed',
            'num_names_with_exemplar',
            'percent_annotmatch_reviewed_str',
            'percent_imgs_reviewed_str',
            'percent_names_with_exemplar_str',
            'processed_flags',
            'shipped_flags',
            'smart_waypoint_ids',
            'smart_xml_contents',
            'smart_xml_fnames',
            'start_time_posix',
            'text',
            'uuid',
            'uuids',
        ]
        # inverse_attrs = [
        #     'gids_from_uuid',
        # ]
        objname = 'imageset'
        _wbia_object._inject_getter_attrs(metaself, objname, attrs, [])


# @ut.reloadable_class
@six.add_metaclass(ImageSetAttrInjector)
class ImageSets(IMAGESET_BASE):
    """
    Represents a group of annotations. Efficiently accesses properties from a
    database using lazy evaluation.

    CommandLine:
        python -m wbia.images ImageSets

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.images import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> gsids = ibs._get_all_imgsetids()
        >>> self = ImageSets(gsids, ibs)
        >>> print(self)
        <ImageSets(num=13)>

    """

    def __init__(self, gsids, ibs, config=None):
        super(ImageSets, self).__init__(gsids, ibs, config)

    @property
    def images(self):
        return [self._ibs.images(gids) for gids in self.gids]

    @property
    def annots(self):
        return [self._ibs.annots(aids) for aids in self.aids]


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.images
        python -m wbia.images --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
