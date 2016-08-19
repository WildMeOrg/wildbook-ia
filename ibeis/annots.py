# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
import itertools as it
from ibeis import _ibeis_object
from ibeis.control.controller_inject import make_ibs_register_decorator
(print, rrr, profile) = ut.inject2(__name__, '[annot]')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


BASE_TYPE = type


@register_ibs_method
def annots(ibs, aids=None, **kwargs):
    """ Makes an Annots object """
    if aids is None:
        aids = ibs.get_valid_aids()
    elif aids.__class__.__name__ == 'Annots':
        return aids
    aids = ut.ensure_iterable(aids)
    return Annots(aids, ibs, **kwargs)


@register_ibs_method
def _annot_groups(ibs, aids_list=None, config=None):
    annots_list = [ibs.annots(aids, config=config) for aids in aids_list]
    return AnnotGroups(annots_list, ibs)


ANNOT_BASE_ATTRS = [
    'aid',
    'parent_aid',

    'uuids', 'hashid_uuid', 'visual_uuids', 'hashid_visual_uuid',
    'semantic_uuids', 'hashid_semantic_uuid', 'verts', 'thetas',
    'bboxes', 'bbox_area',
    'species_uuids', 'species', 'species_rowids', 'species_texts', 'yaw_texts',
    'yaws', 'qualities', 'quality_texts', 'exemplar_flags',
    # Images
    # 'image_rowids',
    'gids', 'image_uuids', 'image_gps', 'image_unixtimes_asfloat',
    'image_datetime_str', 'image_contributor_tag',
    # Names
    'nids', 'names', 'name_uuids',
    # Inferred from context attrs
    'contact_aids', 'num_contact_aids', 'groundfalse', 'groundtruth',
    'num_groundtruth', 'has_groundtruth', 'otherimage_aids',
    # Image Set
    'imgset_uuids', 'imgsetids', 'image_set_texts',
    # Occurrence / Encounter
    'encounter_text', 'occurrence_text', 'primary_imageset',
    # Tags
    'all_tags',
    'case_tags',
    'annotmatch_tags', 'notes',
    # Processing State
    'reviewed', 'reviewed_matching_aids', 'has_reviewed_matching_aids',
    'num_reviewed_matching_aids', 'detect_confidence',
]

ANNOT_SETTABLE_ATTRS = [
    'age_months_est_max', 'age_months_est_min',
    'bboxes', 'thetas', 'verts',
    'quality_texts', 'yaw_texts', 'yaws'
    'sex', 'sex_texts', 'species',
    'exemplar_flags',
    'multiple',
    'detect_confidence', 'reviewed',
    'name_texts', 'names',
    'notes',
    'parent_rowid',
]


class _AnnotPropInjector(BASE_TYPE):
    """
    Example:
        >>> from ibeis import _ibeis_object
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> objname = 'annot'
        >>> blacklist = ['annot_pair']
        >>> _ibeis_object._find_ibeis_attrs(ibs, objname, blacklist)
    """
    def __init__(metaself, name, bases, dct):
        super(_AnnotPropInjector, metaself).__init__(name, bases, dct)
        metaself.rrr = rrr

        attrs = ANNOT_BASE_ATTRS

        settable_attrs = ANNOT_SETTABLE_ATTRS

        configurable_attrs = [
            # Chip
            'chip_dlensqrd', 'chip_fpath', 'chip_sizes', 'chip_thumbpath',
            'chip_thumbtup', 'chips',
            # Feat / FeatWeight / Kpts / Desc
            'feat_rowids', 'num_feats', 'featweight_rowids', 'fgweights',
            'fgweights_subset', 'kpts', 'kpts_distinctiveness', 'vecs',
            'vecs_cache', 'vecs_subset',
        ]
        #misc = [
        #    'gar_rowids', 'alrids', 'alrids_oftype', 'lblannot_rowids',
        #    'lblannot_rowids_oftype', 'lblannot_value_of_lbltype', 'rows',
        #    'instancelist', 'lazy_dict', 'lazy_dict2', 'missing_uuid',
        #    'been_adjusted', 'class_labels',
        #]
        #extra_attrs = [
        #    # Age / Sex
        #    'age_months_est', 'age_months_est_max', 'age_months_est_max_texts',
        #    'age_months_est_min', 'age_months_est_min_texts',
        #    'age_months_est_texts', 'sex', 'sex_texts',

        #    # Stats
        #    'stats_dict', 'per_name_stats', 'qual_stats', 'info', 'yaw_stats',
        #    'intermediate_viewpoint_stats',
        #]
        #inverse_attrs = [
        #    # External lookups via superkeys
        #    'aids_from_semantic_uuid',
        #    'aids_from_uuid',
        #    'aids_from_visual_uuid',
        #    'rowids_from_partial_vuuids',
        #]

        depcache_attrs = [
            ('hog', 'hog'),
        ]

        objname = 'annot'
        _ibeis_object._inject_getter_attrs(metaself, objname, attrs,
                                           configurable_attrs, 'depc_annot',
                                           depcache_attrs, settable_attrs)

        # TODO: incorporate dynamic setters
        #def set_case_tags(self, tags):
        #    self._ibs.append_annot_case_tags(self._rowids, tags)
        #fget = metaself.case_tags.fget
        #fset = set_case_tags
        #setattr(metaself, 'case_tags', property(fget, fset))


@ut.reloadable_class
@six.add_metaclass(_AnnotPropInjector)
class Annots(_ibeis_object.ObjectList1D):
    """
    Represents a group of annotations. Efficiently accesses properties from a
    database using lazy evaluation.

    CommandLine:
        python -m ibeis.annots Annots --show

    Example:
        >>> from ibeis.annots import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids()
        >>> a = self = annots = Annots(aids, ibs)
        >>> a.preload('vecs', 'kpts', 'nids')
        >>> print(ut.depth_profile(a.vecs))
        >>> print(a)
        <Annots(num=13)>

    """
    #def __init__(self, aids, ibs, config=None, caching=False):
    #    super(Annots, self).__init__(aids, ibs, config, caching)

    @property
    def aids(self):
        return self._rowids

    #@property
    def get_speeds(self):
        #import vtool as vt
        edges = self.get_aidpairs()
        speeds = self._ibs.get_annotpair_speeds(edges)
        #edges = vt.pdist_indicies(len(annots))
        #speeds = self._ibs.get_unflat_annots_speeds_list([self.aids])[0]
        edge_to_speed = dict(zip(edges, speeds))
        return edge_to_speed

    def get_aidpairs(self):
        aids = self.aids
        aid_pairs = list(it.combinations(aids, 2))
        return aid_pairs

    def get_am_rowids(self):
        ibs = self._ibs
        edges = self.get_aidpairs()
        aids1 = ut.take_column(edges, 0)
        aids2 = ut.take_column(edges, 1)
        ams = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
        ams = ut.filter_Nones(ams)
        return ams

    def get_am_rowids_and_pairs(self):
        ibs = self._ibs
        aid_pairs = self.get_aidpairs()
        aids1 = ut.take_column(aid_pairs, 0)
        aids2 = ut.take_column(aid_pairs, 1)
        ams = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
        flags = ut.not_list(ut.flag_None_items(ams))
        ams = ut.compress(ams, flags)
        aid_pairs = ut.compress(aid_pairs, flags)
        return ams, aid_pairs

    def get_am_aidpairs(self):
        ibs = self._ibs
        ams = self.get_am_rowids()
        aids1 = ibs.get_annotmatch_aid1(ams)
        aids2 = ibs.get_annotmatch_aid2(ams)
        aid_pairs = list(zip(aids1, aids2))
        return aid_pairs

    @property
    def hog_img(self):
        from ibeis import core_annots
        return [core_annots.make_hog_block_image(hog) for hog in self.hog_hog]

    def append_tags(self, tags):
        self._ibs.append_annot_case_tags(self._rowids, tags)

    def remove_tags(self, tags):
        self._ibs.remove_annot_case_tags(self._rowids, tags)


class _AnnotGroupPropInjector(BASE_TYPE):
    def __init__(metaself, name, bases, dct):
        super(_AnnotGroupPropInjector, metaself).__init__(name, bases, dct)
        metaself.rrr = rrr

        #attrs = ANNOT_BASE_ATTRS
        #objname = 'annot'

        # TODO: move to ibeis object as a group call
        def _make_unflat_getter(objname, attrname):
            ibs_funcname = 'get_%s_%s' % (objname, attrname)
            def ibs_unflat_getter(self, *args, **kwargs):
                ibs_callable = getattr(self._ibs, ibs_funcname)
                rowids = self._rowids_list
                ibs = self._ibs
                return ibs.unflat_map(ibs_callable, rowids, *args, **kwargs)
            ut.set_funcname(ibs_unflat_getter, 'unflat_' + ibs_funcname)
            return ibs_unflat_getter

        for attrname in ANNOT_BASE_ATTRS:
            if hasattr(metaself, attrname):
                print('Cannot inject annot group attrname = %r' % (attrname,))
                continue
            ibs_unflat_getter = _make_unflat_getter('annot', attrname)
            setattr(metaself, '_unflat_get_' + attrname, ibs_unflat_getter)
            setattr(metaself, attrname, property(ibs_unflat_getter))


@ut.reloadable_class
@six.add_metaclass(_AnnotGroupPropInjector)
class AnnotGroups(ut.NiceRepr):
    """ Effciently handle operations on multiple groups of annotations """
    def __init__(self, annots_list, ibs):
        self._ibs = ibs
        self.annots_list = annots_list
        self._rowids_list = [a._rowids for a in self.annots_list]

    def __nice__(self):
        import numpy as np
        len_list = ut.lmap(len, self.annots_list)
        num = len(self.annots_list)
        mean = np.mean(len_list)
        std = np.std(len_list)
        if six.PY3:
            nice = '(n=%r, μ=%.1f, σ=%.1f)' % (num, mean, std)
        else:
            nice = '(n=%r, m=%.1f, s=%.1f)' % (num, mean, std)
        return nice

    @property
    def aids(self):
        return [a.aids for a in self.annots_list]

    @property
    def images(self, config=None):
        return self._ibs.images(self.gids, config)

    @property
    def match_tags(self):
        """ returns pairwise tags within the annotation group """
        ams_list = self._ibs.get_unflat_am_rowids(self.aids)
        tags = self._ibs.unflat_map(self._ibs.get_annotmatch_case_tags, ams_list)
        return tags


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.annot
        python -m ibeis.annot --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
