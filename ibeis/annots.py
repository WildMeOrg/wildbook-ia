# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
import itertools as it
from ibeis import _ibeis_object
from ibeis.control.controller_inject import make_ibs_register_decorator
(print, rrr, profile) = ut.inject2(__name__, '[annot]')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


@register_ibs_method
def annots(ibs, aids=None, config=None):
    if aids is None:
        aids = ibs.get_valid_aids()
    return Annots(aids, ibs, config)


BASE_TYPE = type


class AnnotIBIESPropertyInjector(BASE_TYPE):
    def __init__(metaself, name, bases, dct):
        super(AnnotIBIESPropertyInjector, metaself).__init__(name, bases, dct)
        metaself.rrr = rrr
        attrs = [
            'aid',
            'parent_aid',

            'uuids', 'hashid_uuid', 'visual_uuids', 'hashid_visual_uuid',
            'semantic_uuids', 'hashid_semantic_uuid', 'verts', 'thetas',
            'species_uuids', 'species', 'species_rowids', 'species_texts',
            'yaw_texts', 'yaws', 'qualities', 'quality_texts',
            'exemplar_flags',
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
            'all_tags', 'case_tags', 'annotmatch_tags', 'notes',
            # Processing State
            'reviewed', 'reviewed_matching_aids', 'has_reviewed_matching_aids',
            'num_reviewed_matching_aids', 'detect_confidence',
        ]

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

        objname = 'annot'
        _ibeis_object._inject_getter_attrs(metaself, objname, attrs, configurable_attrs)


@ut.reloadable_class
@six.add_metaclass(AnnotIBIESPropertyInjector)
class Annots(_ibeis_object.PrimaryObject):
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
    def __init__(self, aids, ibs, config=None):
        super(Annots, self).__init__(aids, ibs, config)

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
        ams = ibs.get_annotmatch_rowid_from_undirected_superkey(*zip(*edges))
        ams = ut.filter_Nones(ams)
        return ams

    def get_am_aidpairs(self):
        ibs = self._ibs
        ams = self.get_am_rowids()
        aids1 = ibs.get_annotmatch_aid1(ams)
        aids2 = ibs.get_annotmatch_aid2(ams)
        aid_pairs = list(zip(aids1, aids2))
        return aid_pairs


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
