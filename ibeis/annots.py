# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
from ibeis.control.controller_inject import make_ibs_register_decorator
(print, rrr, profile) = ut.inject2(__name__, '[annot]')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


def _find_annot_attrs(ibs):
    r"""
    Args:
        ibs (ibeis.IBEISController):  image analysis api

    CommandLine:
        python -m ibeis.annot _find_annot_attrs --show

    Example:
        >>> from ibeis.annot import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> _find_annot_attrs(ibs)
    """
    ut.search_module(ibs, 'get')
    import re
    found_funcnames = ut.search_module(ibs, 'get_annot_')
    pat = 'get_annot_' + ut.named_field('attr', '.*')
    found_funcnames = [fn for fn in found_funcnames if 'annot_pair' not in fn]
    matched_attrs = [re.match(pat, fn).groupdict()['attr'] for fn in found_funcnames]
    invalid_attrs = [
    ]
    misc = [
        'gar_rowids',
        'alrids', 'alrids_oftype',
        'lblannot_rowids', 'lblannot_rowids_oftype', 'lblannot_value_of_lbltype',
        'rows',
        'instancelist',
        'lazy_dict', 'lazy_dict2',
        'missing_uuid',
        'been_adjusted',
        'class_labels',
    ]

    core_attrs = [
        # Self
        'aid',

        # Parent
        'parent_aid',

        # UUIDS
        'uuids', 'hashid_uuid',
        'visual_uuids', 'hashid_visual_uuid',  'visual_uuid_info',
        'semantic_uuids', 'hashid_semantic_uuid', 'semantic_uuid_info',

        # Bounding Box
        'bbox_area', 'bboxes', 'verts', 'thetas', 'rotated_verts', 'num_verts',

        # Species
        'species_uuids', 'species', 'species_rowids', 'species_texts',

        # Viewpoint
        'yaw_texts', 'yaws', 'yaws_asfloat',

        # Quality
        'qualities', 'quality_texts',

        # Exemplar
        'exemplar_flags',
    ]

    unsorted_attrs = [
         # Images
        'gids',
        'image_rowids',
        'image_uuids',
        'images',
        'image_names',
        'image_paths',
        'image_gps',
        'image_unixtimes', 'image_unixtimes_asfloat', 'image_datetime_str',
        'image_contributor_tag',

        # Names
        'name_rowids', 'name_texts', 'name_uuids', 'names', 'nids',

        # Inferred from context attrs
        'contact_aids', 'num_contact_aids',
        'groundfalse', 'groundtruth', 'num_groundtruth', 'has_groundtruth',
        'otherimage_aids',

        # Image Set
        'imgset_uuids', 'imgsetids', 'image_set_texts',

        # Occurrence / Encounter
        'encounter_text', 'occurrence_text', 'primary_imageset',

        # Tags
        'all_tags', 'tag_text',
        'case_tags',
        'is_hard', 'isjunk', 'multiple',
        'tag_filterflags',
        'notes', 'annotmatch_tags',

        # Processing State
        'reviewed',
        'reviewed_matching_aids',
        'has_reviewed_matching_aids',
        'num_reviewed_matching_aids',
        'detect_confidence',
    ]


    depcache_attrs = [
        # Chip
        'chip_dlensqrd', 'chip_fpath', 'chip_sizes', 'chip_thumbpath',
        'chip_thumbtup', 'chips',

        # Feat / FeatWeight / Kpts / Desc
        'feat_rowids', 'num_feats',
        'featweight_rowids', 'fgweights', 'fgweights_subset',
        'kpts', 'kpts_distinctiveness',
        'vecs', 'vecs_cache', 'vecs_subset',
    ]

    extra_attrs = [
        # Age / Sex
        'age_months_est', 'age_months_est_max', 'age_months_est_max_texts',
        'age_months_est_min', 'age_months_est_min_texts',
        'age_months_est_texts', 'sex', 'sex_texts',

        # Stats
        'stats_dict', 'per_name_stats', 'qual_stats', 'info', 'yaw_stats',
        'intermediate_viewpoint_stats',
    ]

    inverse_attrs = [
        # External lookups via superkeys
        'aids_from_semantic_uuid',
        'aids_from_uuid',
        'aids_from_visual_uuid',
        'rowids_from_partial_vuuids',
    ]

    valid_attrs = [
    ]
    # print('matched_attrs = %s' % (ut.repr3(matched_attrs),))

    pass


@register_ibs_method
def annots(ibs, aids=None):
    from ibeis import annot
    if aids is None:
        aids = ibs.get_valid_aids()
    return Annots(aids, ibs)


@ut.reloadable_class
class Annots(ut.NiceRepr):
    """
    Represents a group of annotations. Efficiently accesses properties from a
    database using lazy evaluation.

    Example:
        >>> from ibeis.annot import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids()
        >>> a = self = annots = Annots(aids, ibs)
        >>> print(a)
        <Annots(_UUIDS((13)t%%xwy%xcnw0h&2a))>

    """
    def __init__(self, aids, ibs):
        self.ibs = ibs
        self.aids = aids
        self._islist = True
        self.ibs = ibs
        self._on_reload()

    def _on_reload(self):
        attrs = [
            # 'aid',
            # 'parent_aid',
            'uuids', 'hashid_uuid',
            'visual_uuids', 'hashid_visual_uuid',
            'semantic_uuids', 'hashid_semantic_uuid',

            'verts', 'thetas',

            'species_uuids', 'species', 'species_rowids', 'species_texts',

            'yaw_texts', 'yaws',
            'qualities', 'quality_texts',

            'exemplar_flags',

             # Images
            # 'image_rowids',
            'gids',
            'image_uuids',
            'image_gps',
            'image_unixtimes_asfloat',
            'image_datetime_str',
            'image_contributor_tag',

            # Names
            'nids',
            'names',
            'name_uuids',

            # Inferred from context attrs
            'contact_aids', 'num_contact_aids',
            'groundfalse', 'groundtruth', 'num_groundtruth', 'has_groundtruth',
            'otherimage_aids',

            # Image Set
            'imgset_uuids', 'imgsetids', 'image_set_texts',

            # Occurrence / Encounter
            'encounter_text', 'occurrence_text', 'primary_imageset',

            # Tags
            'all_tags', 'case_tags', 'annotmatch_tags',
            'notes',

            # Processing State
            'reviewed',
            'reviewed_matching_aids',
            'has_reviewed_matching_aids',
            'num_reviewed_matching_aids',
            'detect_confidence',

            # Chip
            'chip_dlensqrd', 'chip_fpath', 'chip_sizes', 'chip_thumbpath',
            'chip_thumbtup', 'chips',

            # Feat / FeatWeight / Kpts / Desc
            'feat_rowids', 'num_feats',
            'featweight_rowids', 'fgweights', 'fgweights_subset',
            'kpts', 'kpts_distinctiveness',
            'vecs',
            'vecs_cache', 'vecs_subset',
        ]
        from ibeis import annot
        def make_annot_inject(self, ibs_attr):
            def _wrapped_method(self, *args, **kwargs):
                return ibs_attr(self.aids, *args, **kwargs)
            return _wrapped_method
        for attr in attrs:
            ibs_attr = getattr(self.ibs, 'get_annot_' + attr)
            ibs_getter = make_annot_inject(self, ibs_attr)
            ut.inject_func_as_method(self, ibs_getter, method_name='_get_' + attr, allow_override=True)
            setattr(annot.Annots, attr, property(ibs_getter))
            # ut.inject_func_as_property(self, ibs_getter, method_name=attr, allow_override=True)

    # def filter(self, filterkw):
    #     pass

    # def filter_flags(self, filterkw):
    #     pass

    def take(self, idxs):
        return annot.Annots(ut.take(self.aids, idxs), self.ibs)

    def __iter__(self):
        return iter(self.aids)

    def chunks(self,  chunksize):
        from ibeis import annot
        return (annot.Annots(aids, self.ibs) for aids in ut.ichunks(self, chunksize))

    def compress(self,  flags):
        from ibeis import annot
        return annot.Annots(ut.compress(self.aids, flags), self.ibs)

    def groupby(self, labels):
        unique_labels, groupxs = ut.group_indices(labels)
        annot_groups = [self.take(idxs) for idxs in groupxs]
        return annot_groups

    def __nice__(self):
        if self._islist:
            # try:
            #     return '(%s)' % (self._get_hashid_uuid(),)
            # except Exception:
            return '(num=%r)' % (len(self.aids))
        else:
            return '(single)'


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
