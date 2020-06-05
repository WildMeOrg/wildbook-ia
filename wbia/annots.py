# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
import itertools as it
from wbia import _wbia_object
from wbia.control.controller_inject import make_ibs_register_decorator

(print, rrr, profile) = ut.inject2(__name__, '[annot]')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


BASE_TYPE = type


@register_ibs_method
def annots(ibs, aids=None, uuids=None, **kwargs):
    """ Makes an Annots object """
    if uuids is not None:
        assert aids is None, 'specify one primary key'
        aids = ibs.get_annot_aids_from_uuid(uuids)
    if aids is None:
        aids = ibs.get_valid_aids()
    elif aids.__class__.__name__ == 'Annots':
        return aids
    aids = ut.ensure_iterable(aids)
    return Annots(aids, ibs, **kwargs)


@register_ibs_method
def matches(ibs, ams=None, edges=None, uuid_edges=None, **kwargs):
    """ Makes an Annots object """
    if uuid_edges is not None:
        assert ams is None, 'specify one primary key'
        assert edges is None, 'specify one primary key'
        uuids1, uuids2 = list(zip(*uuid_edges))
        aids1 = ibs.get_annot_aids_from_uuid(uuids1)
        aids2 = ibs.get_annot_aids_from_uuid(uuids2)
        ams = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
    if edges is not None:
        assert ams is None, 'specify one primary key'
        assert uuid_edges is None, 'specify one primary key'
        aids1, aids2 = list(zip(*edges))
        ams = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
    if ams is None:
        ams = ibs._get_all_annotmatch_rowids()
    elif ams.__class__.__name__ == 'AnnotMatches':
        return ams
    ams = ut.ensure_iterable(ams)
    return AnnotMatches(ams, ibs, **kwargs)


@register_ibs_method
def _annot_groups(ibs, aids_list=None, config=None):
    annots_list = [ibs.annots(aids, config=config) for aids in aids_list]
    return AnnotGroups(annots_list, ibs)


ANNOT_BASE_ATTRS = [
    'aid',
    'parent_aid',
    'multiple',
    'age_months_est_max',
    'age_months_est_min',
    'sex',
    'sex_texts',
    'uuids',
    'hashid_uuid',
    'visual_uuids',
    'hashid_visual_uuid',
    'semantic_uuids',
    'hashid_semantic_uuid',
    'verts',
    'thetas',
    'bboxes',
    'bbox_area',
    'species_uuids',
    'species',
    'species_rowids',
    'species_texts',
    'viewpoint_int',
    'viewpoint_code',
    'qualities',
    'quality_texts',
    'exemplar_flags',
    # DEPRICATE YAW
    'yaw_texts',
    'yaws',
    'yaws_asfloat',
    # Images
    # 'image_rowids',
    'gids',
    'image_uuids',
    'image_gps',
    'image_gps2',
    'image_unixtimes_asfloat',
    'image_datetime_str',
    'image_contributor_tag',
    # Names
    'nids',
    'names',
    'name_uuids',
    # Inferred from context attrs
    'contact_aids',
    'num_contact_aids',
    'groundfalse',
    'groundtruth',
    'num_groundtruth',
    'has_groundtruth',
    'otherimage_aids',
    # Image Set
    'imgset_uuids',
    'imgsetids',
    'image_set_texts',
    # Occurrence / Encounter
    'static_encounter',
    'encounter_text',
    'occurrence_text',
    'primary_imageset',
    # Tags
    'all_tags',
    'case_tags',
    'annotmatch_tags',
    'notes',
    # Processing State
    'reviewed',
    'reviewed_matching_aids',
    'has_reviewed_matching_aids',
    'num_reviewed_matching_aids',
    'detect_confidence',
]

ANNOT_SETTABLE_ATTRS = [
    'age_months_est_max',
    'age_months_est_min',
    'bboxes',
    'thetas',
    'verts',
    'qualities',
    'quality_texts',
    'viewpoint_int',
    'viewpoint_code',
    # DEPRICATE YAW
    'yaw_texts',
    'yaws',
    'sex',
    'sex_texts',
    'species',
    'exemplar_flags',
    'static_encounter',
    'multiple',
    'case_tags',
    'detect_confidence',
    'reviewed',
    'name_texts',
    'names',
    'notes',
    'parent_rowid',
]


class _AnnotPropInjector(BASE_TYPE):
    """
    Ignore:
        >>> from wbia import _wbia_object
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> objname = 'annot'
        >>> blacklist = ['annot_pair']
        >>> _wbia_object._find_wbia_attrs(ibs, objname, blacklist)
    """

    def __init__(metaself, name, bases, dct):
        super(_AnnotPropInjector, metaself).__init__(name, bases, dct)
        metaself.rrr = rrr

        attrs = ANNOT_BASE_ATTRS

        settable_attrs = ANNOT_SETTABLE_ATTRS

        configurable_attrs = [
            # Chip
            'chip_dlensqrd',
            'chip_fpath',
            'chip_sizes',
            'chip_thumbpath',
            'chip_thumbtup',
            'chips',
            # Feat / FeatWeight / Kpts / Desc
            'feat_rowids',
            'num_feats',
            'featweight_rowids',
            'fgweights',
            'fgweights_subset',
            'kpts',
            'kpts_distinctiveness',
            'vecs',
            'vecs_cache',
            'vecs_subset',
        ]
        # misc = [
        #    'gar_rowids', 'alrids', 'alrids_oftype', 'lblannot_rowids',
        #    'lblannot_rowids_oftype', 'lblannot_value_of_lbltype', 'rows',
        #    'instancelist', 'lazy_dict', 'lazy_dict2', 'missing_uuid',
        #    'been_adjusted', 'class_labels',
        # ]
        # extra_attrs = [
        #    # Age / Sex
        #    'age_months_est', 'age_months_est_max', 'age_months_est_max_texts',
        #    'age_months_est_min', 'age_months_est_min_texts',
        #    'age_months_est_texts', 'sex', 'sex_texts',

        #    # Stats
        #    'stats_dict', 'per_name_stats', 'qual_stats', 'info', 'yaw_stats',
        #    'intermediate_viewpoint_stats',
        # ]
        # inverse_attrs = [
        #    # External lookups via superkeys
        #    'aids_from_semantic_uuid',
        #    'aids_from_uuid',
        #    'aids_from_visual_uuid',
        #    'rowids_from_partial_vuuids',
        # ]

        depcache_attrs = [
            ('hog', 'hog'),
            ('probchip', 'img'),
        ]

        aliased_attrs = {
            'time': 'image_unixtimes_asfloat',
            'gps': 'image_gps2',
            'chip_size': 'chip_sizes',
            'yaw': 'yaws_asfloat',
            'qual': 'qualities',
            'name': 'names',
            'nid': 'nids',
            'unary_tags': 'case_tags',
            # DEPRICATE
            'rchip': 'chips',
            'rchip_fpath': 'chip_fpath',
        }

        objname = 'annot'
        _wbia_object._inject_getter_attrs(
            metaself,
            objname,
            attrs,
            configurable_attrs,
            'depc_annot',
            depcache_attrs,
            settable_attrs,
            aliased_attrs,
        )

        # TODO: incorporate dynamic setters
        # def set_case_tags(self, tags):
        #    self._ibs.append_annot_case_tags(self._rowids, tags)
        # fget = metaself.case_tags.fget
        # fset = set_case_tags
        # setattr(metaself, 'case_tags', property(fget, fset))


try:
    from wbia import _autogen_annot_base

    BASE = _autogen_annot_base._annot_base_class
except ImportError:
    BASE = _wbia_object.ObjectList1D


# @ut.reloadable_class
@six.add_metaclass(_AnnotPropInjector)
class Annots(BASE):
    """
    Represents a group of annotations. Efficiently accesses properties from a
    database using lazy evaluation.

    CommandLine:
        python -m wbia.annots Annots

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.annots import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids()
        >>> a = self = annots = Annots(aids, ibs)
        >>> a.preload('vecs', 'kpts', 'nids')
        >>> print(Annots.mro())
        >>> print(ut.depth_profile(a.vecs))
        >>> print(a)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.annots import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids()
        >>> a = self = annots = Annots(aids, ibs)
        >>> a.preload('vecs', 'kpts', 'nids')
        >>> a.disconnect()
        >>> assert 'vecs' in a._internal_attrs.keys()
        >>> assert a._ibs is None
        >>> ut.assert_raises(KeyError, a._get_num_feats)
        >>> a._ibs = ibs
        >>> assert len(a._get_num_feats()) > 0
    """

    # def __init__(self, aids, ibs, config=None, caching=False):
    #    super(Annots, self).__init__(aids, ibs, config, caching)

    @property
    def aids(self):
        return self._rowids

    def get_stats(self, **kwargs):
        self._ibs.get_annot_stats_dict(self.aids, **kwargs)

    def print_stats(self, **kwargs):
        self._ibs.print_annot_stats(self.aids, **kwargs)

    # @property
    def get_speeds(self):
        # import vtool as vt
        edges = self.get_aidpairs()
        speeds = self._ibs.get_annotpair_speeds(edges)
        # edges = vt.pdist_indicies(len(annots))
        # speeds = self._ibs.get_unflat_annots_speeds_list([self.aids])[0]
        edge_to_speed = dict(zip(edges, speeds))
        return edge_to_speed

    def get_name_image_closure(self):
        ibs = self._ibs
        aids = self.aids
        old_aids = []
        while len(old_aids) != len(aids):
            old_aids = aids
            gids = ut.unique(ibs.get_annot_gids(aids))
            other_aids = list(set(ut.flatten(ibs.get_image_aids(gids))))
            other_nids = list(set(ibs.get_annot_nids(other_aids)))
            aids = ut.flatten(ibs.get_name_aids(other_nids))
        return aids

    def group2(self, by):
        """
        self = annots
        by = annots.static_encounter
        encounters = annots.group2(annots.static_encounter)
        """
        annots_list = self.group(by)[1]
        return AnnotGroups(annots_list, self._ibs)

    def get_aidpairs(self):
        aids = self.aids
        aid_pairs = list(it.combinations(aids, 2))
        return aid_pairs

    def get_am_rowids(self, internal=True):
        """
        if `internal is True` returns am rowids only between
        annotations in this Annots object, otherwise returns
        any am rowid that contains any aid in this Annots object.
        """
        ibs = self._ibs
        if internal:
            ams = ibs.get_annotmatch_rowids_between(self.aids, self.aids)
        else:
            ams = ut.flatten(ibs.get_annotmatch_rowids_from_aid(self.aids))
        return ams

    def matches(self, internal=True):
        ams = self.get_am_rowids(internal)
        return self._ibs.matches(ams)

    def get_am_rowids_and_pairs(self):
        ibs = self._ibs
        ams = self.get_am_rowids()
        aid_pairs = ibs.get_annotmatch_aids(ams)
        # aid_pairs = self.get_aidpairs()
        # aids1 = ut.take_column(aid_pairs, 0)
        # aids2 = ut.take_column(aid_pairs, 1)
        # ams = ibs.get_annotmatch_rowid_from_undirected_superkey(aids1, aids2)
        # flags = ut.not_list(ut.flag_None_items(ams))
        # ams = ut.compress(ams, flags)
        # aid_pairs = ut.compress(aid_pairs, flags)
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
        from wbia import core_annots

        return [core_annots.make_hog_block_image(hog) for hog in self.hog_hog]

    def append_tags(self, tags):
        self._ibs.append_annot_case_tags(self._rowids, tags)

    def remove_tags(self, tags):
        self._ibs.remove_annot_case_tags(self._rowids, tags)

    def __hash__(self):
        return hash(tuple(self.aids))

    def __lt__(self, other):
        if len(self.aids) == len(other.aids):
            if len(self.aids) == 0:
                return False
            else:
                return tuple(self) < tuple(other)
        elif len(self.aids) < len(other.aids):
            return True
        else:
            return False

    def __eq__(self, other):
        if len(self.aids) == len(other.aids):
            return all(a == b for a, b in zip(self, other))
        return False

    def show(self, *args, **kwargs):
        if len(self) != 1:
            raise ValueError('Can only show one, got {}'.format(len(self)))
        from wbia.viz import viz_chip

        for aid in self:
            return viz_chip.show_chip(self._ibs, aid, *args, **kwargs)


class _AnnotGroupPropInjector(BASE_TYPE):
    def __init__(metaself, name, bases, dct):
        super(_AnnotGroupPropInjector, metaself).__init__(name, bases, dct)
        metaself.rrr = rrr

        # TODO: move to wbia object as a group call
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

    def __len__(self):
        return len(self.annots_list)

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

    def __iter__(self):
        return iter(self.annots_list)

    def __getitem__(self, index):
        return self.annots_list[index]

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


class _AnnotMatchPropInjector(BASE_TYPE):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia import _wbia_object
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> objname = 'annotmatch'
        >>> blacklist = []
        >>> tup = _wbia_object._find_wbia_attrs(ibs, objname, blacklist)
        >>> attrs, settable_attrs = tup
        >>> print('attrs = ' + ut.repr4(attrs))
        >>> print('settable_attrs = ' + ut.repr4(settable_attrs))
    """

    def __init__(metaself, name, bases, dct):
        super(_AnnotMatchPropInjector, metaself).__init__(name, bases, dct)
        metaself.rrr = rrr

        attrs = [
            'aid1',
            'aid2',
            'confidence',
            'count',
            'evidence_decision',
            'meta_decision',
            'posixtime_modified',
            'reviewer',
            'tag_text',
            'case_tags',
        ]
        settable_attrs = [
            'confidence',
            'count',
            'evidence_decision',
            'meta_decision',
            'posixtime_modified',
            'reviewer',
            'tag_text',
        ]

        configurable_attrs = []
        depcache_attrs = []
        aliased_attrs = {}

        objname = 'annotmatch'
        _wbia_object._inject_getter_attrs(
            metaself,
            objname,
            attrs,
            configurable_attrs,
            None,
            depcache_attrs,
            settable_attrs,
            aliased_attrs,
        )


@six.add_metaclass(_AnnotMatchPropInjector)
class AnnotMatches(BASE):
    """
    Represents a group of annotations. Efficiently accesses properties from a
    database using lazy evaluation.

    CommandLine:
        python -m wbia.annots Annots

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.annots import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids()
        >>> annots = Annots(aids, ibs)
        >>> ams = annots.get_am_rowids()
        >>> matches = self = ibs.matches()
        >>> ed1 = matches.evidence_decision
        >>> md2 = matches.meta_decision
        >>> table = ibs.db.get_table_as_pandas('annotmatch')
        >>> assert len(table) == len(matches)
    """

    @property
    def edges(self):
        return list(zip(self.aid1, self.aid2))

    @property
    def confidence_code(self):
        INT_TO_CODE = self._ibs.const.CONFIDENCE.INT_TO_CODE
        return [INT_TO_CODE[c] for c in self.confidence]

    @property
    def meta_decision_code(self):
        INT_TO_CODE = self._ibs.const.META_DECISION.INT_TO_CODE
        return [INT_TO_CODE[c] for c in self.meta_decision]

    @property
    def evidence_decision_code(self):
        INT_TO_CODE = self._ibs.const.EVIDENCE_DECISION.INT_TO_CODE
        return [INT_TO_CODE[c] for c in self.evidence_decision]


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.annot
        python -m wbia.annot --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
