# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
from six.moves import range
(print, rrr, profile) = ut.inject2(__name__, '[_ibeis_object]')


def _find_ibeis_attrs(ibs, objname, blacklist=[]):
    r"""
    Args:
        ibs (ibeis.IBEISController):  images analysis api

    CommandLine:
        python -m ibeis.images _find_ibeis_attrs --show

    Example:
        >>> from ibeis._ibeis_object import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> objname = 'images'
        >>> blacklist = []
        >>> _find_ibeis_attrs(ibs, objname, blacklist)

    Example:
        >>> from ibeis._ibeis_object import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> objname = 'annots'
        >>> blacklist = ['annot_pair']
        >>> _find_ibeis_attrs(ibs, objname, blacklist)
    """
    import re
    getter_prefix = 'get_' + objname + '_'
    found_funcnames = ut.search_module(ibs, getter_prefix)
    pat = getter_prefix + ut.named_field('attr', '.*')
    for stopword in blacklist:
        found_funcnames = [fn for fn in found_funcnames if stopword not in fn]
    matched_attrs = [re.match(pat, fn).groupdict()['attr'] for fn in found_funcnames]
    return matched_attrs


def _inject_getter_attrs(metaself, objname, attrs, configurable_attrs,
                         depc_name=None, depcache_attrs=None):
    """
    for use in the metaclass
    """

    def _make_getter(objname, attrname):
        ibs_funcname = 'get_%s_%s' % (objname, attrname)
        def ibs_getter(self, *args, **kwargs):
            if self._ibs is None:
                return self._internal_attrs[attrname]
            else:
                ibs_callable = getattr(self._ibs, ibs_funcname)
                return ibs_callable(self._rowids, *args, **kwargs)
        ut.set_funcname(ibs_getter, ibs_funcname)
        return ibs_getter

    def _make_configurable_getter(objname, attrname):
        ibs_funcname = 'get_%s_%s' % (objname, attrname)
        def ibs_cfg_getter(self):
            if self._ibs is None:
                return self._internal_attrs[attrname]
            else:
                ibs_callable = getattr(self._ibs, ibs_funcname)
                return ibs_callable(self._rowids, config2_=self._config)
        ut.set_funcname(ibs_cfg_getter, ibs_funcname)
        return ibs_cfg_getter

    def _make_depcache_getter(depc_name, tbl, col):
        attrname = '%s_%s' % (tbl, col)
        def ibs_cfg_getter(self):
            if self._ibs is None:
                return self._internal_attrs[attrname]
            else:
                depc = getattr(self._ibs, depc_name)
                return depc.get(tbl, self._rowids, col, config=self._config)
        ut.set_funcname(ibs_cfg_getter, 'get_' + attrname)
        return ibs_cfg_getter

    # Inject function and property version
    for attrname in attrs:
        ibs_getter = _make_getter(objname, attrname)
        setattr(metaself, '_get_' + attrname, ibs_getter)
        setattr(metaself, attrname, property(ibs_getter))

    for attrname in configurable_attrs:
        ibs_cfg_getter = _make_configurable_getter(objname, attrname)
        setattr(metaself, '_get_' + attrname, ibs_cfg_getter)
        setattr(metaself, attrname, property(ibs_cfg_getter))

    if depcache_attrs is not None:
        for tbl, col in depcache_attrs:
            attrname = '%s_%s' % (tbl, col)
            ibs_depc_getter = _make_depcache_getter(depc_name, tbl, col)
            setattr(metaself, '_get_' + attrname, ibs_depc_getter)
            setattr(metaself, attrname, property(ibs_depc_getter))
        #import utool
        #utool.embed()


#@ut.reloadable_class
class PrimaryObject(ut.NiceRepr, ut.HashComparable2):
    def __init__(self, rowids, ibs, config=None):
        self._rowids = rowids
        self._ibs = ibs
        self._islist = True
        self._config = config
        self._internal_attrs = {}
        self._rowid_to_idx = None
        #ut.make_index_lookup(self._rowids)

    def lookup_idxs(self, rowids):
        if self._rowid_to_idx is None:
            self._rowid_to_idx = ut.make_index_lookup(self._rowids)
        idx_list = ut.take(self._rowid_to_idx, rowids)
        return idx_list

    def loc(self, rowids):
        idxs = self.lookup_idxs(rowids)
        return self.take(idxs)

    lookup = loc  # TODO: depricate lookup, use loc to be more like pandas

    def __nice__(self):
        return '(num=%r)' % (len(self))

    def __hash__(self):
        return hash(self.group_uuid())

    def group_uuid(self):
        sorted_uuids = sorted(self.uuids)
        group_uuid = ut.util_hash.augment_uuid(*sorted_uuids)
        return group_uuid

    def disconnect(self):
        """
        Disconnects object from the state of the database. All information has
        been assumed to be preloaded.
        """
        self._ibs = None

    def preload(self, *attrs):
        assert self._ibs is not None, 'must be connected to preload'
        for attrname in attrs:
            self._internal_attrs[attrname] = getattr(self, attrname)

    def __iter__(self):
        return iter(self._rowids)

    def __len__(self):
        return len(self._rowids)

    def take(self, idxs):
        rowids = ut.take(self._rowids, idxs)
        newself = self.__class__(rowids, self._ibs, self._config)
        _new_internal = {key: ut.take(val, idxs)
                         for key, val in self._internal_attrs.items()}
        newself._internal_attrs = _new_internal
        return newself

    def compress(self,  flags):
        idxs = ut.where(flags)
        return self.take(idxs)

    def chunks(self,  chunksize):
        for idxs in ut.ichunks(self, range(len(self))):
            yield self.take(idxs)

    def group_indicies(self, labels):
        unique_labels, groupxs = ut.group_indices(labels)
        return unique_labels, groupxs

    def group_items(self, labels):
        """ group as dict """
        unique_labels, groups = self.group(labels)
        label_to_group = ut.odict(zip(unique_labels, groups))
        return label_to_group

    def group(self, labels):
        """ group as list """
        unique_labels, groupxs = self.group_indicies(labels)
        groups = [self.take(idxs) for idxs in groupxs]
        return unique_labels, groups

    # def filter(self, filterkw):
    #     pass

    # def filter_flags(self, filterkw):
    #     pass
