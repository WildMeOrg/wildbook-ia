# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
from six.moves import range
(print, rrr, profile) = ut.inject2(__name__, '[_ibeis_object]')


def _find_ibeis_attrs(ibs, objname, blacklist=[]):
    r"""
    Developer function to help figure out what attributes are available

    Args:
        ibs (ibeis.IBEISController):  images analysis api

    CommandLine:
        python -m ibeis.images _find_ibeis_attrs

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
        >>> objname = 'annot'
        >>> blacklist = ['annot_pair']
        >>> _find_ibeis_attrs(ibs, objname, blacklist)
    """
    import re
    getter_prefix = 'get_' + objname + '_'
    found_getters = ut.search_module(ibs, getter_prefix)
    pat = getter_prefix + ut.named_field('attr', '.*')
    for stopword in blacklist:
        found_getters = [fn for fn in found_getters if stopword not in fn]
    matched_getters = [re.match(pat, fn).groupdict()['attr'] for fn in found_getters]

    setter_prefix = 'set_' + objname + '_'
    found_setters = ut.search_module(ibs, setter_prefix)
    pat = setter_prefix + ut.named_field('attr', '.*')
    for stopword in blacklist:
        found_setters = [fn for fn in found_setters if stopword not in fn]
    matched_setters = [re.match(pat, fn).groupdict()['attr'] for fn in found_setters]
    return matched_getters, matched_setters


def _inject_getter_attrs(metaself, objname, attrs, configurable_attrs,
                         depc_name=None, depcache_attrs=None,
                         settable_attrs=None, aliased_attrs=None):
    """
    Used by the metaclass to inject methods and properties into the class
    inheriting from ObjectList1D
    """

    if settable_attrs is None:
        settable_attrs = []
    settable_attrs = set(settable_attrs)

    # Inform the class of which variables will be injected
    metaself._settable_attrs = settable_attrs
    metaself._attrs = attrs
    metaself._configurable_attrs = configurable_attrs
    metaself._depcache_attrs = depcache_attrs
    if depcache_attrs is None:
        metaself._depcache_attrs = []
    if aliased_attrs is not None:
        metaself._attrs_aliases = aliased_attrs
    else:
        metaself._attrs_aliases = {}

    attr_to_aliases = ut.invert_dict(metaself._attrs_aliases, unique_vals=False)

    def _make_getter(objname, attrname):
        ibs_funcname = 'get_%s_%s' % (objname, attrname)
        #def ibs_getter(self, *args, **kwargs):
        def ibs_getter(self):
            if self._ibs is None or (self._caching and
                                     attrname in self._internal_attrs):
                data = self._internal_attrs[attrname]
            else:
                ibs_callable = getattr(self._ibs, ibs_funcname)
                #data = ibs_callable(self._rowids, *args, **kwargs)
                data = ibs_callable(self._rowids)
                if self._caching:
                    self._internal_attrs[attrname] = data
            return data
        ut.set_funcname(ibs_getter, ibs_funcname)
        return ibs_getter

    def _make_setter(objname, attrname):
        ibs_funcname = 'set_%s_%s' % (objname, attrname)
        def ibs_setter(self, values, *args, **kwargs):
            if self._ibs is None:
                return self._internal_attrs[attrname]
            else:
                ibs_callable = getattr(self._ibs, ibs_funcname)
                return ibs_callable(self._rowids, values, *args, **kwargs)
        ut.set_funcname(ibs_setter, ibs_funcname)
        return ibs_setter

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
                data = self._internal_attrs[attrname]
            else:
                depc = getattr(self._ibs, depc_name)
                data = depc.get(tbl, self._rowids, col, config=self._config)
            return data
        ut.set_funcname(ibs_cfg_getter, 'get_' + attrname)
        return ibs_cfg_getter

    # Inject function and property version
    for attrname in attrs:
        ibs_getter = _make_getter(objname, attrname)
        if attrname in settable_attrs:
            ibs_setter = _make_setter(objname, attrname)
        else:
            ibs_setter = None
        prop = property(fget=ibs_getter, fset=ibs_setter)
        setattr(metaself, '_get_' + attrname, ibs_getter)
        if ibs_setter is not None:
            setattr(metaself, '_set_' + attrname, ibs_setter)
        setattr(metaself, attrname, prop)
        for alias in attr_to_aliases.pop(attrname, []):
            setattr(metaself, alias, prop)

    for attrname in configurable_attrs:
        ibs_cfg_getter = _make_configurable_getter(objname, attrname)
        prop = property(ibs_cfg_getter)
        setattr(metaself, '_get_' + attrname, ibs_cfg_getter)
        setattr(metaself, attrname, prop)
        for alias in attr_to_aliases.pop(attrname, []):
            setattr(metaself, alias, prop)

    if depcache_attrs is not None:
        for tbl, col in depcache_attrs:
            attrname = '%s_%s' % (tbl, col)
            ibs_depc_getter = _make_depcache_getter(depc_name, tbl, col)
            prop = property(ibs_depc_getter)
            setattr(metaself, '_get_' + attrname, ibs_depc_getter)
            setattr(metaself, attrname, prop)
            for alias in attr_to_aliases.pop(attrname, []):
                setattr(metaself, alias, prop)
        #import utool
        #utool.embed()
    if attr_to_aliases:
        raise AssertionError('Unmapped aliases %r' % (attr_to_aliases,))


class ObjectScalar0D(ut.NiceRepr, ut.HashComparable2):
    """
    This actually stores a ObjectList1D of length 1 and
    simply calls those functions where available
    """
    def __init__(self, obj1d):
        assert len(obj1d) == 1
        self.obj1d = obj1d

    def __nice__(self):
        return '(rowid=%s, uuid=%s)' % (self._rowids, self.uuids)

    def __getattr__(self, key):
        return getattr(self.obj1d, key)[0]

    def __dir__(self):
        attrs = dir(object)
        attrs += list(self.__class__.__dict__.keys())
        attrs += self.obj1d.__vector_attributes__()
        return attrs

    def _make_lazy_dict(self):
        """
        CommandLine:
            python -m ibeis._ibeis_object ObjectScalar0D._make_lazy_dict

        Example:
            >>> from ibeis._ibeis_object import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb('testdb1')
            >>> annots = ibs.annots()
            >>> subset = annots.take([0, 2, 5])
            >>> scalar = annots[0]
            >>> assert scalar.obj1d._attrs == annots._attrs
            >>> self = scalar
            >>> print(dir(self))
            >>> metadata = self._make_lazy_dict()
            >>> print('metadata = %r' % (metadata,))
            >>> aid = metadata['aid']
            >>> print('aid = %r' % (aid,))
        """
        metadata = ut.LazyDict()
        for attr in self.obj1d.__vector_attributes__():
            metadata[attr] = ut.partial(getattr, self, attr)
        return metadata


#@ut.reloadable_class
class ObjectList1D(ut.NiceRepr, ut.HashComparable2):
    """
    An object that efficiently operates on a list of ibeis objects using
    vectorized code. Single instances can be returned as ObjectScalar0D's
    """
    def __init__(self, rowids, ibs, config=None, caching=False):
        self._rowids = rowids
        #self._islist = True
        # Internal cache
        self._internal_attrs = {}
        # Internal behaviors
        self._ibs = ibs
        self._config = config
        self._caching = caching
        # Private attributes
        self.__rowid_to_idx = None
        #ut.make_index_lookup(self._rowids)

    def __vector_attributes__(self):
        attrs = (self._attrs + self._configurable_attrs +
                 list(self._attrs_aliases.keys()))
        return attrs

    def set_caching(self, flag):
        self._caching = flag

    def __nice__(self):
        return '(num=%r)' % (len(self))

    def __hash__(self):
        return hash(self.group_uuid())

    def __add__(self, other):
        assert self.__class__ is other.__class__, 'incompatable'
        assert self._ibs is other._ibs, 'incompatable'
        assert self._config is other._config, 'incompatable'
        rowids = ut.unique(self._rowids + other._rowids)
        new = self.__class__(rowids, self._ibs, self._config)
        return new

    def take(self, idxs):
        """
        Creates a subset of the list using the specified indices.
        """
        rowids = ut.take(self._rowids, idxs)
        # Create a new instance pointing only to the requested subset
        newself = self.__class__(rowids, ibs=self._ibs, config=self._config,
                                 caching=self._caching)
        # Pass along any internally cached values
        _new_internal = {key: ut.take(val, idxs)
                         for key, val in self._internal_attrs.items()}
        newself._internal_attrs = _new_internal
        return newself

    def preload(self, *attrs):
        assert self._ibs is not None, 'must be connected to preload'
        for attrname in attrs:
            self._internal_attrs[attrname] = getattr(self, attrname)

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

    def __iter__(self):
        return iter(self._rowids)

    def __len__(self):
        return len(self._rowids)

    def __getitem__(self, idx):
        if not ut.isiterable(idx):
            obj0d_ = self.take([idx])
            obj0d = ObjectScalar0D(obj0d_)
            return obj0d
        if not isinstance(idx, slice):
            raise AssertionError('only slice supported currently')
        return self.take(idx)

    def scalars(self):
        scalar_list = [self[idx] for idx in range(len(self))]
        return scalar_list

    def compress(self,  flags):
        idxs = ut.where(flags)
        return self.take(idxs)

    def take_column(self, keys):
        vals_list = zip(*[getattr(self, key) for key in keys])
        dict_list = [dict(zip(keys, vals)) for vals in vals_list]
        return dict_list

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

    def lookup_idxs(self, rowids):
        """ Lookup subset indicies by rowids """
        if self.__rowid_to_idx is None:
            self.__rowid_to_idx = ut.make_index_lookup(self._rowids)
        idx_list = ut.take(self.__rowid_to_idx, rowids)
        return idx_list

    def loc(self, rowids):
        """ Lookup subset by rowids """
        idxs = self.lookup_idxs(rowids)
        return self.take(idxs)

    # def filter(self, filterkw):
    #     pass

    # def filter_flags(self, filterkw):
    #     pass


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis._ibeis_object
        python -m ibeis._ibeis_object --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
