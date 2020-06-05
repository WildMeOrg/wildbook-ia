# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np  # NOQA
from six.moves import range

(print, rrr, profile) = ut.inject2(__name__, '[_wbia_object]')


def _find_wbia_attrs(ibs, objname, blacklist=[]):
    r"""
    Developer function to help figure out what attributes are available

    Args:
        ibs (wbia.IBEISController):  images analysis api

    CommandLine:
        python -m wbia.images _find_wbia_attrs

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia._wbia_object import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> objname = 'images'
        >>> blacklist = []
        >>> _find_wbia_attrs(ibs, objname, blacklist)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia._wbia_object import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> objname = 'annot'
        >>> blacklist = ['annot_pair']
        >>> _find_wbia_attrs(ibs, objname, blacklist)
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


def _inject_getter_attrs(
    metaself,
    objname,
    attrs,
    configurable_attrs,
    depc_name=None,
    depcache_attrs=None,
    settable_attrs=None,
    aliased_attrs=None,
):
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
    if depcache_attrs is None:
        metaself._depcache_attrs = []
    else:
        metaself._depcache_attrs = ['%s_%s' % (tbl, col) for tbl, col in depcache_attrs]
    if aliased_attrs is not None:
        metaself._attrs_aliases = aliased_attrs
    else:
        metaself._attrs_aliases = {}

    # if not getattr(metaself, '__needs_inject__', True):
    #     return

    attr_to_aliases = ut.invert_dict(metaself._attrs_aliases, unique_vals=False)

    # What is difference between configurable and depcache getters?
    # Could depcache getters just be made configurable?
    # I guess its just an efficincy thing. Actually its config2_-vs-config
    # FIXME: rectify differences between normal / configurable / depcache
    # getter

    def _make_caching_setter(attrname, _rowid_setter):
        def _setter(self, values, *args, **kwargs):
            if self._ibs is None:
                self._internal_attrs[attrname] = values
            else:
                if self._caching and attrname in self._internal_attrs:
                    self._internal_attrs[attrname] = values
                _rowid_setter(self, self._rowids, values)

        ut.set_funcname(_setter, '_set_' + attrname)
        return _setter

    def _make_caching_getter(attrname, _rowid_getter):
        def _getter(self):
            if self._ibs is None or (self._caching and attrname in self._internal_attrs):
                data = self._internal_attrs[attrname]
            else:
                data = _rowid_getter(self, self._rowids)
                if self._caching:
                    self._internal_attrs[attrname] = data
            return data

        ut.set_funcname(_getter, '_get_' + attrname)
        return _getter

    # make default version use implicit rowids and another
    # that takes explicit rowids.

    def _make_setters(objname, attrname):
        ibs_funcname = 'set_%s_%s' % (objname, attrname)

        def _rowid_setter(self, rowids, values, *args, **kwargs):
            ibs_callable = getattr(self._ibs, ibs_funcname)
            ibs_callable(rowids, values, *args, **kwargs)

        ut.set_funcname(_rowid_setter, '_rowid_set_' + attrname)
        _setter = _make_caching_setter(attrname, _rowid_setter)
        return _rowid_setter, _setter

    # ---

    def _make_getters(objname, attrname):
        ibs_funcname = 'get_%s_%s' % (objname, attrname)

        def _rowid_getter(self, rowids):
            ibs_callable = getattr(self._ibs, ibs_funcname)
            data = ibs_callable(rowids)
            if self._asarray:
                data = np.array(data)
            return data

        ut.set_funcname(_rowid_getter, '_rowid_get_' + attrname)
        _getter = _make_caching_getter(attrname, _rowid_getter)
        return _rowid_getter, _getter

    def _make_cfg_getters(objname, attrname):
        ibs_funcname = 'get_%s_%s' % (objname, attrname)

        def _rowid_getter(self, rowids):
            ibs_callable = getattr(self._ibs, ibs_funcname)
            data = ibs_callable(rowids, config2_=self._config)
            if self._asarray:
                data = np.array(data)
            return data

        ut.set_funcname(_rowid_getter, '_rowid_get_' + attrname)
        _getter = _make_caching_getter(attrname, _rowid_getter)
        return _rowid_getter, _getter

    def _make_depc_getters(depc_name, attrname, tbl, col):
        def _rowid_getter(self, rowids):
            depc = getattr(self._ibs, depc_name)
            data = depc.get(tbl, rowids, col, config=self._config)
            if self._asarray:
                data = np.array(data)
            return data

        ut.set_funcname(_rowid_getter, '_rowid_get_' + attrname)
        _getter = _make_caching_getter(attrname, _rowid_getter)
        return _rowid_getter, _getter

    # Collect setter / getter functions and properties
    rowid_getters = []
    getters = []
    setters = []
    properties = []
    for attrname in attrs:
        _rowid_getter, _getter = _make_getters(objname, attrname)
        if attrname in settable_attrs:
            _rowid_setter, _setter = _make_setters(objname, attrname)
            setters.append(_setter)
        else:
            _setter = None
        prop = property(fget=_getter, fset=_setter)
        rowid_getters.append((attrname, _rowid_getter))
        getters.append(_getter)
        properties.append((attrname, prop))

    for attrname in configurable_attrs:
        _rowid_getter, _getter = _make_cfg_getters(objname, attrname)
        prop = property(fget=_getter)
        rowid_getters.append((attrname, _rowid_getter))
        getters.append(_getter)
        properties.append((attrname, prop))

    if depcache_attrs is not None:
        for tbl, col in depcache_attrs:
            attrname = '%s_%s' % (tbl, col)
            _rowid_getter, _getter = _make_depc_getters(depc_name, attrname, tbl, col)
            prop = property(fget=_getter, fset=None)
            rowid_getters.append((attrname, _rowid_getter))
            getters.append(_getter)
            properties.append((attrname, prop))

    aliases = []

    # Inject all gathered information
    for attrname, func in rowid_getters:
        funcname = ut.get_funcname(func)
        setattr(metaself, funcname, func)
        # ensure aliases have rowid getters
        for alias in attr_to_aliases.get(attrname, []):
            alias_funcname = '_rowid_get_' + alias
            setattr(metaself, alias_funcname, func)

    for func in getters:
        funcname = ut.get_funcname(func)
        setattr(metaself, funcname, func)

    for func in setters:
        funcname = ut.get_funcname(func)
        setattr(metaself, funcname, func)

    for attrname, prop in properties:
        setattr(metaself, attrname, prop)
        for alias in attr_to_aliases.pop(attrname, []):
            aliases.append((alias, attrname))
            setattr(metaself, alias, prop)

    if ut.get_argflag('--autogen-core'):
        # TODO: turn on autogenertion given a flag
        def expand_closure_source(funcname, func):
            source = ut.get_func_sourcecode(func)
            closure_vars = [
                (k, v.cell_contents)
                for k, v in zip(func.func_code.co_freevars, func.func_closure)
            ]
            source = ut.unindent(source)
            import re

            for k, v in closure_vars:
                source = re.sub('\\b' + k + '\\b', ut.repr2(v), source)
            source = re.sub(r'def .*\(self', 'def ' + funcname + '(self', source)
            source = ut.indent(source.strip(), '    ') + '\n'
            return source

        explicit_lines = []
        # build explicit version for jedi?
        for funcname, func in getters:
            source = expand_closure_source(funcname, func)
            explicit_lines.append(source)
        # build explicit version for jedi?
        for funcname, func in setters:
            source = expand_closure_source(funcname, func)
            explicit_lines.append(source)

        for attrname, prop in properties:
            getter_name = None if prop.fget is None else ut.get_funcname(prop.fget)
            setter_name = None if prop.fset is None else ut.get_funcname(prop.fset)
            source = '    %s = property(%s, %s)' % (attrname, getter_name, setter_name)
            explicit_lines.append(source)

        for alias, attrname in aliases:
            source = '    %s = %s' % (alias, attrname)
            explicit_lines.append(source)

        explicit_source = '\n'.join(
            [
                'from wbia import _wbia_object',
                '',
                '',
                'class _%s_base_class(_wbia_object.ObjectList1D):',
                '    __needs_inject__ = False',
                '',
            ]
        ) % (objname,)
        explicit_source += '\n'.join(explicit_lines)
        explicit_fname = '_autogen_%s_base.py' % (objname,)
        from os.path import dirname, join

        ut.writeto(join(dirname(__file__), explicit_fname), explicit_source + '\n')

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
        return 'rowid=%s, uuid=%s' % (self._rowids, self.uuids)

    def __getattr__(self, key):
        vals = getattr(self.obj1d, key)
        if key == 'show':
            return vals
        return vals[0]

    def __dir__(self):
        attrs = dir(object)
        attrs += list(self.__class__.__dict__.keys())
        attrs += self.obj1d.__vector_attributes__()
        return attrs

    def _make_lazy_dict(self):
        """
        CommandLine:
            python -m wbia._wbia_object ObjectScalar0D._make_lazy_dict

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia._wbia_object import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb('testdb1')
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


# @ut.reloadable_class
class ObjectList1D(ut.NiceRepr, ut.HashComparable2):
    """
    An object that efficiently operates on a list of wbia objects using
    vectorized code. Single instances can be returned as ObjectScalar0D's
    """

    def __init__(self, rowids, ibs, config=None, caching=False, asarray=False):
        self._rowids = rowids
        # self._islist = True
        # Internal cache
        self._internal_attrs = {}
        # Internal behaviors
        self._ibs = ibs
        self._config = config
        self._caching = caching
        # Private attributes
        self._rowid_to_idx = None
        self._asarray = asarray
        # ut.make_index_lookup(self._rowids)

    def __vector_attributes__(self):
        attrs = (
            self._attrs
            + self._configurable_attrs
            + self._depcache_attrs
            + list(self._attrs_aliases.keys())
        )
        return attrs

    def set_caching(self, flag):
        self._caching = flag

    def __nice__(self):
        return 'num=%r' % (len(self))

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
        newself = self.__class__(
            rowids, ibs=self._ibs, config=self._config, caching=self._caching
        )
        # Pass along any internally cached values
        _new_internal = {
            key: ut.take(val, idxs) for key, val in self._internal_attrs.items()
        }
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
        if isinstance(idx, slice):
            idxs = list(range(*idx.indices(len(self))))
            return self.take(idxs)
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

    def compress(self, flags):
        idxs = ut.where(flags)
        return self.take(idxs)

    def take_column(self, keys):
        vals_list = zip(*[getattr(self, key) for key in keys])
        dict_list = [dict(zip(keys, vals)) for vals in vals_list]
        return dict_list

    def chunks(self, chunksize):
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
        if self._rowid_to_idx is None:
            self._rowid_to_idx = ut.make_index_lookup(self._rowids)
        idx_list = ut.take(self._rowid_to_idx, rowids)
        return idx_list

    def loc(self, rowids):
        """ Lookup subset by rowids """
        idxs = self.lookup_idxs(rowids)
        return self.take(idxs)

    # def filter(self, filterkw):
    #     pass

    # def filter_flags(self, filterkw):
    #     pass

    def view(self, rowids=None):
        """
        Like take, but returns a view proxy that maps to the original parent
        """
        if rowids is None:
            rowids = self._rowids
        # unique_parent = self.take(unique_idxs)
        view = ObjectView1D(rowids, obj1d=self)
        return view


class ObjectView1D(ut.NiceRepr):
    # ut.HashComparable2):
    """
    Allows for proxy caching.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia._wbia_object import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids = ibs.get_valid_aids()
        >>> a = self = annots = ibs.annots(aids)
        >>> rowids = [1, 1, 3, 2, 1, 2]
        >>> self = v = a.view(rowids)
        >>> assert np.all(v.vecs[0] == v.vecs[1])
        >>> assert v.vecs[0] is v.vecs[1]
        >>> assert v.vecs[0] is not v.vecs[2]
    """

    def __init__(self, rowids, obj1d, cache=None):
        self._rowids = list(rowids)
        self._obj1d = obj1d
        self._unique_rowids = set(self._rowids)
        self._unique_inverse = ut.list_alignment(self._unique_rowids, self._rowids)
        if cache is None:
            self._cache = ut.ddict(dict)
        else:
            self._cache = cache
        # Views always cache data for now
        self._caching = True

    def __dir__(self):
        attrs = dir(object)
        attrs += self.__dict__.keys()
        attrs += ['__dict__', '__module__', '__weakref__']
        # ['_unique_parent', '_caching', '_attr_rowid_value', '_rowids']
        attrs += list(self.__class__.__dict__.keys())
        attrs += self._obj1d.__vector_attributes__()
        return attrs

    def __vector_attributes__(self):
        return self._obj1d.__vector_attributes__()

    def __getattr__(self, key):
        """
        key = 'vecs'
        """
        try:
            _rowid_getter = getattr(self._obj1d, '_rowid_get_%s' % (key,))
        except AttributeError:
            raise AttributeError('ObjectView1D has no attribute %r' % (key,))
        if self._caching:
            rowid_to_value = self._cache[key]
            miss_rowids = [
                rowid for rowid in self._unique_rowids if rowid not in rowid_to_value
            ]
            miss_data = _rowid_getter(miss_rowids)
            for rowid, value in zip(miss_rowids, miss_data):
                rowid_to_value[rowid] = value
            unique_data = ut.take(rowid_to_value, self._unique_rowids)
        else:
            unique_data = _rowid_getter(self._unique_rowids)
        data = ut.take(unique_data, self._unique_inverse)
        return data

    def __iter__(self):
        return iter(self._rowids)

    def __len__(self):
        return len(self._rowids)

    def __nice__(self):
        return 'unique=%r, num=%r' % (len(self._unique_rowids), len(self))

    # def __hash__(self):
    #     return hash(self.group_uuid())

    def view(self, rowids):
        """
        returns a view of a view that uses the same per-item cache

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia._wbia_object import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb(defaultdb='testdb1')
            >>> aids = ibs.get_valid_aids()
            >>> annots = ibs.annots(aids)
            >>> self = annots.view(annots._rowids)
            >>> v1 = self.view([1, 1, 2, 3, 1, 2])
            >>> v2 = self.view([3, 4, 5])
            >>> v3 = self.view([1, 4])
            >>> v4 = self.view(3)
            >>> lazy4 = v4._make_lazy_dict()
            >>> assert v1.vecs[0] is v3.vecs[0]
            >>> assert v2._cache is self._cache
            >>> assert v2._cache is v1._cache
        """
        if ut.isiterable(rowids):
            childview = self.__class__(rowids, obj1d=self._obj1d, cache=self._cache)
        else:
            childview = self.__class__([rowids], obj1d=self._obj1d, cache=self._cache)
            childview = ObjectScalar0D(childview)
        return childview


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia._wbia_object
        python -m wbia._wbia_object --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
