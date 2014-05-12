from __future__ import absolute_import, division, print_function
import copy
from .Printable import AbstractPrintable


class DynStruct(AbstractPrintable):
    """ dynamically add and remove members """
    def __init__(self, child_exclude_list=[], copy_dict=None, copy_class=None):
        super(DynStruct, self).__init__(child_exclude_list)
        if isinstance(copy_dict, dict):
            self.add_dict(copy_dict)

    def dynget(self, *prop_list):
        return tuple([self.__dict__[prop_name] for prop_name in prop_list])

    def dynset(self, *propval_list):
        offset = len(propval_list) / 2
        for i in range(offset):
            self.__dict__[propval_list[i]] = propval_list[i + offset]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            for k, v in zip(key, value):
                setattr(self, k, v)
        else:
            setattr(self, key, value)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            val = []
            for k in key:
                val.append(getattr(self, k))
        else:
            try:
                val = getattr(self, key)
            except TypeError as ex:
                print('[dyn] TYPE_ERROR: %r' % ex)
                print('[dyn] key=%r' % key)
                raise
        return val

    def update(self, **kwargs):
        self_keys = set(self.__dict__.keys())
        for key, val in kwargs.iteritems():
            if key in self_keys:
                if isinstance(val, list):
                    val = val[0]
                self.__dict__[key] = val

    def add_dict(self, dyn_dict):
        'Adds a dictionary to the prefs'
        if not isinstance(dyn_dict, dict):
            raise Exception('DynStruct.add_dict expects a dictionary.' +
                            'Recieved: ' + str(type(dyn_dict)))
        for (key, val) in dyn_dict.iteritems():
            self[key] = val

    def to_dict(self):
        '''Converts dynstruct to a dictionary.  '''
        dyn_dict = {}
        for (key, val) in self.__dict__.iteritems():
            if key not in self._printable_exclude:
                dyn_dict[key] = val
        return dyn_dict

    def flat_dict(self, dyn_dict={}, only_public=True):
        for (key, val) in self.__dict__.iteritems():
            if key in self._printable_exclude:
                continue
            elif only_public and key.find('_') == 0:
                continue
            elif isinstance(val, DynStruct):
                val.flat_dict(dyn_dict, only_public)
            else:
                dyn_dict[key] = val
        return dyn_dict

    def deepcopy(self, **kwargs):
        copy_ = copy.deepcopy(self)
        copy_.update(**kwargs)
        return copy_

    def execstr(self, local_name):
        '''returns a string which when evaluated will
           add the stored variables to the current namespace

           localname is the name of the variable in the current scope
           * use locals().update(dyn.to_dict()) instead
        '''
        execstr = ''
        for (key, val) in self.__dict__.iteritems():
            if key not in self._printable_exclude:
                execstr += key + ' = ' + local_name + '.' + key + '\n'
        return execstr
