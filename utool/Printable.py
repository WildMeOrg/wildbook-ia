from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import re
import numpy as np
#from .util_classes import AutoReloader

MAX_VALSTR = -1
#100000

#__BASE_CLASS__ = AutoReloader
__BASE_CLASS__ = object


class AbstractPrintable(__BASE_CLASS__):
    'A base class that prints its attributes instead of the memory address'

    def __init__(self, child_print_exclude=[]):
        self._printable_exclude = ['_printable_exclude'] + child_print_exclude

    def __str__(self):
        head = printableType(self)
        body = self.get_printable(type_bit=True)
        body = re.sub('\n *\n *\n', '\n\n', body)
        return head + ('\n' + body).replace('\n', '\n    ')

    def printme(self):
        print(str(self))

    def printme3(self):
        print(self.get_printable())

    def printme2(self,
                 type_bit=True,
                 print_exclude_aug=[],
                 val_bit=True,
                 max_valstr=MAX_VALSTR,
                 justlength=True):
        to_print = self.get_printable(type_bit=type_bit,
                                      print_exclude_aug=print_exclude_aug,
                                      val_bit=val_bit,
                                      max_valstr=max_valstr,
                                      justlength=justlength)
        print(to_print)

    def get_printable(self,
                      type_bit=True,
                      print_exclude_aug=[],
                      val_bit=True,
                      max_valstr=MAX_VALSTR,
                      justlength=False):
        body = ''
        attri_list = []
        exclude_key_list = list(self._printable_exclude) + list(print_exclude_aug)
        for (key, val) in self.__dict__.iteritems():
            try:
                if key in exclude_key_list:
                    continue
                namestr = str(key)
                typestr = printableType(val, name=key, parent=self)
                if not val_bit:
                    attri_list.append((typestr, namestr, '<ommited>'))
                    continue
                valstr  = printableVal(val, type_bit=type_bit, justlength=justlength)
                if len(valstr) > max_valstr and max_valstr > 0:
                    pos1 =  max_valstr // 2
                    pos2 = -max_valstr // 2
                    valstr = valstr[0:pos1] + ' \n ~~~ \n ' + valstr[pos2: - 1]
                attri_list.append((typestr, namestr, valstr))
            except Exception as ex:
                print('[printable] ERROR %r' % ex)
                print('[printable] ERROR key = %r' % key)
                print('[printable] ERROR val = %r' % val)
                try:
                    print('[printable] ERROR valstr = %r' % valstr)
                except Exception:
                    pass
                raise
        attri_list.sort()
        for (typestr, namestr, valstr) in attri_list:
            entrytail = '\n' if valstr.count('\n') <= 1 else '\n\n'
            typestr2 = typestr + ' ' if type_bit else ''
            body += typestr2 + namestr + ' = ' + valstr + entrytail
        return body

    def format_printable(self, type_bit=False, indstr='  * '):
        _printable_str = self.get_printable(type_bit=type_bit)
        _printable_str = _printable_str.replace('\r', '\n')
        _printable_str = indstr + _printable_str.strip('\n').replace('\n', '\n' + indstr)
        return _printable_str


def npArrInfo(arr):
    from .DynamicStruct import DynStruct
    info = DynStruct()
    info.shapestr  = '[' + ' x '.join([str(x) for x in arr.shape]) + ']'
    info.dtypestr  = str(arr.dtype)
    if info.dtypestr == 'bool':
        info.bittotal = 'T=%d, F=%d' % (sum(arr), sum(1 - arr))
    elif info.dtypestr == 'object':
        info.minmaxstr = 'NA'
    elif info.dtypestr[0] == '|':
        info.minmaxstr = 'NA'
    else:
        if arr.size > 0:
            info.minmaxstr = '(%r, %r)' % (arr.min(), arr.max())
        else:
            info.minmaxstr = '(None)'
    return info


# - --------------
def printableType(val, name=None, parent=None):
    if hasattr(parent, 'customPrintableType'):
        # Hack for non - trivial preference types
        _typestr = parent.customPrintableType(name)
        if _typestr is not None:
            return _typestr
    if type(val) == np.ndarray:
        info = npArrInfo(val)
        _typestr = info.dtypestr
    elif isinstance(val, object):
        _typestr = val.__class__.__name__
    else:
        _typestr = str(type(val))
        _typestr = _typestr.replace('type', '')
        _typestr = re.sub('[\'><]', '', _typestr)
        _typestr = re.sub('  *', ' ', _typestr)
        _typestr = _typestr.strip()
    return _typestr


def printableVal(val, type_bit=True, justlength=False):
    # NUMPY ARRAY
    if type(val) is np.ndarray:
        info = npArrInfo(val)
        if info.dtypestr.startswith('bool'):
            _valstr = '{ shape:' + info.shapestr + ' bittotal: ' + info.bittotal + '}'  # + '\n  |_____'
        elif info.dtypestr.startswith('float'):
            _valstr = printable_mystats(val)
        else:
            _valstr = '{ shape:' + info.shapestr + ' mM:' + info.minmaxstr + ' }'  # + '\n  |_____'
    # String
    elif isinstance(val, str):
        _valstr = '\'%s\'' % val
    # List
    elif isinstance(val, list):
        if justlength or len(val) > 30:
            _valstr = 'len=' + str(len(val))
        else:
            _valstr = '[ ' + (', \n  '.join([str(v) for v in val])) + ' ]'
    elif hasattr(val, 'get_printable') and type(val) != type:  # WTF? isinstance(val, AbstractPrintable):
        _valstr = val.get_printable(type_bit=type_bit)
    elif isinstance(val, dict):
        _valstr = '{\n'
        for val_key in val.keys():
            val_val = val[val_key]
            _valstr += '  ' + str(val_key) + ' : ' + str(val_val) + '\n'
        _valstr += '}'
    else:
        _valstr = str(val)
    if _valstr.find('\n') > 0:  # Indent if necessary
        _valstr = _valstr.replace('\n', '\n    ')
        _valstr = '\n    ' + _valstr
    _valstr = re.sub('\n *$', '', _valstr)  # Replace empty lines
    return _valstr


def printable_mystats(_list, newlines=False):
    stat_dict = mystats(_list)
    stat_strs = ['%r: %s' % (key, val) for key, val in stat_dict.iteritems()]
    if newlines:
        indent = '    '
        head = '{\n' + indent
        sep  = ',\n' + indent
        tail = '\n}'
    else:
        head = '{'
        sep = ', '
        tail = '}'
    ret = head + sep.join(stat_strs) + tail
    return ret


def mystats(_list):
    if len(_list) == 0:
        return {'empty_list': True}
    nparr = np.array(_list)
    min_val = nparr.min()
    max_val = nparr.max()
    nMin = np.sum(nparr == min_val)  # number of entries with min val
    nMax = np.sum(nparr == max_val)  # number of entries with min val
    return OrderedDict([('max',   np.float32(max_val)),
                        ('min',   np.float32(min_val)),
                        ('mean',  np.float32(nparr.mean())),
                        ('std',   np.float32(nparr.std())),
                        ('nMin',  np.int32(nMin)),
                        ('nMax',  np.int32(nMax)),
                        ('shape', repr(nparr.shape))])
