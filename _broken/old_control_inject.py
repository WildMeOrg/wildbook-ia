

def find_unregistered_methods():
    r"""
    CommandLine:
        python -m ibeis.control.controller_inject --test-find_unregistered_methods --enableall

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.controller_inject import *  # NOQA
        >>> result = find_unregistered_methods()
        >>> print(result)
    """
    from os.path import dirname
    import utool as ut
    import ibeis.control
    import re
    #regex = r'[^@]*\ndef'
    modfpath = dirname(ibeis.control.__file__)
    fpath_list = ut.glob(modfpath, 'manual_*_funcs.py')
    #fpath_list += ut.glob(modfpath, '_autogen_*_funcs.py')

    def multiline_grepfile(regex, fpath):
        found_matchtexts = []
        found_linenos   = []
        text = ut.read_from(fpath, verbose=False)
        for match in  re.finditer(regex, text, flags=re.MULTILINE):
            lineno = text[:match.start()].count('\n')
            matchtext = ut.get_match_text(match)
            found_linenos.append(lineno)
            found_matchtexts.append(matchtext)
        return found_matchtexts, found_linenos

    def multiline_grep(regex, fpath_list):
        found_fpath_list      = []
        found_matchtexts_list = []
        found_linenos_list    = []
        for fpath in fpath_list:
            found_matchtexts, found_linenos = multiline_grepfile(regex, fpath)
            # append anything found in this file
            if len(found_matchtexts) > 0:
                found_fpath_list.append(fpath)
                found_matchtexts_list.append(found_matchtexts)
                found_linenos_list.append(found_linenos)
        return found_fpath_list, found_matchtexts_list, found_linenos_list

    def print_mutliline_matches(tup):
        found_fpath_list, found_matchtexts_list, found_linenos_list = tup
        for fpath, found_matchtexts, found_linenos in zip(found_fpath_list,
                                                          found_matchtexts_list,
                                                          found_linenos_list):
            print('+======')
            print(fpath)
            for matchtext, lineno in zip(found_matchtexts, found_linenos):
                print('    ' + '+----')
                print('    ' + str(lineno))
                print('    ' + str(matchtext))
                print('    ' + 'L____')

    #print(match)
    print('\n\n GREPING FOR UNDECORATED FUNCTIONS')
    regex = '^[^@\n]*\ndef\\s.*$'
    tup = multiline_grep(regex, fpath_list)
    print_mutliline_matches(tup)

    print('\n\n GREPING FOR UNDECORATED FUNCTION ALIASES')
    regex = '^' + ut.REGEX_VARNAME + ' = ' + ut.REGEX_VARNAME
    tup = multiline_grep(regex, fpath_list)
    print_mutliline_matches(tup)
    #ut.grep('aaa\rdef', modfpath, include_patterns=['manual_*_funcs.py',
    #'_autogen_*_funcs.py'], reflags=re.MULTILINE)


r"""
Vim add decorator
%s/^\n^@\([^r]\)/\r\r@register_ibs_method\r@\1/gc
%s/^\n\(def .*(ibs\)/\r\r@register_ibs_method\r\1/gc
%s/\n\n\n\n/\r\r\r/gc

# FIND UNREGISTERED METHODS
/^[^@]*\ndef
"""


def sort_module_functions():
    from os.path import dirname, join
    import utool as ut
    import ibeis.control
    import re
    #import re
    #regex = r'[^@]*\ndef'
    modfpath = dirname(ibeis.control.__file__)
    fpath = join(modfpath, 'manual_annot_funcs.py')
    #fpath = join(modfpath, 'manual_dependant_funcs.py')
    #fpath = join(modfpath, 'manual_lblannot_funcs.py')
    #fpath = join(modfpath, 'manual_name_species_funcs.py')
    text = ut.read_from(fpath, verbose=False)
    lines =  text.splitlines()
    indent_list = [ut.get_indentation(line) for line in lines]
    isfunc_list = [line.startswith('def ') for line in lines]
    isblank_list = [len(line.strip(' ')) == 0 for line in lines]
    isdec_list = [line.startswith('@') for line in lines]

    tmp = ['def' if isfunc else indent for isfunc, indent in  zip(isfunc_list, indent_list)]
    tmp = ['b' if isblank else t for isblank, t in  zip(isblank_list, tmp)]
    tmp = ['@' if isdec else t for isdec, t in  zip(isdec_list, tmp)]
    #print('\n'.join([str((t, count + 1)) for (count, t) in enumerate(tmp)]))
    block_list = re.split('\n\n\n', text, flags=re.MULTILINE)

    #for block in block_list:
    #    print('#====')
    #    print(block)

    isfunc_list = [re.search('^def ', block, re.MULTILINE) is not None for block in block_list]

    whole_varname = ut.whole_word(ut.REGEX_VARNAME)
    funcname_regex = r'def\s+' + ut.named_field('funcname', whole_varname)

    def findfuncname(block):
        match = re.search(funcname_regex, block)
        return match.group('funcname')

    funcnameblock_list = [findfuncname(block) if isfunc else None
                          for isfunc, block in zip(isfunc_list, block_list)]

    funcblock_list = ut.filter_items(block_list, isfunc_list)
    funcname_list = ut.filter_items(funcnameblock_list, isfunc_list)

    nonfunc_list = ut.filterfalse_items(block_list, isfunc_list)

    nonfunc_list = ut.filterfalse_items(block_list, isfunc_list)
    ismain_list = [re.search('^if __name__ == ["\']__main__["\']', nonfunc) is not None
                   for nonfunc in nonfunc_list]

    mainblock_list = ut.filter_items(nonfunc_list, ismain_list)
    nonfunc_list = ut.filterfalse_items(nonfunc_list, ismain_list)

    newtext_list = []

    for nonfunc in nonfunc_list:
        newtext_list.append(nonfunc)
        newtext_list.append('\n')

    #funcname_list
    for funcblock in ut.sortedby(funcblock_list, funcname_list):
        newtext_list.append(funcblock)
        newtext_list.append('\n')

    for mainblock in mainblock_list:
        newtext_list.append(mainblock)

    newtext = '\n'.join(newtext_list)
    print('newtext = %s' % (newtext,))
    print('len(newtext) = %r' % (len(newtext),))
    print('len(text) = %r' % (len(text),))

    backup_fpath = ut.augpath(fpath, augext='.bak', augdir='_backup', ensure=True)

    ut.write_to(backup_fpath, text)
    ut.write_to(fpath, newtext)

    #for block, isfunc in zip(block_list, isfunc_list):
    #    if isfunc:
    #        print(block)

    #for block, isfunc in zip(block_list, isfunc_list):
    #    if isfunc:
    #        print('----')
    #        print(block)
    #        print('\n')

#class JSONPythonObjectEncoder(json.JSONEncoder):
#    """
#        References:
#            http://stackoverflow.com/questions/8230315/python-sets-are-not-json-serializable
#            http://stackoverflow.com/questions/11561932/why-does-json-dumpslistnp-arange5-fail-while-json-dumpsnp-arange5-tolis
#            https://github.com/jsonpickle/jsonpickle
#    """
#    numpy_type_tuple = tuple([np.ndarray] + list(set(np.typeDict.values())))

#    def default(self, obj):
#        r"""
#        Args:
#            obj (object):


#def _as_python_object(value, verbose=False, **kwargs):
#    if verbose:
#        print('PYTHONIZE: %r' % (value, ))
#    if JSON_PYTHON_OBJECT_TAG in value:
#        pickled_obj = str(value[JSON_PYTHON_OBJECT_TAG])
#        return pickle.loads(pickled_obj)
#    return value

#        Returns:
#            str: json string

#        CommandLine:
#            python -m ibeis.control.controller_inject --test-JSONPythonObjectEncoder.default

#        Example:
#            >>> # DISABLE_DOCTEST
#            >>> from ibeis.control.controller_inject import *  # NOQA
#            >>> self = JSONPythonObjectEncoder()
#            >>> obj_list = [1, [1], {}, 'foobar', np.array([1, 2, 3])]
#            >>> result_list = []
#            >>> for obj in obj_list:
#            ...     try:
#            ...         encoded = json.dumps(obj, cls=JSONPythonObjectEncoder)
#            ...         print(encoded)
#            ...     except Exception as ex:
#            ...         ut.printex(ex)
#        """
#        if isinstance(obj, (list, dict, str, unicode, int, float, bool, type(None))):
#            return json.JSONEncoder.default(self, obj)
#        elif isinstance(obj, self.numpy_type_tuple):
#            #return json.JSONEncoder.default(self, obj.tolist())
#            return obj.tolist()
#        pickled_obj = pickle.dumps(obj)
#        return {JSON_PYTHON_OBJECT_TAG: pickled_obj}
