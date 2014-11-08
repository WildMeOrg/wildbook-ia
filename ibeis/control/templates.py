"""
Templated Autogenerator for the IBEIS Controller

CommandLine:
    python ibeis/control/templates.py
    python ibeis/control/templates.py --dump-autogen-controller
"""
from __future__ import absolute_import, division, print_function
import six
import utool  # NOQA
import utool as ut
from ibeis import constants
from os.path import dirname, join, relpath  # NOQA
import ibeis.control.template_definitions as Tdef


STRIP_DOCSTR   = False
STRIP_EXAMPLE   = True
STRIP_COMMENTS = False
USE_SHORTNAMES = True
USE_FUNCTYPE_HEADERS = True


def format_controller_func(func_code):
    """
    CommandLine:
        python ibeis/control/templates.py
    """
    func_code = remove_sentinals(func_code)
    # BOTH OPTIONS ARE NOT GARUENTEED TO WORK. If there are bugs here may be a
    # good place to look.
    REMOVE_NPARAMS = True
    REMOVE_EAGER = True
    REMOVE_QREQ = False
    WITH_PEP8 = True
    WITH_DECOR = True

    if REMOVE_NPARAMS:
        func_code = remove_kwarg('nInput', 'None', func_code)
    if REMOVE_EAGER:
        func_code = remove_kwarg('eager', 'True', func_code)
    if REMOVE_QREQ:
        func_code = remove_kwarg('qreq_', 'None', func_code)
    if STRIP_COMMENTS:
        #func_code = ut.regex_replace(r'  # .*$', '', func_code)
        #func_code = ut.regex_replace('^ *# .*$\n', '', func_code)
        pass
    if STRIP_DOCSTR:
        # HACKY: might not always work. newline hacks away dumb blank line
        func_code = ut.regex_replace('""".*"""\n    ', '', func_code)
    if STRIP_EXAMPLE:
        func_code = ut.regex_replace('Example.*"""', '"""', func_code)
    if USE_SHORTNAMES:
        # Execute search and replaces without changing strings
        func_code = ut.replace_nonquoted_text(func_code,
                                              variable_aliases.keys(),
                                              variable_aliases.values())
    # add decorators
    if WITH_DECOR:
        func_code = '@register_ibs_method\n' + func_code
    # ensure pep8 formating
    if WITH_PEP8:
        func_code = ut.autofix_codeblock(func_code).strip()
    return func_code


class SHORTNAMES(object):
    ANNOT      = 'annot'
    CHIP       = 'chip'
    FEAT       = 'feat'
    FEATWEIGHT = 'featweight'
    RVEC       = 'residual'  # 'rvec'
    VOCABTRAIN = 'vocabtrain'
    DETECT     = 'detect'

depends_map = {
    SHORTNAMES.ANNOT: None,
    SHORTNAMES.CHIP: SHORTNAMES.ANNOT,
    SHORTNAMES.FEAT: SHORTNAMES.CHIP,
    SHORTNAMES.FEATWEIGHT: SHORTNAMES.FEAT,
    SHORTNAMES.RVEC:       SHORTNAMES.FEAT,
}

# shortened tablenames
tablename2_tbl = {
    constants.ANNOTATION_TABLE     : SHORTNAMES.ANNOT,
    constants.CHIP_TABLE           : SHORTNAMES.CHIP,
    constants.FEATURE_TABLE        : SHORTNAMES.FEAT,
    constants.FEATURE_WEIGHT_TABLE : SHORTNAMES.FEATWEIGHT,
    constants.RESIDUAL_TABLE       : SHORTNAMES.RVEC,
}

# mapping to variable names in constants
tbl2_tablename = ut.invert_dict(tablename2_tbl)
tbl2_TABLE = {key: ut.get_varname_from_locals(val, constants.__dict__)
              for key, val in six.iteritems(tbl2_tablename)}

# Lets just use strings in autogened files for now: TODO: use constant vars
# later
#tbl2_TABLE = {key: '\'%s\'' % (val,) for key, val in six.iteritems(tbl2_tablename)}
tbl2_TABLE = {key: 'constants.' + ut.get_varname_from_locals(val, constants.__dict__)
                for key, val in six.iteritems(tbl2_tablename)}


variable_aliases = {
    #'chip_rowid_list': 'cid_list',
    #'annot_rowid_list': 'aid_list',
    #'feature_rowid_list': 'fid_list',
    'chip_rowid': 'cid',
    'annot_rowid': 'aid',
    'feat_rowid': 'fid',
    'num_feats': 'nFeat',
    'featweight_forground_weight': 'fgweight',
    'keypoints': 'kpt_list',
    'vectors': 'vec_list',
    'residualvecs': 'rvec_list',
}


def get_tableinfo(tablename, ibs=None):
    dbself = None
    tableinfo = None
    if ibs is not None:
        valid_db_tablenames = ibs.db.get_table_names()
        valid_dbcache_tablenames = ibs.dbcache.get_table_names()

        sqldb = None
        if tablename in valid_db_tablenames:
            sqldb = ibs.db
            dbself = 'db'
        elif tablename in valid_dbcache_tablenames:
            if sqldb is not None:
                raise AssertionError('Tablename=%r is specified in both schemas' % tablename)
            sqldb = ibs.dbcache
            dbself = 'dbcache'
        else:
            print('WARNING unknown tablename=%r' % tablename)

        if sqldb is not None:
            all_colnames = sqldb.get_column_names(tablename)
            superkey_colnames = sqldb.get_table_superkey_colnames(tablename)
            primarykey_colnames = sqldb.get_table_primarykey_colnames(tablename)
            other_colnames = sqldb.get_table_otherkey_colnames(tablename)
    if dbself is None:
        dbself = 'dbunknown'
        all_colnames        = []
        superkey_colnames   = []
        primarykey_colnames = []
        other_colnames      = []
        if tablename == constants.FEATURE_WEIGHT_TABLE:
            dbself = 'dbcache'
            all_colnames = ['feature_weight_fg']
    if tablename == constants.RESIDUAL_TABLE:
        other_colnames.append('rvecs')
    tableinfo = (dbself, all_colnames, superkey_colnames, primarykey_colnames, other_colnames)
    return tableinfo


def remove_sentinals(code_text):
    code_text = ut.regex_replace(r'^ *# STARTBLOCK *$\n', '', code_text)
    code_text = ut.regex_replace(r'^ *# ENDBLOCK *$\n?', '', code_text)
    code_text = ut.regex_replace(r'^ *# *REM [^\n]*$\n?', '', code_text)
    code_text = code_text.rstrip()
    return code_text


def remove_kwarg(kwname, kwdefault, func_code):
    func_code = ut.regex_replace(r' *>>> *{0} *= *{1} *\n'.format(kwname, kwdefault), '', func_code)
    func_code = ut.regex_replace(r',? *{0} *= *{1}'.format(kwname, kwname), '', func_code)
    for val in kwdefault if isinstance(kwdefault, (list, tuple)) else [kwdefault]:
        func_code = ut.regex_replace(r',? *{0} *= *{1}'.format(kwname, val), '', func_code)
    return func_code


def build_dependent_controller_funcs(tablename, tableinfo):
    """
        python ibeis/control/templates.py
        python ibeis/control/templates.py --dump-autogen-controller
    """
    # +-----
    # Setup
    # +-----
    (dbself, all_colnames, superkey_colnames, primarykey_colnames, other_colnames) = tableinfo
    other_cols = list(map(lambda colname: colname2_col(colname, tablename), other_colnames))
    other_COLNAMES = list(map(lambda colname: colname.upper(), other_colnames))
    nonprimary_leaf_colnames = ut.setdiff_ordered(all_colnames, primarykey_colnames)
    leaf_other_propnames = ', '.join(other_colnames)
    leaf_other_propname_lists = ', '.join([colname + '_list' for colname in other_colnames])
    # for the preproc_tbe.compute... method
    leaf_props = '_'.join(other_colnames)
    superkey_args = ', '.join([colname + '_list' for colname in superkey_colnames])

    fmtdict = {
    }

    fmtdict['nonprimary_leaf_colnames'] = nonprimary_leaf_colnames
    fmtdict['leaf_other_propnames'] = leaf_other_propnames
    fmtdict['leaf_other_propname_lists'] = leaf_other_propname_lists
    fmtdict['leaf_props'] = leaf_props
    fmtdict['superkey_args'] = superkey_args
    fmtdict['self'] = 'ibs'
    fmtdict['dbself'] = dbself

    CONSTANT_COLNAMES = []
    CONSTANT_COLNAMES.extend(other_colnames)
    functype2_func_list = ut.ddict(list)
    constant_list = []
    # L_____

    # +----------------------------
    # | Format dict helper functions
    # +----------------------------
    def _setupper(fmtdict, key, val):
        fmtdict[key] = val
        fmtdict[key.upper()] = val.upper()

    def set_parent_child(parent, child):
        _setupper(fmtdict, 'parent', parent)
        _setupper(fmtdict, 'child', child)

    def set_root_leaf(root, leaf, leaf_parent):
        _setupper(fmtdict, 'root', root)
        _setupper(fmtdict, 'leaf', leaf)
        _setupper(fmtdict, 'leaf_parent', leaf_parent)
        fmtdict['LEAF_TABLE'] = tbl2_TABLE[leaf]  # tblname1_TABLE[child]
        fmtdict['ROOT_TABLE'] = tbl2_TABLE[root]  # tblname1_TABLE[child]

    def set_tbl(tbl):
        _setupper(fmtdict, 'tbl', tbl)
        fmtdict['TABLE'] = tbl2_TABLE[tbl]

    def append_func(func_type, func_code_fmtstr, tablename=tablename):
        func_type = func_type
        try:
            func_code = func_code_fmtstr.format(**fmtdict)
            func_code = format_controller_func(func_code)
            functype2_func_list[func_type].append(func_code)
        except Exception as ex:
            utool.printex(ex, keys=['func_type', 'tablename'])
            raise

    def append_constant(varname, valstr):
        const_fmtstr = varname + ' = \'%s\'' % (valstr,)
        constant_list.append(const_fmtstr.format(**fmtdict))
    # L____________________________

    # Build dependency path
    tbl = tablename2_tbl[tablename]
    depends_list = build_depends_path(tbl)
    print('depends_list = %r' % depends_list)

    # set native variables
    for tbl_ in depends_list:
        set_tbl(tbl_)
        # rowid constants
        append_constant('{TBL}_ROWID', '{tbl}_rowid')
    # set table
    set_tbl(tbl)

    # Build pc dependeant lines
    pc_dependant_rowid_lines = []
    pc_dependant_delete_lines = []
    # For each parent child dependancy
    for parent, child in ut.itertwo(depends_list):
        set_parent_child(parent, child)
        # depenant rowid lines
        pc_dependant_rowid_lines.append( Tdef.Tline_pc_dependant_rowid.format(**fmtdict))
        pc_dependant_delete_lines.append(Tdef.Tline_pc_dependant_delete.format(**fmtdict))
    # At this point parent = leaf_parent and child=leaf
    fmtdict['pc_dependant_rowid_lines']  = ut.indent(ut.indentjoin(pc_dependant_rowid_lines)).strip()
    fmtdict['pc_dependant_delete_lines'] = ut.indent(ut.indentjoin(pc_dependant_delete_lines)).strip()

    # ------------------
    #  Parent Leaf Dependency
    # ------------------
    if len(depends_list) > 1:
        if len(depends_list) == 2:
            set_root_leaf(depends_list[0], depends_list[-1], depends_list[0])
        else:
            set_root_leaf(depends_list[0], depends_list[-1], depends_list[-2])
        append_func('0_PL.Tadder',   Tdef.Tadder_pl_dependant)
        append_func('0_PL.Tgetter_rowids',  Tdef.Tgetter_pl_dependant_rowids)
        append_func('0_PL.Tgetter_rowids_',  Tdef.Tgetter_pl_dependant_rowids_)

    # ----------------------------
    # Root Leaf Dependancy
    # ----------------------------
    if len(depends_list) > 2:
        set_root_leaf(depends_list[0], depends_list[-1], depends_list[-2])
        append_func('1_RL.Tadder',   Tdef.Tadder_rl_dependant)
        append_func('1_RL.Tgetter',  Tdef.Tgetter_rl_dependant_all_rowids)
        append_func('1_RL.Tgetter',  Tdef.Tgetter_rl_dependant_rowids)
        append_func('1_RL.Tdeleter', Tdef.Tdeleter_rl_depenant)

    # --------
    #  Native
    # --------
    append_func('2_Native.Tider_all_rowids', Tdef.Tider_all_rowids)
    append_func('2_Native.Tget_from_superkey', Tdef.Tgetter_native_rowid_from_superkey)
    append_func('2_Native.Tdeleter', Tdef.Tdeleter_native_tbl)
    if len(depends_list) > 1:
        # Only dependants have native configs
        append_func('2_Native.Tcfg', Tdef.Tcfg_rowid_getter)

    # For each column property
    for colname, col, COLNAME in zip(other_colnames, other_cols, other_COLNAMES):
        fmtdict['COLNAME'] = COLNAME
        fmtdict['col'] = col
        # Getter template: dependant columns
        for parent, child in ut.itertwo(depends_list):
            set_parent_child(parent, child)
            fmtdict['TABLE'] = tbl2_TABLE[child]  # tblname1_TABLE[child]
        # Getter template: native (Level 0) columns
        append_func('2_Native.Tgetter_native', Tdef.Tgetter_table_column)
        append_func('2_Native.Tsetter_native', Tdef.Tsetter_native_column)
        if len(depends_list) > 1:
            append_func('RL.Tgetter_dependant', Tdef.Tgetter_rl_pclines_dependant_column)
        constant_list.append(COLNAME + ' = \'%s\'' % (colname,))
        append_constant(COLNAME, colname)

    return functype2_func_list, constant_list


#def test():
#    from ibeis.control.templates import *  # NOQA

def build_depends_path(child):
    parent = depends_map[child]
    if parent is not None:
        return build_depends_path(parent) + [child]
    else:
        return [child]
    #depends_list = ['annot', 'chip', 'feat', 'featweight']


def colname2_col(colname, tablename):
    # col is a short alias for colname
    col = colname.replace(ut.singular_string(tablename) + '_', '')
    return col


def main(ibs):
    """
    CommandLine:
        python dev.py --db testdb1 --cmd
        %run dev.py --db testdb1 --cmd
    """
    tblname_list = [
        #constants.ANNOTATION_TABLE,

        #constants.CHIP_TABLE,
        #constants.FEATURE_TABLE,
        constants.FEATURE_WEIGHT_TABLE,

        #constants.RESIDUAL_TABLE
    ]
    #child = 'featweight'
    tblname2_functype2_func_list = ut.ddict(lambda: ut.ddict(list))
    constant_list_ = [
        'CONFIG_ROWID = \'config_rowid\'',
        'FEATWEIGHT_ROWID = \'featweight_rowid\'',
    ]
    for tablename in tblname_list:
        tableinfo = get_tableinfo(tablename, ibs)

        functype2_func_list, constant_list = build_dependent_controller_funcs(tablename, tableinfo)
        constant_list_.extend(constant_list)
        tblname2_functype2_func_list[tablename] = functype2_func_list

    functype_set = set([])
    for tblname, val in six.iteritems(tblname2_functype2_func_list):
        for functype in six.iterkeys(val):
            functype_set.add(functype)
    functype_list = sorted(list(functype_set))

    # Append constants to body
    aligned_constants = '\n'.join(ut.align_lines(sorted(list(set(constant_list_)))))
    autogen_constants = ('# AUTOGENED CONSTANTS:\n' + aligned_constants)

    body_codeblocks = []
    # Append functions to body
    seen = set([])
    for count1, functype in enumerate(functype_list):
        functype_codeblocks = []
        if USE_FUNCTYPE_HEADERS:
            functype_section_header = ut.codeblock(
                '''
                # =========================
                # {FUNCTYPE} METHODS
                # =========================
                '''
            ).format(FUNCTYPE=functype.upper())
            functype_codeblocks.append(functype_section_header)
        for count, item in enumerate(six.iteritems(tblname2_functype2_func_list)):
            tblname, val = item
            #functype_table_section_header = ut.codeblock(
            #    '''
            #    #
            #    # {functype} tablename='{tblname}'
            #    '''
            #).format(functype=functype, tblname=tblname)
            #functype_codeblocks.append(functype_table_section_header)
            for func_codeblock in val[functype]:
                if func_codeblock in seen:
                    continue
                seen.add(func_codeblock)
                functype_codeblocks.append(func_codeblock)
        body_codeblocks.extend(functype_codeblocks)

    # Make main docstr
    testable_name_list = ['get_annot_featweight_rowids']
    autogen_rel_fpath = join(relpath(dirname(ibeis.control.__file__), dirname(dirname(ibeis.__file__))), '_autogen_ibeiscontrol_funcs.py')
    autogen_fpath = join(ut.truepath(dirname(ibeis.control.__file__)), '_autogen_ibeiscontrol_funcs.py')
    autogen_fpath2 = autogen_fpath.replace(ut.truepath('~'), '~')
    autogen_module_name = ut.get_absolute_import(autogen_fpath)
    main_commandline_block_lines = [
        'python ' + autogen_rel_fpath,
    ]
    for testable_name in testable_name_list:
        main_commandline_block_lines.append('python ' + autogen_rel_fpath + ' --test-' + testable_name)
        pass
    main_commandline_block = '\n'.join(main_commandline_block_lines)
    main_commandline_docstr = 'CommandLine:\n' + utool.indent(main_commandline_block, ' ' * 8)
    main_docstr_blocks = [main_commandline_docstr]
    main_docstr_body = '\n'.join(main_docstr_blocks)

    # Contenate autogen parts into autogen_text

    autogen_header = remove_sentinals(Tdef.Theader_ibeiscontrol.format(timestamp=ut.get_timestamp('printable')))

    autogen_body = ('\n\n\n'.join(body_codeblocks))

    autogen_footer = remove_sentinals(Tdef.Tfooter_ibeiscontrol.format(main_docstr_body=main_docstr_body))

    autogen_text = '\n'.join([
        autogen_header,
        '',
        autogen_constants,
        '',
        autogen_body,
        '',
        autogen_footer,
        '',
    ])

    # POSTPROCESSING HACKS:
    autogen_text = autogen_text.replace('\'feat_rowid\'', '\'feature_rowid\'')
    autogen_text = ut.regex_replace(r'kptss', 'kpt_lists', autogen_text)
    autogen_text = ut.regex_replace(r'vecss', 'vec_lists', autogen_text)
    autogen_text = ut.regex_replace(r'nFeatss', 'nFeat_list', autogen_text)
    autogen_text = autogen_text.replace('\'feat_rowid\'', '\'feature_rowid\'')

    if ut.get_flag('--dump-autogen-controller'):
        ut.write_to(autogen_fpath, autogen_text)
    else:
        print(autogen_text)

    return locals()


if __name__ == '__main__':
    """
    CommandLine:
        python ibeis/control/templates.py
        python ibeis/control/templates.py --dump-autogen-controller
    """
    if 'ibs' not in vars():
        import ibeis
        ibs = ibeis.opendb('testdb1')
    locals_ = main(ibs)
    exec(ut.execstr_dict(locals_))
