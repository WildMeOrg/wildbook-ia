"""
Templated Autogenerator for the IBEIS Controller

Concepts:

    template definitions are in template_def.py

    ALL_CAPS_VARIABLES: stores the name of a python GLOBAL constant
        EG:
            * COLNAME : stores the name of a python constant like 'FEAT_ROWID'
            * MULTICOLNAME : stores the name of a python constant that contains the actual

    all_lower_variables: stores the template python representation.
         EG:
             * colname: stores the actual column name like 'feature_rowid'


CommandLine:
    # for current schema
    python -m ibeis.control.DB_SCHEMA --test-test_db_schema

CommandLine:
    python ibeis/templates/template_generator.py
    python -m ibeis.templates.template_generator --key featweight --write
    python -m ibeis.templates.template_generator --key featweight
    python -m ibeis.templates.template_generator --key encounter
    python -m ibeis.templates.template_generator --key encounter --onlyfn
    python -m ibeis.templates.template_generator --key encounter --onlyfn --Tcfg with_native=False
    python -m ibeis.templates.template_generator --key egr --Tcfg with_relations=True with_getters=True
    python -m ibeis.templates.template_generator --key egr --Tcfg with_native=False

    python -m ibeis.templates.template_generator --key match --Tcfg with_native=False
    python -m ibeis.templates.template_generator --key party --Tcfg with_native=False
    python -m ibeis.templates.template_generator --key party_contrib_relation

    python -m ibeis.templates.template_generator --key party_contrib_relation --Tcfg strip_docstr=True strip_eager=True strip_nparams=True
    python -m ibeis.templates.template_generator --key match --Tcfg strip_docstr=True strip_eager=True strip_nparams=True

    python -m ibeis.templates.template_generator --key party --Tcfg with_api_cache=False with_deleters=False
    python -m ibeis.templates.template_generator --key party --Tcfg with_api_cache=False with_deleters=False --write
    python -m ibeis.templates.template_generator --key annotmatch
    --Tcfg with_api_cache=False with_deleters=False --write

    python -m ibeis.templates.template_generator --key images --funcname-filter party --Tcfg with_api_cache=False with_deleters=False
    python -m ibeis.templates.template_generator --key images --funcname-filter contrib --Tcfg with_api_cache=False with_deleters=False
    python -m ibeis.templates.template_generator --key annotmatch --Tcfg with_web_api=False with_api_cache=False with_deleters=False


TODO:
   * autogen testdata function
   * finish autogen chips and features
   * add autogen probchip
   * consistency check that all chips in the sql table exist

"""
from __future__ import absolute_import, division, print_function
import six
import utool  # NOQA
import utool as ut
from ibeis import constants as const
from os.path import dirname, join, relpath  # NOQA
from ibeis.templates import template_definitions as Tdef


STRIP_DOCSTR   = False
STRIP_LONGDESC = False  # True
STRIP_EXAMPLE  = False  # True
STRIP_COMMENTS = False
USE_ALIASES = True
USE_FUNCTYPE_HEADERS = False  # True


#strip_nparams = False  # True
#strip_eager = False  # True
REMOVE_CONFIG2_ = False  # False
WITH_PEP8 = True
WITH_DECOR = True
WITH_API_CACHE = False
WITH_WEB_API = False


#STRIP_DOCSTR   = True
#STRIP_COMMENTS  = True

const.PROBCHIP_TABLE = 'probchips'

# defines which tables to generate
TBLNAME_LIST = [
    #const.ANNOTATION_TABLE,
    #const.CHIP_TABLE,
    #const.PROBCHIP_TABLE,
    #const.FEATURE_TABLE,
    #const.FEATURE_WEIGHT_TABLE,
    #const.RESIDUAL_TABLE
    #const.ENCOUNTER_TABLE
    #const.LBLIMAGE_TABLE
]

multicolumns_dict = ut.odict([
    (
        const.CHIP_TABLE,
        [
            ('chip_size', ('chip_width', 'chip_height'), True),
        ]
    ),
    (
        const.ANNOTATION_TABLE,
        [
            ('annot_bbox', ('annot_xtl', 'annot_ytl', 'annot_width', 'annot_height'), True),
            ('annot_visualinfo', ('annot_verts', 'annot_theta', 'annot_view'), False),
            # TODO: Need to make this happen by performing nested sql calls
            #('annot_semanticinfo', ('annot_image_uuid', 'annot_verts', 'annot_theta', 'annot_view', 'annot_name', 'annot_species'), False),
        ]
    ),
])


tblname2_ignorecolnames = ut.odict([
    #(const.ANNOTATION_TABLE, ('annot_parent_rowid', 'annot_detect_confidence')),
])

readonly_set = {
    const.CHIP_TABLE,
    #const.PROBCHIP_TABLE,
    #const.FEATURE_TABLE,
    #const.FEATURE_WEIGHT_TABLE,
    #const.RESIDUAL_TABLE
}


# HACK
#class SHORTNAMES(object):
#    ANNOT      = 'annot'
#    CHIP       = 'chip'
#    PROBCHIP   = 'probchip'
#    FEAT       = 'feat'
#    FEATWEIGHT = 'featweight'
#    RVEC       = 'residual'  # 'rvec'
#    VOCABTRAIN = 'vocabtrain'
#    DETECT     = 'detect'
#    ENCOUNTER  = 'encounter'
#    IMAGE      = 'image'
#    MATCH      = 'annotmatch'
#    EGR        = 'egr'
#    PCR        = 'party_contrib_relation'
#    CONTRIB    = 'contributor'
#    PARTY      = 'party'

#depends_map = {
#    #SHORTNAMES.MATCH     : None,
#    #SHORTNAMES.EGR       : None,
#    #SHORTNAMES.PCR       : None,
#    #SHORTNAMES.IMAGE     : None,
#    #SHORTNAMES.ENCOUNTER : None,
#    #SHORTNAMES.ANNOT     : None,
#    SHORTNAMES.CHIP:       SHORTNAMES.ANNOT,
#    SHORTNAMES.PROBCHIP:   SHORTNAMES.CHIP,
#    SHORTNAMES.FEAT:       SHORTNAMES.CHIP,
#    SHORTNAMES.FEATWEIGHT: SHORTNAMES.FEAT,  # TODO: and PROBCHIP
#    SHORTNAMES.RVEC:       SHORTNAMES.FEAT,
#}

#relationship_map = {
#    SHORTNAMES.EGR: (SHORTNAMES.IMAGE, SHORTNAMES.ENCOUNTER),
#    SHORTNAMES.PCR: (SHORTNAMES.PARTY, SHORTNAMES.CONTRIB),
#    SHORTNAMES.MATCH: (SHORTNAMES.ANNOT, SHORTNAMES.ANNOT),
#}

# shortened tablenames
# Maps full table names to short table names

#tablename2_tbl = {
#    const.MATCH_TABLE                  : SHORTNAMES.MATCH,
#    const.ANNOTATION_TABLE             : SHORTNAMES.ANNOT,
#    const.CHIP_TABLE                   : SHORTNAMES.CHIP,
#    const.PROBCHIP_TABLE               : SHORTNAMES.PROBCHIP,
#    const.FEATURE_TABLE                : SHORTNAMES.FEAT,
#    const.FEATURE_WEIGHT_TABLE         : SHORTNAMES.FEATWEIGHT,
#    const.RESIDUAL_TABLE               : SHORTNAMES.RVEC,
#    const.ENCOUNTER_TABLE              : SHORTNAMES.ENCOUNTER,
#    const.IMAGE_TABLE                  : SHORTNAMES.IMAGE,
#    #
#    const.EG_RELATION_TABLE            : SHORTNAMES.EGR,
#    const.PARTY_TABLE                  : SHORTNAMES.PARTY,
#    const.PARTY_CONTRIB_RELATION_TABLE : SHORTNAMES.PCR,
#    const.CONTRIBUTOR_TABLE            : SHORTNAMES.CONTRIB,
#}


# FIXME: keys might conflict and need to be ordered
variable_aliases = {
    #'chip_rowid_list': 'cid_list',
    #'annot_rowid_list': 'aid_list',
    #'feature_rowid_list': 'fid_list',
    #

    #'chip_rowid'                  : 'cid',
    'annot_rowid'                 : 'aid',
    #'feat_rowid'                  : 'fid',
    'num_feats'                   : 'nFeat',
    'featweight_forground_weight' : 'fgweight',
    'keypoints'                   : 'kpt_arr',
    'vecs'                        : 'vec_arr',
    'residualvecs'                : 'rvec_arr',
    'verts'                       : 'vert_arr',
    'posixs'                      : 'posix',
    'party_contrib_relation_rowid'  : 'pcr_rowid',
}


PLURAL_FIX_LIST = [('bboxs',    'bboxes'),
                   ('qualitys', 'qualities')]

func_aliases = {
    #'get_feat_vec_lists': 'get_feat_vecs',
    #'get_feat_kpt_lists': 'get_feat_kpts',
}

# mapping to variable names in const


def remove_sentinals(code_text):
    """ Removes template comments and vim sentinals """
    code_text = ut.regex_replace(r'^ *# STARTBLOCK *$\n', '', code_text)
    code_text = ut.regex_replace(r'^ *# ENDBLOCK *$\n?', '', code_text)
    code_text = ut.regex_replace(r'^ *# *REM [^\n]*$\n?', '', code_text)
    code_text = code_text.rstrip()
    return code_text


def format_controller_func(func_code_fmtstr, flagskw, func_type, fmtdict):
    """
    Applys formatting and filtering to function code strings
    Format the template into a function and apply postprocessing

    CommandLine:
        python ibeis/templates/template_generator.py
    """
    func_code = func_code_fmtstr.format(**fmtdict)
    func_code = remove_sentinals(func_code)
    # BOTH OPTIONS ARE NOT GARUENTEED TO WORK. If there are bugs here may be a
    # good place to look.
    if flagskw.get('strip_nparams', False):
        func_code = remove_kwarg('nInput', 'None', func_code)
    if flagskw.get('strip_eager', False):
        func_code = remove_kwarg('eager', 'True', func_code)
    if REMOVE_CONFIG2_:
        func_code = remove_kwarg('config2_', 'None', func_code)
        func_code = func_code.replace('if config2_ is not None', 'if False')
    if STRIP_COMMENTS:
        func_code = ut.strip_line_comments(func_code)
    if flagskw.get('strip_docstr', STRIP_DOCSTR):
        # HACKY: might not always remove docstr correctly. newline HACK away dumb blank line
        func_code = ut.regex_replace('""".*"""\n    ', '', func_code)
    else:
        if STRIP_LONGDESC:
            func_code_lines = func_code.split('\n')
            new_lines = []
            begin = False
            startstrip = False
            finished = False
            for line in func_code_lines:
                if finished is False:
                    # Find the start of the docstr
                    striped_line = line.strip()
                    if not begin and striped_line.startswith('"""') or striped_line.startswith('r"""'):
                        begin = True
                    elif begin:
                        # A blank line signals the start and end of the long
                        # description
                        if len(striped_line) == 0 or striped_line.startswith('"""'):
                            if startstrip is False:
                                # Found first blank line, start stripping
                                startstrip = True
                            else:
                                finished = True
                                #continue
                        elif startstrip:
                            continue
                new_lines.append(line)
            func_code = '\n'.join(new_lines)
        if STRIP_EXAMPLE:
            func_code = ut.regex_replace('Example.*"""', '"""', func_code)
    if USE_ALIASES:
        # Execute search and replaces without changing strings
        func_code = ut.replace_nonquoted_text(func_code,
                                              variable_aliases.keys(),
                                              variable_aliases.values())
    # add decorators
    # HACK IN API_CACHE decorators
    with_api_cache = flagskw.get('with_api_cache', WITH_API_CACHE)
    with_web_api = flagskw.get('with_web_api', WITH_WEB_API)
    if with_api_cache:
        if func_type == '2_Native.getter_col':
            func_code = '@accessor_decors.cache_getter({TABLE}, {COLNAME})\n'.format(**fmtdict) + func_code
        if func_type == '2_Native.deleter':
            func_code = '@accessor_decors.cache_invalidator({TABLE})\n'.format(**fmtdict) + func_code
        if func_type == '2_Native.setter':
            func_code = '@accessor_decors.cache_invalidator({TABLE}, {COLNAME}, native_rowids=True)\n'.format(**fmtdict) + func_code
    if with_web_api:
        if func_type == '2_Native.adder':
            func_code = '@register_route(\'/{tbl}/\', methods=[\'POST\'])\n'.format(**fmtdict) + func_code
        if func_type == '2_Native.getter_col':
            func_code = '@register_route(\'/{tbl}/{col}/\', methods=[\'GET\'])\n'.format(**fmtdict) + func_code
        if func_type == '2_Native.ider':
            func_code = '@register_route(\'/{tbl}/\', methods=[\'GET\'])\n'.format(**fmtdict) + func_code
        if func_type == '2_Native.deleter':
            func_code = '@register_route(\'/{tbl}/\', methods=[\'DELETE\'])\n'.format(**fmtdict) + func_code
        if func_type == '2_Native.setter':
            func_code = '@register_route(\'/{tbl}/{col}/\', methods=[\'PUT\'])\n'.format(**fmtdict) + func_code
    # Need to register all function with ibs
    if flagskw.get('with_decor', WITH_DECOR):
        func_code = '@register_ibs_method\n' + func_code
    # ensure pep8 formating
    if flagskw.get('with_pep8', WITH_PEP8):
        func_code = ut.autofix_codeblock(func_code).strip()
    if ut.VERBOSE:
        print('fmtdict = ' + ut.align(ut.dict_str(fmtdict), ':'))
    return func_code


def get_tableinfo(tablename, ibs=None):
    """
    Gets relevant info from the sql controller and dependency graph
    """
    if ut.NOT_QUIET:
        print('[TEMPLATE] get_tableinfo (ibs=%r)' % (ibs,))
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
            print('[TEMPLATE] WARNING unknown tablename=%r' % tablename)
            print('[TEMPLATE] Known db tables = ' + ut.list_str(valid_db_tablenames))
            print('[TEMPLATE] Known dbcache tables = ' + ut.list_str(valid_dbcache_tablenames))

        if sqldb is not None:
            all_colnames = sqldb.get_column_names(tablename)
            # TODO: handle more than one superkey_colnames
            superkey_colnames_list = sqldb.get_table_superkey_colnames(tablename)
            superkey_colnames = superkey_colnames_list[0]
            primarykey_colnames = sqldb.get_table_primarykey_colnames(tablename)
            other_colnames = sqldb.get_table_otherkey_colnames(tablename)
    if dbself is None:
        dbself = 'dbunknown'
        all_colnames        = []
        superkey_colnames   = []
        primarykey_colnames = []
        other_colnames      = []
        if tablename == const.FEATURE_WEIGHT_TABLE:
            dbself = 'dbcache'
            all_colnames = ['feature_weight_fg']
        if tablename == const.PARTY_CONTRIB_RELATION_TABLE:
            dbself = 'db'
    if tablename == const.RESIDUAL_TABLE:
        other_colnames.append('rvecs')
    # hack out a few colnames
    ignorecolnames = tblname2_ignorecolnames.get(tablename, [])
    other_colnames = [colname for colname in other_colnames
                      if colname not in set(ignorecolnames)]
    tableinfo = (dbself, all_colnames, superkey_colnames, primarykey_colnames, other_colnames)
    return tableinfo


def remove_kwarg(kwname, kwdefault, func_code):
    func_code = ut.regex_replace(r' *>>> *{0} *= *{1} *\n'.format(kwname, kwdefault), '', func_code)
    func_code = ut.regex_replace(r',? *{0} *= *{1}'.format(kwname, kwname), '', func_code)
    for val in kwdefault if isinstance(kwdefault, (list, tuple)) else [kwdefault]:
        func_code = ut.regex_replace(r',? *{0} *= *{1}'.format(kwname, val), '', func_code)
    return func_code


def parse_first_func_name(func_code):
    """ get first function name defined in a codeblock """
    try:
        parse_result = ut.padded_parse('def {func_name}({args}):', func_code)
        assert parse_result is not None
        func_name = parse_result['func_name']
    except AssertionError as ex:
        ut.printex(ex, 'parse result is None', keys=['parse_result', 'func_code'])
        raise
    return func_name


def build_depends_path(child, depends_map):
    #parent = depends_map[child]
    parent = depends_map.get(child, None)
    if parent is not None:
        return build_depends_path(parent, depends_map) + [child]
    else:
        return [child]
    #depends_list = ['annot', 'chip', 'feat', 'featweight']


def postprocess_and_combine_templates(autogen_modname, autogen_key,
                                      constant_list_,
                                      tblname2_functype2_func_list,
                                      flagskw):
    """ Sorts and combines augen function dictionary """
    if ut.VERBOSE:
        print('[TEMPLATE] postprocess_and_combine_templates(%r)' % (autogen_key,))

    func_name_list = []
    func_type_list = []
    func_code_list = []
    func_tbl_list  = []

    #functype_set = set([])
    for tblname, functype2_funclist in six.iteritems(tblname2_functype2_func_list):
        for functype, funclist in six.iteritems(functype2_funclist):
            for func_tup in funclist:
                func_name, func_code = func_tup
                # Append code to flat lists
                func_tbl_list.append(tblname)
                func_type_list.append(functype)
                func_name_list.append(func_name)
                func_code_list.append(func_code)

    if ut.VERBOSE:
        print('[TEMPLATE] len(func_name_list) = %r' % (len(func_name_list,)))

    # sort by multiple values
    #sorted_indexes = ut.list_argsort(func_tbl_list, func_name_list, func_type_list)
    sorted_indexes = ut.list_argsort(func_name_list, func_tbl_list, func_type_list)
    sorted_func_code = ut.list_take(func_code_list, sorted_indexes)
    #sorted_func_name = ut.list_take(func_name_list, sorted_indexes)
    #sorted_func_type = ut.list_take(func_type_list, sorted_indexes)
    #sorted_func_tbl = ut.list_take(func_tbl_list, sorted_indexes)

    #functype_set.add(functype)
    #functype_list = sorted(list(functype_set))

    body_codeblocks = []
    for func_code in sorted_func_code:
        body_codeblocks.append(func_code)

    # --- MORE POSTPROCESSING ---

    # Append const to body
    aligned_constants = '\n'.join(ut.align_lines(sorted(list(set(constant_list_)))))
    autogen_constants = ('# AUTOGENED CONSTANTS:\n' + aligned_constants)
    autogen_constants += '\n\n\n'

    # Make main docstr
    #testable_name_list = ['get_annot_featweight_rowids']
    def make_docstr_main(autogen_modname):
        """ Creates main docstr """
        main_commandline_block_lines = [
            'python -m ' + autogen_modname,
            'python -m ' + autogen_modname + ' --allexamples'
        ]
        main_commandline_block = '\n'.join(main_commandline_block_lines)
        main_commandline_docstr = 'CommandLine:\n' + utool.indent(main_commandline_block, ' ' * 8)
        main_docstr_blocks = [main_commandline_docstr]
        main_docstr_body = '\n'.join(main_docstr_blocks)
        return main_docstr_body

    main_docstr_body = make_docstr_main(autogen_modname)

    # --- CONCAT ---
    # Contenate autogen parts into autogen_text

    if flagskw.get('with_header', True):
        fmtdict = dict(timestamp=ut.get_timestamp('printable'),
                       autogen_key=autogen_key)
        argv_tail = ut.get_argv_tail('template_generator.py')
        argv_tail_str =  ' '.join(argv_tail)
        #print('!!!!!!!')
        #print(argv_tail_str)
        #print('!!!!!!!')
        fmtdict['argv_tail_str1'] = argv_tail_str.replace(' --write', '').replace(' --diff', '') + ' --diff'
        fmtdict['argv_tail_str2'] = argv_tail_str.replace(' --write', '').replace(' --diff', '') + ' --write'
        # Nope the following may not be true:
        #if len(tblname2_functype2_func_list) == 1:
        #    # hack to make this wrt to a single table.
        #    # it is written in the context of multiple
        #    # but should actually just be put into a single
        #    # tables autogenerated funcs?
        #    fmtdict['tbl'] = tblname2_functype2_func_list.keys()[0]
        autogen_header = remove_sentinals(Tdef.Theader_ibeiscontrol.format(**fmtdict))
        autogen_header += '\n\n'
    else:
        autogen_header = ''

    autogen_body = ('\n\n\n'.join(body_codeblocks)) + '\n'

    if flagskw.get('with_footer', True):
        autogen_footer = (
            '\n\n' +
            remove_sentinals(Tdef.Tfooter_ibeiscontrol.format(main_docstr_body=main_docstr_body)) +
            '\n')
    else:
        autogen_footer = ''

    autogen_text = ''.join([
        autogen_header,
        autogen_constants,
        autogen_body,
        autogen_footer,
    ])

    # POSTPROCESSING HACKS:
    #autogen_text = autogen_text.replace('\'feat_rowid\'', '\'feature_rowid\'')
    #autogen_text = ut.regex_replace(r'kptss', 'kpt_lists', autogen_text)
    #autogen_text = ut.regex_replace(r'vecss', 'vec_lists', autogen_text)
    #autogen_text = ut.regex_replace(r'nFeatss', 'nFeat_list', autogen_text)
    #autogen_text = autogen_text.replace('\'feat_rowid\'', '\'feature_rowid\'')

    return autogen_text


def find_valstr(func_code, varname_):
    import re
    assignregex = ''.join((varname_, ' = ', ut.named_field('valstr', '.*')))
    #+ assignregex + '\s*'
    match = re.search(assignregex, func_code)
    if match is None:
        return func_code
    groupdict = match.groupdict()
    valstr = groupdict['valstr']
    return valstr


def replace_constant_varname(func_code, varname, valstr=None):
    """
    Example:
        >>> from ibeis.templates.template_generator import *
        >>> func_code = Tdef.Tsetter_native_multicolumn
        >>> new_func_code = replace_constant_varname(func_code, 'id_iter')
        >>> new_func_code = replace_constant_varname(new_func_code, 'colnames')
        >>> result = new_func_code
        >>> print(result)
    """
    import re
    if func_code.find(varname) == -1:
        return func_code
    varname_ = ut.whole_word(varname)
    if valstr is None:
        valstr = find_valstr(func_code, varname_)
    assignline = ''.join((r'\s*', varname_, ' = .*'))
    new_func_code = re.sub(assignline, '', func_code)
    new_func_code = re.sub(varname_, valstr, new_func_code)
    return new_func_code


def build_templated_funcs(ibs, autogen_modname, tblname_list, autogen_key,
                          flagdefault=True, flagskw={},
                          table_structure={}):
    """ Builds lists of requested functions"""
    print('[TEMPLATE] build_templated_funcs')
    print('  * autogen_modname=%r' % (autogen_modname,))
    print('  * tblname_list=%r' % (tblname_list,))
    print('  * autogen_key=%r' % (autogen_key,))
    print('  * flagdefault=%r' % (flagdefault,))
    print('  * flagskw=%s' % (ut.dict_str(flagskw),))
    #child = 'featweight'
    tblname2_functype2_func_list = ut.ddict(lambda: ut.ddict(list))
    # HACKED IN CONSTANTS
    constant_list_ = [
        'CONFIG_ROWID = \'config_rowid\'',
        'FEATWEIGHT_ROWID = \'featweight_rowid\'',
    ]
    # --- AUTOGENERATE FUNCTION TEXT ---
    for tablename in tblname_list:
        if ut.NOT_QUIET:
            print('[TEMPLATE] building %r table' % (tablename,))
        tableinfo = get_tableinfo(tablename, ibs)
        tup = build_controller_table_funcs(
            tablename, tableinfo,
            autogen_modname,
            autogen_key,
            flagdefault=flagdefault,
            flagskw=flagskw,
            table_structure=table_structure,
            # HACK DONT PASS IBS IN THE FUTURE
            ibs=ibs)
        functype2_func_list, constant_list = tup
        constant_list_.extend(constant_list)
        tblname2_functype2_func_list[tablename] = functype2_func_list
    tfunctup = constant_list_, tblname2_functype2_func_list
    return tfunctup


def get_autogen_modpaths(parent_module, autogen_key='default', flagskw={}):
    """
    Returns info on where the autogen module will be placed if is written
    """
    # Build output filenames and info
    if flagskw['mod_fname'] is not None:
        autogen_mod_fname = flagskw['mod_fname']
    else:
        autogen_mod_fname_fmt = '_autogen_{autogen_key}_funcs.py'
        autogen_mod_fname = autogen_mod_fname_fmt.format(autogen_key=autogen_key)
    # module we will autogenerate next to
    parent_modpath = dirname(parent_module.__file__)
    # Build autogen paths and modnames
    autogen_fpath = join(parent_modpath, autogen_mod_fname)
    autogen_rel_fpath = ut.get_relative_modpath(autogen_fpath)
    autogen_modname = ut.get_modname_from_modpath(autogen_fpath)
    modpath_info = autogen_fpath, autogen_rel_fpath, autogen_modname
    return modpath_info


def build_controller_table_funcs(tablename, tableinfo, autogen_modname,
                                 autogen_key, flagdefault=True, flagskw={},
                                 table_structure={}, ibs=None):
    """
    BIG FREAKING FUNCTION THAT REALIZES TEMPLATES

    Builds function strings for a single type of table using the template
    definitions.

    Returns:
        tuple: (functype2_func_list, constant_list)

    CommandLine:
        python -m ibeis.templates.template_generator
    """
    tbl2_TABLE       = table_structure['tbl2_TABLE']
    tablename2_tbl   = table_structure['tablename2_tbl']
    depends_map      = table_structure['depends_map']
    relationship_map = table_structure['relationship_map']
    externtbl_map    = table_structure['externtbl_map']
    tbl2_tablename   = table_structure['tbl2_tablename']

    if ut.VERBOSE:
        print('[TEMPLATE] build_controller_table_funcs(%r)' % (tablename,))
    # +-----
    # Setup
    # +-----

    (dbself, all_colnames, superkey_colnames, primarykey_colnames, other_colnames) = tableinfo

    # not sure if this is kosher
    #other_colnames += superkey_colnames
    nonprimary_leaf_colnames = ut.setdiff_ordered(all_colnames, primarykey_colnames)
    leaf_other_propnames = ', '.join(other_colnames)
    #leaf_other_propname_lists = ', '.join([colname + '_list' for colname in other_colnames])
    # for the preproc_tbe.compute... method
    leaf_props = '_'.join(other_colnames)
    # TODO: handle more than one superkey_colnames
    superkey_args = ', '.join([colname + '_list' for colname in superkey_colnames])
    superkey_COLNAMES = ', '.join([colname.upper() for colname in superkey_colnames])

    # WE WILL DEFINE SEVERAL CLOSURES THAT USE THIS DICTIONARY
    fmtdict = {
    }

    allcols = [colname for colname in all_colnames if colname not in primarykey_colnames]
    allcol_items = ', '.join(allcols)
    allcol_args = ', '.join([colname + '_list' for colname in allcols])
    allCOLNAMES = ', '.join([colname.upper() for colname in allcols])

    fmtdict['allcol_args'] = allcol_args
    fmtdict['allcol_items'] = allcol_items
    fmtdict['allCOLNAMES'] = allCOLNAMES

    fmtdict['nonprimary_leaf_colnames'] = nonprimary_leaf_colnames
    fmtdict['autogen_modname'] = autogen_modname
    fmtdict['autogen_key'] = autogen_key
    fmtdict['leaf_other_propnames'] = leaf_other_propnames
    #fmtdict['leaf_other_propname_lists'] = leaf_other_propname_lists
    fmtdict['leaf_props'] = leaf_props
    fmtdict['superkey_args'] = superkey_args
    fmtdict['superkey_COLNAMES'] = superkey_COLNAMES
    fmtdict['self'] = 'ibs'
    fmtdict['dbself'] = dbself

    CONSTANT_COLNAMES = []
    CONSTANT_COLNAMES.extend(other_colnames)
    functype2_func_list = ut.ddict(list)
    constant_list = []
    #constant_varname_list = []
    #constant_varstr_list = []
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

    def set_col(col, COLNAME):
        fmtdict['COLNAME'] = COLNAME
        #print(col)
        #if col == 'feature_num_feats':
        #    ut.embed()
        #ut.embed()
        # <HACK>
        # For getting column names into the shortname format
        # feature is actually features which is why this doesnt work
        # </HACK>
        col = col.replace('feature', 'feat')

        fmtdict['col'] = col

    def set_multicol(multicol, MULTICOLNAMES):
        fmtdict['MULTICOLNAMES'] = str(multicolnames)
        fmtdict['multicol'] = multicol

    def set_relation_tables(relation_tbl, relation_tables):
        tbl1, tbl2 = relation_tables
        #print('---')
        #print('tbl1 = %r' % (tbl1,))
        #print('tbl2 = %r' % (tbl2,))
        fmtdict['relation_tbl'] = relation_tbl
        fmtdict['RELATION_TABLE'] = tbl2_TABLE[relation_tbl]
        fmtdict['tbl1'] = tbl1
        fmtdict['tbl2'] = tbl2
        fmtdict['TABLE1'] = tbl2_TABLE[tbl1]
        fmtdict['TABLE2'] = tbl2_TABLE[tbl2]
    # L____________________________

    # +----------------------------
    # | Template appenders
    # +----------------------------
    def append_constant(varname, valstr):
        """ Used for rowid and colname const """
        const_fmtstr = ''.join((varname, ' = \'', valstr, '\''))
        const_line = const_fmtstr.format(**fmtdict)
        constant_list.append(const_line)
        #constant_varname_list.append(varname)
        #constant_varstr_list.append(valstr)

    def append_func(func_type, func_code_fmtstr):
        """
        Filters, formats, and organizes functions as they are added
        applys hacks


        func_type = '2_Native.setter'
        func_code_fmtstr = Tdef.Tsetter_native_column

        """
        #if ut.VERBOSE:
        #    print('[TEMPLATE] append_func()')
        #if func_type.find('add') < 0:
        #    return
        #type1, type2 = func_type.split('.')
        func_type = func_type
        try:
            # Format the template into a function and apply postprocessing
            func_code = format_controller_func(func_code_fmtstr, flagskw, func_type, fmtdict)
            # HACK to remove double table names like: get_chip_chip_width
            for single_tbl in ut.filter_Nones((fmtdict['tbl'], fmtdict.get('externtbl', None))):
                #single_tbl = fmtdict['tbl']
                double_tbl = single_tbl + '_' + single_tbl
                func_code = func_code.replace(double_tbl, single_tbl)
            # HACK for plural bbox
            for bad_plural, good_plural in PLURAL_FIX_LIST:
                func_code = func_code.replace(bad_plural, good_plural)
            # ENDHACK
            # parse out function name
            func_name = parse_first_func_name(func_code)

            funcname_filter = flagskw['funcname_filter']
            if funcname_filter is not None:
                import re
                if re.search(funcname_filter, func_name) is None:
                    return
            #if func_name == 'get_featweight_fgweights':
            # <HACKS>
            #print(tablename)
            if tablename == const.ANNOTATION_TABLE:
                func_code = func_code.replace('ENABLE_DOCTEST', 'DISABLE_DOCTEST')
            elif tablename == const.FEATURE_WEIGHT_TABLE:
                if func_name in ['get_annot_featweight_rowids', 'add_feat_featweights', 'add_annot_featweights']:
                    func_code = func_code.replace('ENABLE_DOCTEST', 'SLOW_DOCTEST')

            #replace_constants =  True
            replace_constants =  False
            if replace_constants:
                varname_list = ['id_iter', 'colnames', 'superkey_paramx', 'andwhere_colnames']
                for varname in varname_list:
                    func_code = replace_constant_varname(func_code, varname)
                #for varname, valstr in zip(constant_varname_list, constant_varstr_list):
                #for const_line in constant_list:
                #    varname, valstr = const_line.split(' = ')
                #    print(varname)
                #    func_code = replace_constant_varname(func_code, varname, valstr)

            # </HACKS>
            #
            #
            # Register function
            func_tup = (func_name, func_code)
            functype2_func_list[func_type].append(func_tup)
            # Register function aliases
            if func_name in func_aliases:
                func_aliasname = func_aliases[func_name]
                func_aliascode = ut.codeblock('''
                @register_ibs_method
                def {func_aliasname}(*args, **kwargs):
                    return {func_name}(*args, **kwargs)
                ''').format(func_aliasname=func_aliasname, func_name=func_name)
                func_aliastup = (func_aliasname, func_aliascode)
                functype2_func_list[func_type].append(func_aliastup)
        except Exception as ex:
            utool.printex(ex, keys=['func_type', 'tablename'])
            raise
    # L____________________________

    # +----------------------------
    # | Higher level helper functions
    # +----------------------------
    def build_rowid_constants(depends_list):
        """ Ensures all rowid const have been added corectly """
        for tbl_ in depends_list:
            set_tbl(tbl_)
            # HACK: fix feature column names in dbschema
            constval_fmtstr = '{tbl}_rowid' if tbl_ != 'feat' else 'feature_rowid'
            append_constant('{TBL}_ROWID', constval_fmtstr)

    def build_pc_dependant_lines(depends_list):
        """
        builds parent child dependency function chains for pc_line dependent
        templates
        """
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

    multicol_list = multicolumns_dict.get(tablename, [])

    def is_disabled_by_multicol(colname):
        for multicoltup in multicol_list:
            if len(multicoltup) == 3 and multicoltup[2]:
                invalid_colnames = multicoltup[1]
                if colname in invalid_colnames:
                    return True
        return False
    # L____________________________

    tbl = tablename2_tbl[tablename]
    # Build dependency path
    depends_list    = build_depends_path(tbl, depends_map)

    #=========================================
    # THIS IS WHERE THE TEMPLATES ARE FORMATED
    #=========================================
    with_getters      = flagskw.get('with_getters', flagdefault)
    with_setters      = flagskw.get('with_setters', flagdefault)
    with_iders        = flagskw.get('with_iders', flagdefault)
    with_adders       = flagskw.get('with_adders', flagdefault)
    with_deleters     = flagskw.get('with_deleters', flagdefault)
    with_fromsuperkey = flagskw.get('with_fromsuperkey', flagdefault)
    with_configs      = flagskw.get('with_configs', flagdefault)

    with_columns      = flagskw.get('with_columns', flagdefault)
    with_multicolumns = flagskw.get('with_multicolumns', flagdefault)
    with_parentleaf   = flagskw.get('with_parentleaf', flagdefault)
    with_rootleaf     = flagskw.get('with_rootleaf', flagdefault)
    with_native       = flagskw.get('with_native', flagdefault)
    with_relations    = flagskw.get('with_relations', flagdefault)

    # Setup
    build_rowid_constants(depends_list)
    set_tbl(tbl)
    build_pc_dependant_lines(depends_list)

    # -----------------------
    #  Parent Leaf Dependency
    # -----------------------
    if len(depends_list) > 1 and with_parentleaf:
        if len(depends_list) == 2:
            set_root_leaf(depends_list[0], depends_list[-1], depends_list[0])
        else:
            set_root_leaf(depends_list[0], depends_list[-1], depends_list[-2])
        if with_adders:
            append_func('0_PL.adder',   Tdef.Tadder_pl_dependant)
        if with_deleters:
            append_func('0_PL.deleter', Tdef.Tdeleter_pl_depenant)
        if with_getters:
            append_func('0_PL.getter_rowids_',  Tdef.Tgetter_pl_dependant_rowids_)
            append_func('0_PL.getter_rowids',   Tdef.Tgetter_pl_dependant_rowids)

    # ----------------------------
    # Root Leaf Dependancy
    # ----------------------------
    if len(depends_list) > 2 and with_rootleaf:
        set_root_leaf(depends_list[0], depends_list[-1], depends_list[-2])
        if with_adders:
            append_func('1_RL.adder',   Tdef.Tadder_rl_dependant)
        if with_deleters:
            append_func('1_RL.deleter', Tdef.Tdeleter_rl_depenant)
        if with_iders:
            append_func('1_RL.ider',    Tdef.Tider_rl_dependant_all_rowids)
        if with_getters:
            append_func('1_RL.getter_rowids',  Tdef.Tgetter_rl_dependant_rowids)

    # ----------------------------
    # Many to Many Relationships
    # ----------------------------
    if with_relations:
        if ut.VERBOSE:
            print('[TEMPLATE]  * Building many-to-many relationships')
        relation_tables = relationship_map.get(tbl, None)
        if relation_tables is not None:
            (tbl1, tbl2) = relation_tables
            set_relation_tables(tbl, relation_tables)
            if tbl1 != tbl2:
                # FIXME: hack
                if with_adders:
                    append_func('3_RELATE.adder', Tdef.Tadder_relationship)
                # Add both directions in relationships
                for count, direction in enumerate([1, -1], start=1):
                    relation_tables_ = relationship_map.get(tbl, None)[::direction]
                    (tbl1, tbl2) = relation_tables_
                    set_relation_tables(tbl, relation_tables_)
                    if with_deleters:
                        append_func('3_RELATE{count}.deleter'.format(count=count), Tdef.Tdeleter_table1_relation)
                    if with_getters:
                        append_func('3_RELATE{count}.getter'.format(count=count), Tdef.Tgetter_table1_rowids)
                        pass

    # ------------------
    #  Native Noncolumn
    # ------------------
    if with_native:
        if ut.VERBOSE:
            print('[TEMPLATE]  * Building native non-column funcs')
        if with_deleters:
            append_func('2_Native.deleter', Tdef.Tdeleter_native_tbl)
        if with_iders:
            append_func('2_Native.ider', Tdef.Tider_all_rowids)
        if with_fromsuperkey:
            append_func('2_Native.fromsuperkey_getter', Tdef.Tgetter_native_rowid_from_superkey)
        if with_adders:
            append_func('2_Native.adder',   Tdef.Tadder_native)
        # Only dependants have native configs
        if len(depends_list) > 1 and with_configs:
            append_func('2_Native.config_getter', Tdef.Tcfg_rowid_getter)

        with_extern = True
        if with_extern:
            # many to one table relationships
            extern_tables = externtbl_map[tbl]
            if extern_tables is not None:
                for extern_tbl in extern_tables:
                    fmtdict['externtbl'] = extern_tbl
                    extern_table = tbl2_tablename[extern_tbl]
                    tup = get_tableinfo(extern_table, ibs)
                    extern_dbself, extern_all_colnames, extern_superkey_colnames, extern_primarykey_colnames, extern_other_colnames = tup
                    externcol_list = list(extern_superkey_colnames) + list(extern_other_colnames)
                    for externcol in externcol_list:
                        fmtdict['externcol'] = externcol
                        #constant_list.append(externcol.upper() + ' = \'%s\'' % (externcol,))
                        append_func('4_Extern.getter', Tdef.Tgetter_extern)

    # ------------------
    #  Column Properties
    # ------------------

    with_col_rootleaf = len(depends_list) > 1 and with_rootleaf

    if with_columns:
        if ut.VERBOSE:
            print('[TEMPLATE]  * Building column funcs len(other_colnames) = %r' % (len(other_colnames),))
        # For each column property
        def col_generator(_list):
            # FIXME: clean up this generator func
            for colname in _list:
                COLNAME = colname.upper()
                if is_disabled_by_multicol(colname):
                    continue
                set_col(colname, COLNAME)
                constant_list.append(COLNAME + ' = \'%s\'' % (colname,))
                append_constant(COLNAME, colname)
                yield colname
        if with_getters and with_col_rootleaf:
            for colname in col_generator(list(other_colnames)):
                append_func('1_RL.getter_col', Tdef.Tgetter_rl_pclines_dependant_column)
        if with_getters and with_native:
            for colname in col_generator(list(other_colnames) + list(superkey_colnames)):
                if colname == 'config_rowid':
                    # HACK
                    continue
                append_func('2_Native.getter_col', Tdef.Tgetter_table_column)
        if with_setters and with_native and  tablename not in readonly_set:
            # Setter template: columns
            for colname in col_generator(list(other_colnames)):
                append_func('2_Native.setter', Tdef.Tsetter_native_column)

    if with_multicolumns:
        # For each multicolumn property
        for multicoltup in multicol_list:
            multicol, multicolnames = multicoltup[0:2]
            set_multicol(multicol, multicolnames)
            if with_getters:
                # Getter template: multicolumns
                if with_col_rootleaf:
                    append_func('RL.getter_mutli_dependant', Tdef.Tgetter_rl_pclines_dependant_multicolumn)
                if with_native:
                    append_func('2_Native.getter_multi_native', Tdef.Tgetter_native_multicolumn)
            if with_setters and  tablename not in readonly_set:
                # Setter template: columns
                if with_native:
                    append_func('2_Native.setter_multi_native', Tdef.Tsetter_native_multicolumn)

    return functype2_func_list, constant_list


def get_autogen_text(
        parent_module,
        tblname_list=TBLNAME_LIST,
        autogen_key='default',
        flagdefault=True,
        flagskw={},
        table_structure={}):
    """
    autogenerated text main entry point

    Returns:
        tuple : (autogen_fpath, autogen_text)

    CommandLine:
        python ibeis/templates/template_generator.py
    """
    print('[TEMPLATE] get_autogen_text()')
    # Filepath info
    modpath_info = get_autogen_modpaths(parent_module, autogen_key, flagskw)
    autogen_fpath, autogen_rel_fpath, autogen_modname = modpath_info
    # Build functions and constant containers
    tfunctup = build_templated_funcs(
        ibs, autogen_modname, tblname_list, autogen_key,
        flagdefault=flagdefault, flagskw=flagskw,
        table_structure=table_structure)
    constant_list_, tblname2_functype2_func_list = tfunctup
    # Combine into a text file
    autogen_text = postprocess_and_combine_templates(
        autogen_modname, autogen_key, constant_list_, tblname2_functype2_func_list, flagskw)
    # Return path and text
    return autogen_fpath, autogen_text


def parse_table_structure(ibs):
    print('[TEMPLATE] parse_table_structure()')
    # hack tablenames to be singular
    import re
    keep_plural_hacks = ['species']
    ignore_table_hacks = ['keys', 'metadata']

    def get_tablename_tbl(db, tablename):
        shortname = db.get_metadata_val(tablename + '_shortname', eval_=True)
        if shortname is not None:
            tbl = shortname
        else:
            if tablename not in keep_plural_hacks:
                tbl = re.sub('s$', '', tablename)
            else:
                tbl = tablename
        return tbl

    db_tablename_list = [tablename for tablename in ibs.db.get_table_names() if tablename not in ignore_table_hacks]
    dbcache_tablename_list = [tablename for tablename in ibs.dbcache.get_table_names() if tablename not in ignore_table_hacks]

    tablename2_tbl = {tablename: get_tablename_tbl(ibs.db, tablename) for tablename in db_tablename_list}
    tablename2_tbl.update({tablename: get_tablename_tbl(ibs.dbcache, tablename) for tablename in dbcache_tablename_list})
    # more hacks
    tablename2_tbl[const.FEATURE_WEIGHT_TABLE] = 'featweight'
    tablename2_tbl[const.ANNOTATION_TABLE] = 'annot'
    tablename2_tbl[const.FEATURE_TABLE] = 'feat'
    #
    tbl2_tablename = ut.invert_dict(tablename2_tbl)
    db_tbl_list      = [tablename2_tbl.get(tablename, tablename) for tablename in db_tablename_list]
    dbcache_tbl_list = [tablename2_tbl.get(tablename, tablename) for tablename in dbcache_tablename_list]

    # Parse dependencies out of the SQL Schemas
    def get_tbl_depends(db, tbl):
        tablename = tbl2_tablename[tbl]
        depends = db.get_metadata_val(tablename + '_dependson', eval_=True)
        if depends is None:
            return None
        if isinstance(depends, six.string_types):
            return tablename2_tbl.get(depends, depends)
        return depends
    depends_map      = {tbl: get_tbl_depends(ibs.db, tbl) for tbl in db_tbl_list}
    depends_map.update({tbl: get_tbl_depends(ibs.dbcache, tbl) for tbl in dbcache_tbl_list})

    def get_tbl_relationship(tbl, db):
        tablename = tbl2_tablename[tbl]
        relates = db.get_metadata_val(tablename + '_relates', eval_=True)
        if relates is not None:
            relates = ut.dict_take(tablename2_tbl, relates)
        return relates

    def get_tbl_externtbls(tbl, db):
        tablename = tbl2_tablename[tbl]
        externtbls = db.get_metadata_val(tablename + '_extern_tables', eval_=True)
        if externtbls is not None:
            externtbls = ut.dict_take(tablename2_tbl, externtbls)
        return externtbls

    # Parse relationships out of the SQL Schemas
    relationship_map = {tbl: get_tbl_relationship(tbl, ibs.db)
                        for tbl in db_tbl_list}
    relationship_map.update({tbl: get_tbl_relationship(tbl, ibs.dbcache)
                             for tbl in dbcache_tbl_list})
    # Parse the many to one relationships
    externtbl_map      = {tbl: get_tbl_externtbls(tbl, ibs.db)
                          for tbl in db_tbl_list}
    externtbl_map.update({tbl: get_tbl_externtbls(tbl, ibs.dbcache)
                          for tbl in dbcache_tbl_list})

    import operator
    # I'm not sure why is is not working
    tbl2_TABLE = {key: 'const.' + ut.get_varname_from_locals(val, const.__dict__, cmpfunc_=operator.eq)
                    for key, val in six.iteritems(tbl2_tablename)}

    table_structure = {
        'tablename2_tbl'   : tablename2_tbl,
        'depends_map'      : depends_map,
        'relationship_map' : relationship_map,
        'tbl2_tablename'   : tbl2_tablename,
        'tbl2_TABLE'       : tbl2_TABLE,
        'externtbl_map'    : externtbl_map,
    }
    print('table_structure = ' + ut.dict_str(table_structure))

    return table_structure


def main(ibs, verbose=None):
    """
    MAIN FUNCTION

    CommandLine:
        python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'ibeis.templates.template_generator')"

        sh Tgen.sh --tbls annotations --Tcfg with_getters:True strip_docstr:True
        sh Tgen.sh --tbls annotations --tbls annotations --Tcfg with_getters:True strip_docstr:False with_columns:False

        sh Tgen.sh --key featweight
        sh Tgen.sh --key annot --onlyfn
        sh Tgen.sh --key featweight --onlyfn
        sh Tgen.sh --key chip --onlyfn --Tcfg with_setters=False
        sh Tgen.sh --key chip --Tcfg with_setters=False
        sh Tgen.sh --key chip --Tcfg with_setters=False with_getters=False with_adders=True
        sh Tgen.sh --key feat --onlyfn

        python -m ibeis.templates.template_generator
        python -m ibeis.templates.template_generator --dump-autogen-controller
        gvim ibeis/templates/_autogen_default_funcs.py
        python dev.py --db testdb1 --cmd
        %run dev.py --db testdb1 --cmd
    """
    print('\n\n[TEMPLATE] main()')
    # Parse command line args
    onlyfuncname = ut.get_argflag(('--onlyfuncname', '--onlyfn'),
                                  help_='if specified only prints the function signatures')
    dowrite = ut.get_argflag(('-w', '--write', '--dump-autogen-controller'))
    show_diff = ut.get_argflag('--diff')
    num_context_lines = ut.get_argval('--diff', type_=int, default=None)
    show_diff = show_diff or num_context_lines is not None
    dowrite = dowrite and not show_diff
    autogen_key = ut.get_argval(('--key',), type_=str, default='default')

    table_structure = parse_table_structure(ibs)

    if verbose is None:
        verbose = not dowrite

    if autogen_key == 'default':
        default_tblname_list = TBLNAME_LIST
    else:
        tbl2_tablename = table_structure['tbl2_tablename']
        tablename2_tbl = table_structure['tablename2_tbl']
        if autogen_key in tbl2_tablename:
            default_tblname_list = [tbl2_tablename[autogen_key], ]
        elif autogen_key in tablename2_tbl:
            default_tblname_list = [autogen_key, ]
        else:
            raise AssertionError('unknown autogen_key=%r. known tables are %r' %
                                 (autogen_key, list(tbl2_tablename.keys())))

    flagskw = {}
    tblname_list = ut.get_argval(('--autogen-tables', '--tbls'), type_=list, default=default_tblname_list)
    #print(tblname_list)
    # Parse dictionary flag list
    template_flags = ut.get_argval(('--Tcfg', '--template-config'), type_=list, default=[])
    flagskw['funcname_filter']  = ut.get_argval(('--funcname-filter', '--fnfilt'), type_=str, default=None)
    flagskw['mod_fname']  = ut.get_argval(('--mod-fname', '--modfname'), type_=str, default=None)

    # Processes command line args
    if len(template_flags) > 0:
        flagdefault = True
        flagskw['with_decor'] = flagdefault
        flagskw['with_footer'] = flagdefault
        flagskw['with_header'] = flagdefault
        flagskw.update(ut.parse_cfgstr_list(template_flags))
        for flag in six.iterkeys(flagskw):
            if flagskw[flag] in ['True', 'On', '1']:
                flagskw[flag] = True
            elif flagskw[flag] in ['False', 'Off', '0']:
                flagskw[flag] = False
            else:
                pass
                #flagskw[flag] = False
    else:
        flagdefault = True
    #flagskw = ut.parse_dict_from_argv(flagskw)

    for tblname in tblname_list:
        assert tblname in tablename2_tbl

    parent_module = ibeis.control

    # Autogenerate text
    autogen_fpath, autogen_text = get_autogen_text(
        parent_module, tblname_list=tblname_list, autogen_key=autogen_key,
        flagdefault=flagdefault, flagskw=flagskw,
        table_structure=table_structure)

    print('[TEMPLATE] Finished text generation...')

    # output to disk or stdout
    if onlyfuncname:
        text = ('\n'.join([line for line in autogen_text.splitlines() if line.startswith('def ')]))
        ut.print_python_code(text)
    else:
        if not ut.QUIET and (not dowrite or verbose):
            print('[TEMPLATE] Dumping autogenerated text...\n+---\n')
            if not dowrite:
                ut.print_python_code(autogen_text)
                print('\nL___\n...would write to: %s' % autogen_fpath)
            if show_diff:
                if ut.checkpath(autogen_fpath):
                    prev_text = ut.read_from(autogen_fpath)
                    textdiff = ut.util_str.get_textdiff(prev_text, autogen_text, num_context_lines=num_context_lines)
                    #textdiff = ut.util_str.get_textdiff(prev_text, autogen_text, num_context_lines=None)
                    ut.print_difftext(textdiff)
                pass
    if dowrite:
        ut.write_to(autogen_fpath, autogen_text)

    #ibs.db.print_table_csv('metadata', exclude_columns=['metadata_value'])
    #ibs.dbcache.print_table_csv('metadata', exclude_columns=['metadata_value'])
    #ibs.dbcache.print_table_csv('metadata')

    #return locals()


if __name__ == '__main__':
    """
    CommandLine:
        python ibeis/templates/template_generator.py
        python ibeis/templates/template_generator.py --dump-autogen-controller

        python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'ibeis.templates.template_generator')"
        chmod +x Tgen.sh

        Tgen.sh --tbls annotations --Tcfg with_getters:True strip_docstr:True
        Tgen.sh --tbls annotations --tbls annotations --Tcfg with_getters:True strip_docstr:False with_columns:False

        sh Tgen.sh --tbls encounters --Tcfg with_getters:True with_setters=True strip_docstr:False
    """
    if 'ibs' not in vars():
        import ibeis
        ibs = ibeis.opendb('emptydatabase', allow_newdir=True, delete_ibsdir=True)
    main(ibs)
    #exec(ut.execstr_dict(locals_))
