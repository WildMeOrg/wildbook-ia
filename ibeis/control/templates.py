from __future__ import absolute_import, division, print_function
import six
import utool as ut
from ibeis import constants


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
    'num_feats': 'nFeats',
    'forground_weight': 'fgweight',
    'keypoints': 'kpts',
    'vectors': 'vecs',
    'residualvecs': 'rvecs',
}


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


def format_controller_func(func_code):
    STRIP_DOCSTR   = True
    USE_SHORTNAMES = True

    if STRIP_DOCSTR:
        # might not always work the newline is a hack to remove
        # that dumb blank line
        func_code = ut.regex_replace('""".*"""\n    ', '', func_code)
    #func_code = ut.regex_replace('\'[^\']*\'', '', func_code)
    if USE_SHORTNAMES:
        # Cannot do this until quoted strings are preserved
        # This should hack it in
        def preserve_quoted_str(quoted_str):
            #print(quoted_str)
            return '\'' + '_'.join(list(quoted_str[1:-1])) + '\''
        def unpreserve_quoted_str(quoted_str):
            #print(quoted_str)
            return '\'' + ''.join(list(quoted_str[1:-1])[::2]) + '\''
        func_code = ut.modify_quoted_strs(func_code, preserve_quoted_str)
        for varname, alias in six.iteritems(variable_aliases):
            func_code = func_code.replace(varname, alias)
        func_code = ut.modify_quoted_strs(func_code, unpreserve_quoted_str)
    func_code = ut.autofix_codeblock(func_code).strip()
    #func_code = ut.indent(func_code)
    func_code = '@register_ibs_method\n' + func_code
    return func_code


def build_dependent_controller_funcs(tablename, other_colnames, all_colnames, dbself):
    child = tablename2_tbl[tablename]
    depends_list = build_depends_path(child)

    CONSTANT_COLNAMES = []

    fmtdict = {
        'self': 'ibs',
        #'parent': None,
        #'child':  None,
        #'CHILD':  None,
        #'COLNAME': None,  # 'FGWEIGHTS',
        #'parent_rowid_list': 'aid_list',
        'dbself': dbself,
        #'TABLE': None,
        #'FEATURE_TABLE',
    }
    functype2_func_list = ut.ddict(list)
    constant_list = []

    def append_func(func_code, func_type):
        func_code = format_controller_func(func_code)
        functype2_func_list[func_type].append(func_code)

    # Getter template: config_rowid
    for parent, child in ut.itertwo(depends_list):
        fmtdict['parent'] = parent
        fmtdict['child'] = child
        fmtdict['PARENT'] = parent.upper()
        fmtdict['CHILD'] = child.upper()
        fmtdict['TABLE'] = tbl2_TABLE[child]  # tblname1_TABLE[child]
        append_func(getter_template_dependant_primary_rowid.format(**fmtdict), 'child_rowids')

    CONSTANT_COLNAMES.extend(other_colnames)

    other_cols = list(map(lambda colname: colname2_col(colname, tablename), other_colnames))
    other_COLNAMES = list(map(lambda colname: colname.upper(), other_colnames))

    for colname, col, COLNAME in zip(other_colnames, other_cols, other_COLNAMES):
        fmtdict['COLNAME'] = COLNAME
        fmtdict['col'] = col
        # Getter template: dependant columns
        for parent, child in ut.itertwo(depends_list):
            fmtdict['parent'] = parent
            fmtdict['PARENT'] = parent.upper()
            fmtdict['child'] = child
            fmtdict['TABLE'] = tbl2_TABLE[child]  # tblname1_TABLE[child]
            append_func(getter_template_dependant_column.format(**fmtdict), 'dependant_property')
        # Getter template: native (Level 0) columns
        fmtdict['tbl'] = child  # tblname is the last child in dependency path
        fmtdict['TABLE'] = tbl2_TABLE[child]
        append_func(getter_template_native_column.format(**fmtdict), 'native_property')
        constant_list.append(COLNAME + ' = \'%s\'' % (colname,))
        constant_list.append('{CHILD}_ROWID = \'{child}_rowid\''.format(child=child, CHILD=child.upper()))
        constant_list.append('{PARENT}_ROWID = \'{parent}_rowid\''.format(parent=parent, PARENT=parent.upper()))

    fmtdict['child_colnames'] = all_colnames
    append_func(adder_template_dependant_child.format(**fmtdict), 'stubs')
    append_func(getter_template_table_config_rowid.format(**fmtdict), 'config_rowid')

    return functype2_func_list, constant_list


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


def main(ibs):
    """
    CommandLine:
        python dev.py --db testdb1 --cmd
        %run dev.py --db testdb1 --cmd
    """
    tblname_list = [constants.CHIP_TABLE,
                    constants.FEATURE_TABLE,
                    constants.FEATURE_WEIGHT_TABLE,
                    constants.RESIDUAL_TABLE
                    ]
    #child = 'featweight'
    tblname2_functype2_func_list = ut.ddict(lambda: ut.ddict(list))
    constant_list_ = [
        'CONFIG_ROWID = \'config_rowid\'',
        'FEATWEIGHT_ROWID = \'featweight_rowid\'',
    ]
    for tablename in tblname_list:
        tableinfo = get_tableinfo(tablename, ibs)
        (dbself, all_colnames, superkey_colnames, primarykey_colnames, other_colnames) = tableinfo

        functype2_func_list, constant_list = build_dependent_controller_funcs(tablename, other_colnames, all_colnames, dbself)
        constant_list_.extend(constant_list)
        tblname2_functype2_func_list[tablename] = functype2_func_list

    functype_set = set([])
    for tblname, val in six.iteritems(tblname2_functype2_func_list):
        for functype in six.iterkeys(val):
            functype_set.add(functype)
    functype_list = list(functype_set)

    body_codeblocks = []

    # Append constants to body
    aligned_constants = '\n'.join(ut.align_lines(sorted(list(set(constant_list_)))))
    body_codeblocks.append('# AUTOGENED CONSTANTS:\n' + aligned_constants)

    # Append functions to body
    seen = set([])
    for count1, functype in enumerate(functype_list):
        functype_codeblocks = []
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
            functype_table_section_header = ut.codeblock(
                '''
                #
                # {functype} {tblname}
                '''
            ).format(functype=functype, tblname=tblname)
            functype_codeblocks.append(functype_table_section_header)
            for func_codeblock in val[functype]:
                if func_codeblock in seen:
                    continue
                seen.add(func_codeblock)
                functype_codeblocks.append(func_codeblock)
        body_codeblocks.extend(functype_codeblocks)

    from os.path import dirname, join
    autogen_fpath = join(ut.truepath(dirname(ibeis.control.__file__)), '_autogen_ibeiscontrol_funcs.py')

    autogen_header = ut.codeblock(
        '''
        # AUTOGENERATED ON {timestamp}
        from __future__ import absolute_import, division, print_function
        import functools
        import six  # NOQA
        from six.moves import map, range  # NOQA
        from ibeis import constants
        from ibeis.control.IBEISControll import IBEISController
        import utool  # NOQA
        import utool as ut  # NOQA
        print, print_, printDBG, rrr, profile = ut.inject(__name__, '[autogen_ibsfuncs]')
        register_ibs_method = ut.classmember(IBEISController)
        '''
    ).format(timestamp=ut.get_timestamp('printable'))
    #from ibeis.constants import (IMAGE_TABLE, ANNOTATION_TABLE, LBLANNOT_TABLE,
    #                             ENCOUNTER_TABLE, EG_RELATION_TABLE,
    #                             AL_RELATION_TABLE, GL_RELATION_TABLE,
    #                             CHIP_TABLE, FEATURE_TABLE, LBLIMAGE_TABLE,
    #                             CONFIG_TABLE, CONTRIBUTOR_TABLE, LBLTYPE_TABLE,
    #                             METADATA_TABLE, VERSIONS_TABLE, __STR__)

    autogen_body = '\n\n'
    autogen_body += ('\n\n\n'.join(body_codeblocks))

    autogen_text = '\n'.join([autogen_header, autogen_body, ''])
    #print(autogen_text)

    ut.write_to(autogen_fpath, autogen_text)

    return locals()


# Adders

#where_clause = {PARENT}_ROWID + '=? AND ' + CONFIG_ROWID + '=?'
# Template for a child rowid that depends on a parent rowid + a config
getter_template_dependant_primary_rowid = ut.codeblock(
    '''
    def get_{parent}_{child}_rowids({self}, {parent}_rowid_list,
                                    config_rowid=None, all_configs=False,
                                    ensure=True, eager=True,
                                    num_params=None):
        """
        get_{parent}_{child}_rowids

        get {child} rowids of {parent} under the current state configuration


        Args:
            {parent}_rowid_list (list):

        Returns:
            list: {child}_rowid_list
        """
        if ensure:
            {self}.add_{child}s({parent}_rowid_list)
        if config_rowid is None:
            config_rowid = {self}.get_{child}_config_rowid()
        colnames = ({CHILD}_ROWID,)
        if all_configs:
            config_rowid = {self}.{dbself}.get(
                {TABLE}, colnames, {parent}_rowid_list,
                id_colname={PARENT}_ROWID, eager=eager, num_params=num_params)
        else:
            config_rowid = {self}.get_{child}_config_rowid()
            andwhere_colnames = [{PARENT}_ROWID, CONFIG_ROWID]
            params_iter = (({parent}_rowid, config_rowid,) for {parent}_rowid in {parent}_rowid_list)
            {child}_rowid_list = {self}.{dbself}.get_where2(
                {TABLE}, colnames, params_iter, andwhere_colnames, eager=eager,
                num_params=num_params)
        return {child}_rowid_list
    ''')


# Config

getter_template_table_config_rowid = ut.codeblock(
    '''
    def get_{child}_config_rowid({self}):
        """
        returns config_rowid of the current configuration

        """
        {child}_cfg_suffix = {self}.cfg.{child}_cfg.get_cfgstr()
        cfgsuffix_list = [{child}_cfg_suffix]
        {child}_cfg_rowid = {self}.add_config(cfgsuffix_list)
        return {child}_cfg_rowid
    '''
)


# Adders

adder_template_dependant_child = ut.codeblock(
    '''
    def add_{parent}_{child}({self}, {parent}_rowid_list, config_rowid=None):
        """
        Adds / ensures / computes a dependent property

        returns config_rowid of the current configuration
        """
        raise NotImplementedError('this code is a stub, you must populate it')
        if config_rowid is None:
            config_rowid = {self}.get_{child}_config_rowid()
        {child}_rowid_list = ibs.get_{parent}_{child}_rowids(
            {parent}_rowid_list, config_rowid=config_rowid, ensure=False)
        dirty_{parent}_rowid_list = utool.get_dirty_items({parent}_rowid_list, {child}_rowid_list)
        if len(dirty_{parent}_rowid_list) > 0:
            if utool.VERBOSE:
                print('[ibs] adding %d / %d {child}' % (len(dirty_{parent}_rowid_list), len({parent}_rowid_list)))

            # params_iter = preproc_{child}.add_{child}_params_gen(ibs, dirty_{parent}_rowid_list)
            colnames = {child_colnames}
            get_rowid_from_superkey = functools.partial(ibs.get_{parent}_{child}_rowids, ensure=False)
            params_iter = None
            {child}_rowid_list = ibs.dbcache.add_cleanly({TABLE}, colnames, params_iter, get_rowid_from_superkey)
        return {child}_rowid_list
    '''
)

adder_template_relationship = ut.codeblock(
    '''
    @adder
    def add_image_relationship(ibs, gid_list, eid_list):
        """ Adds a relationship between an image and and encounter """
        colnames = ('image_rowid', 'encounter_rowid',)
        params_iter = list(zip(gid_list, eid_list))
        get_rowid_from_superkey = ibs.get_egr_rowid_from_superkey
        superkey_paramx = (0, 1)
        egrid_list = ibs.db.add_cleanly(EG_RELATION_TABLE, colnames, params_iter,
                                        get_rowid_from_superkey, superkey_paramx)
        return egrid_list
    ''')

# Getters

getter_template_dependant_column = ut.codeblock(
    '''
    def get_{parent}_{col}({self}, {parent}_rowid_list, config_rowid=None):
        """ get {col} data of the {parent} table using the dependant {child} table

        Args:
            {parent}_rowid_list (list):

        Returns:
            list: {col}_list
        """
        {child}_rowid_list = {self}.get_{parent}_{child}_rowids({parent}_rowid_list)
        {col}_list = {self}.get_{child}_{col}({child}_rowid_list, config_rowid=config_rowid)
        return {col}_list
    ''')


getter_template_native_column = ut.codeblock(
    '''
    def get_{tbl}_{col}({self}, {tbl}_rowid_list):
        """gets data from the level 0 column "{col}" in the "{tbl}" table

        Args:
            {tbl}_rowid_list (list):

        Returns:
            list: {col}_list
        """
        params_iter = (({tbl}_rowid,) for {tbl}_rowid in {tbl}_rowid_list)
        colnames = ({COLNAME},)
        {col}_list = {self}.dbcache.get({TABLE}, colnames, params_iter)
        return {col}_list
    ''')


getter_template_native_rowid_from_superkey = ut.codeblock(
    '''
    def get_{tbl}_rowid_from_superkey({self}, {superkey_args},
                                      eager=False, num_params=None):
        colnames = ({tbl}_rowid),
        params_iter = zip({superkey_args})
        andwhere_colnames = [{superkey_args}]
        {tbl}_rowid_list = {self}.{dbself}.get_where2(
            {TABLE}, colnames, params_iter, andwhere_colnames, eager=eager,
            num_params=num_params)
        return {tbl}_rowid_list
    ''')


getter_template_native_multicolumn = ut.codeblock(
    '''
    def get_{tbl}_{multicolname}({self}, {tbl}_rowid_list):
        pass
    ''')


# Setters

setter_template_native_column = ut.codeblock(
    '''
    def set_{tbl}_{colname}({self}, {tbl}_rowid_list, val_list):
        pass
    ''')

setter_template_native_multicolumn = ut.codeblock(
    '''
    def set_{tbl}_{multicolname}({self}, {tbl}_rowid_list, vals_list):
        pass
    ''')

# Deleters

deleter_template_native_tbl = ut.codeblock(
    '''
    @deleter
    @cache_invalidator({TABLE})
    def delete_annots({self}, {tbl}_rowid_list):
        """ deletes annotations from the database """
        if utool.VERBOSE:
            print('[{self}] deleting %d {tbl} rows' % len({tbl}_rowid_list))
        # Delete dependant properties
        {self}.delete_{tbl}_chips({tbl}_rowid_list)
        {self}.{dbself}.delete_rowids({TABLE}, {tbl}_rowid_list)
        {self}.delete_{tbl}_relations({tbl}_rowid_list)
    '''
)

'''
s/ibs/{self}/gc
s/db/{dbself}/gc
'''


def autogenerate_controller_methods():
    pass

#if __name__ == '__main__':
#ibs = None
if __name__ == '__main__':
    """
    python ibeis/control/templates.py
    """
    if 'ibs' not in vars():
        import ibeis
        ibs = ibeis.opendb('testdb1')
    locals_ = main(ibs)
    exec(ut.execstr_dict(locals_))
