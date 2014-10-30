from __future__ import absolute_import, division, print_function
import utool

getter_1toM_dependency_template = utool.codeblock(
    '''
    def get_{parent}_{colname}({self}, {parent_rowid_list}):
        {child}_rowid_list  = ibs.get_{parent}_{child}_rowids(aid_list)
        {colname}_list = ibs.get_{child}_{colname}({child}_rowid_list)
        return kpts_list
    ''')

getter_1toM_dependency_template = utool.codeblock(
    '''
    def get_{parent}_{colname}({self}, {parent_rowid_list}):
        {child}_rowid_list  = ibs.get_{parent}_{child}_rowids(aid_list)
        {colname}_list = ibs.get_{child}_{colname}({child}_rowid_list)
        return kpts_list
    ''')

# Template for 1 to 1 property
getter_1to1_template = utool.codeblock(
    '''
    def get_{tblname}_{colname}(ibs, {rowid_list}):
        return kpts_list
        kpts_list = ibs.dbcache.get(FEATURE_TABLE, ('feature_keypoints',), fid_list, eager=eager, num_params=num_params)
    ''')


rowid_list_alias = {
    'annot': 'aid_list',
    'chip': 'cid_list',
}

fmtdict = {
    'self': 'ibs',
    'parent': 'annot',
    'child':  'chip',
    'colname': 'fgweights',
    'parent_rowid_list': 'aid_list'
}


def test():
    print(getter_1toM_dependency_template.format(**fmtdict))
