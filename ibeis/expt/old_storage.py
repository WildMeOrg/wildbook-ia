# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import six
#import six
import utool
import utool as ut
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[expt_harn]')


@six.add_metaclass(ut.ReloadingMetaclass)
class ResultMetadata(object):
    def __init__(metadata, fpath, autoconnect=False):
        """
        metadata_fpath = join(figdir, 'result_metadata.shelf')
        metadata = ResultMetadata(metadata_fpath)
        """
        metadata.fpath = fpath
        metadata.dimensions = ['query_cfgstr', 'qaid']
        metadata._shelf = None
        metadata.dictstore = None
        if autoconnect:
            metadata.connect()

    def connect(metadata):
        import shelve
        metadata._shelf = shelve.open(metadata.fpath)
        if 'dictstore' not in metadata._shelf:
            dictstore = ut.AutoVivification()
            metadata._shelf['dictstore'] = dictstore
        metadata.dictstore = metadata._shelf['dictstore']

    def clear(metadata):
        dictstore = ut.AutoVivification()
        metadata._shelf['dictstore'] = dictstore
        metadata.dictstore = metadata._shelf['dictstore']

    def __del__(metadata):
        metadata.close()

    def close(metadata):
        metadata._shelf.close()

    def write(metadata):
        metadata._shelf['dictstore'] = metadata.dictstore
        metadata._shelf.sync()

    def set_global_data(metadata, cfgstr, qaid, key, val):
        metadata.dictstore[cfgstr][qaid][key] = val

    def sync_test_results(metadata, test_result):
        """ store all test results in the shelf """
        for cfgx in range(len(test_result.cfgx2_qreq_)):
            cfgstr = test_result.get_cfgstr(cfgx)
            qaids = test_result.qaids
            cfgresinfo = test_result.cfgx2_cfgresinfo[cfgx]
            for key, val_list in six.iteritems(cfgresinfo):
                for qaid, val in zip(qaids, val_list):
                    metadata.set_global_data(cfgstr, qaid, key, val)
        metadata.write()

    def get_cfgstr_list(metadata):
        cfgstr_list = list(metadata.dictstore.keys())
        return cfgstr_list

    def get_column_keys(metadata):
        unflat_colname_list = [
            [cols.keys() for cols in qaid2_cols.values()]
            for qaid2_cols in six.itervalues(metadata.dictstore)
        ]
        colname_list = ut.unique_keep_order2(ut.flatten(ut.flatten(unflat_colname_list)))
        return colname_list

    def get_square_data(metadata, cfgstr=None):
        # can only support one config at a time right now
        if cfgstr is None:
            cfgstr = metadata.get_cfgstr_list()[0]
        qaid2_cols = metadata.dictstore[cfgstr]
        qaids = list(qaid2_cols.keys())
        col_name_list = ut.unique_keep_order2(ut.flatten([cols.keys() for cols in qaid2_cols.values()]))
        #col_name_list = ['qx2_scoreexpdiff', 'qx2_gt_aid']
        #colname2_colvals = [None for colname in col_name_list]
        column_list = [
            [colvals.get(colname, None) for qaid, colvals in six.iteritems(qaid2_cols)]
            for colname in col_name_list]
        col_name_list = ['qaids'] + col_name_list
        column_list = [qaids] + column_list
        print('depth_profile(column_list) = %r' % (ut.depth_profile(column_list),))
        return col_name_list, column_list
