# developer convenience functions for ibs
from __future__ import absolute_import, division, print_function
import utool
from itertools import izip

# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[duct_tape]', DEBUG=False)


def fix_compname_configs(ibs):
    """ duct tape to keep version in check """
    from ibeis import constants
    #ibs.MANUAL_CONFIG_SUFFIX = '_MANUAL_'  #+ utool.get_computer_name()
    #ibs.MANUAL_CONFIGID = ibs.add_config(ibs.MANUAL_CONFIG_SUFFIX)
    # We need to fix the manual config suffix to not use computer names anymore

    configid_list = ibs.get_valid_configids()
    cfgsuffix_list = ibs.get_config_suffixes(configid_list)

    ibs.MANUAL_CONFIG_SUFFIX = 'MANUAL_CONFIG'
    ibs.MANUAL_CONFIGID = ibs.add_config(ibs.MANUAL_CONFIG_SUFFIX)

    for rowid, suffix in filter(lambda tup:
                                tup[1].startswith('_MANUAL_'),
                                izip(configid_list, cfgsuffix_list)):
        # Fix the tables with bad config_rowids
        ibs.db.executeone(
            '''
            UPDATE {AL_RELATION_TABLE}
            SET config_rowid=?
            WHERE config_rowid=?
            '''.format(**constants.__dict__),
            params=(ibs.MANUAL_CONFIGID, rowid))

        # Delete the bad config_suffixes
        ibs.db.executeone(
            '''
            DELETE
            FROM {CONFIG_TABLE}
            WHERE config_rowid=?
            '''.format(**constants.__dict__),
            params=(rowid))
