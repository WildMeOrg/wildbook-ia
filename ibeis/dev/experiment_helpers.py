from __future__ import absolute_import, division, print_function
import utool
import re
from itertools import imap
from . import experiment_configs
from ibeis.model import Config
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[expt_helpers]', DEBUG=False)

QUIET = utool.QUIET


def get_vary_dicts(test_cfg_name_list):
    vary_dicts = []
    for cfg_name in test_cfg_name_list:
        test_cfg = experiment_configs.__dict__[cfg_name]
        vary_dicts.append(test_cfg)
    if len(vary_dicts) == 0:
        valid_cfg_names = experiment_configs.TEST_NAMES
        raise Exception('Choose a valid testcfg:\n' + valid_cfg_names)
    return vary_dicts


def rankscore_str(thresh, nLess, total):
    #helper to print rank scores of configs
    percent = 100 * nLess / total
    fmtsf = '%' + str(utool.num2_sigfig(total)) + 'd'
    fmtstr = '#ranks < %d = ' + fmtsf + '/%d = (%.1f%%) (err=' + fmtsf + ')'
    rankscore_str = fmtstr % (thresh, nLess, total, percent, (total - nLess))
    return rankscore_str


def wrap_uid(uid):
    # REGEX to locate _XXXX(
    cfg_regex = r'_[A-Z][A-Z]*\('
    uidmarker_list = re.findall(cfg_regex, uid)
    uidconfig_list = re.split(cfg_regex, uid)
    args = [uidconfig_list, uidmarker_list]
    interleave_iter = utool.interleave(args)
    new_uid_list = []
    total_len = 0
    prefix_str = ''
    # If unbalanced there is a prefix before a marker
    if len(uidmarker_list) < len(uidconfig_list):
        frag = interleave_iter.next()
        new_uid_list += [frag]
        total_len = len(frag)
        prefix_str = ' ' * len(frag)
    # Iterate through markers and config strings
    while True:
        try:
            marker_str = interleave_iter.next()
            config_str = interleave_iter.next()
            frag = marker_str + config_str
        except StopIteration:
            break
        total_len += len(frag)
        new_uid_list += [frag]
        # Go to newline if past 80 chars
        if total_len > 80:
            total_len = 0
            new_uid_list += ['\n' + prefix_str]
    wrapped_uid = ''.join(new_uid_list)
    return wrapped_uid


def format_uid_list(uid_list):
    indented_list = utool.indent_list('    ', uid_list)
    wrapped_list = imap(wrap_uid, indented_list)
    return utool.joins('\n', wrapped_list)


#---------------
# Big Test Cache
#-----------


def load_cached_test_results(ibs, qreq, qrids, drids, nocache_testres, test_results_verbosity):
    pass
    #test_uid = qreq.get_query_uid(ibs, qrids)
    #cache_dir = join(ibs.dirs.cache_dir, 'experiment_harness_results')
    #io_kwargs = {'dpath': cache_dir,
                 #'fname': 'test_results',
                 #'uid': test_uid,
                 #'ext': '.cPkl'}

    #if test_results_verbosity == 2:
        #print('[harn] test_uid = %r' % test_uid)

    ## High level caching
    #if not params.args.nocache_query and (not nocache_testres):
        #qx2_bestranks = io.smart_load(**io_kwargs)
        #if qx2_bestranks is None:
            #print('[harn] qx2_bestranks cache returned None!')
        #elif len(qx2_bestranks) != len(qrids):
            #print('[harn] Re-Caching qx2_bestranks')
        #else:
            #return qx2_bestranks


def cache_test_results(qx2_bestranks, ibs, qreq, qrids, drids):
    pass
    #test_uid = qreq.get_query_uid(ibs, qrids)
    #cache_dir = join(ibs.dirs.cache_dir, 'experiment_harness_results')
    #utool.ensuredir(cache_dir)
    #io_kwargs = {'dpath': cache_dir,
                 #'fname': 'test_results',
                 #'uid': test_uid,
                 #'ext': '.cPkl'}
    #io.smart_save(qx2_bestranks, **io_kwargs)


def get_varied_params_list(test_cfg_name_list):
    vary_dicts = get_vary_dicts(test_cfg_name_list)
    get_all_dict_comb = utool.all_dict_combinations
    dict_comb_list = [get_all_dict_comb(_dict) for _dict in vary_dicts]
    varied_params_list = [comb for dict_comb in dict_comb_list for comb in dict_comb]
    #map(lambda x: print('\n' + str(x)), varied_params_list)
    return varied_params_list


def get_cfg_list(ibs, test_cfg_name_list):
    print('[harn] building cfg_list: %s' % test_cfg_name_list)
    if 'custom' == test_cfg_name_list:
        print('   * custom cfg_list')
        cfg_list = [ibs.prefs.query_cfg]
        return cfg_list
    varied_params_list = get_varied_params_list(test_cfg_name_list)
    # Add unique configs to the list
    cfg_list = []
    cfg_set = set([])
    for _dict in varied_params_list:
        cfg = Config.QueryConfig(**_dict)
        if not cfg in cfg_set:
            cfg_list.append(cfg)
            cfg_set.add(cfg)
    if not QUIET:
        print('[harn] return %d / %d unique configs' % (len(cfg_list), len(varied_params_list)))
    return cfg_list
