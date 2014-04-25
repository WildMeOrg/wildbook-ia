#!/usr/bin/env python
"""
This is a hacky script meant to be run interactively
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import ibeis
ibeis._preload()
from _devscript import devcmd,  DEVCMD_FUNCTIONS
from ibeis.dev.all_imports import *  # NOQA
from plottool import draw_func2 as df2
from ibeis.dev import main_helpers
from ibeis.viz import interact
from ibeis.dev import experiment_configs
from ibeis.dev import experiment_harness
import utool
import multiprocessing
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dev]', DEBUG=False)

if not 'back' in vars():
    back = None


@devcmd
def vdd(ibs=None, qrid_list=None):
    utool.view_directory(ibs.get_dbdir())


@devcmd('show')
def show_rids(ibs, qrid_list):
    for rid in qrid_list:
        interact.ishow_chip(ibs, rid, fnum=df2.next_fnum())


@devcmd()
def change_names(ibs, qrid_list):
    #new_name = utool.get_arg('--name', str, default='<name>_the_<species>')
    new_name = utool.get_arg('--name', str, default='glob')
    for rid in qrid_list:
        ibs.print_name_table()
        #(nid,) = ibs.add_names((new_name,))
        ibs.set_roi_properties((rid,), 'name', (new_name,))
        ibs.print_name_table()
        ibs.print_roi_table()
    new_nid = ibs.get_name_nids(new_name, ensure=False)
    if back is not None:
        back.select_nid(new_nid)


@devcmd('gravity', 'gv')
def compare_gravity(ibs, qrid_list):
    ibs_nogv = ibs.clone_handle()
    #ibslist = [ibs.clone_handle() for _ in xrange(1000)]
    ibs_nogv.update_cfg(nogravity_hack=True)
    for rid in qrid_list:
        interact.ishow_chip(ibs_nogv, rid, fnum=df2.next_fnum(), eig=True)
    for rid in qrid_list:
        interact.ishow_chip(ibs, rid, fnum=df2.next_fnum(), eig=True)


@devcmd('query')
def query_rids(ibs, qrid_list):
    qrid2_qres = ibs.query_database(qrid_list)
    for qrid in qrid_list:
        qres = qrid2_qres[qrid]
        interact.ishow_qres(ibs, qres, fnum=df2.next_fnum(), annote_mode=1)
    return qrid2_qres


@devcmd('sver')
def sver_rids(ibs, qrid_list):
    qrid2_qres = ibs.query_database(qrid_list)
    for qrid in qrid_list:
        qres = qrid2_qres[qrid]
        rid2 = qres.get_top_rids()[0]
        interact.ishow_sver(ibs, qrid, rid2, fnum=df2.next_fnum(), annote_mode=1)
    return qrid2_qres


@devcmd('cfg')
def printcfg(ibs, qrid_list):
    ibs.cfg.printme3()
    print(ibs.cfg.query_cfg.get_uid())


@devcmd('hsdbs')
def list_hsdbs(*args):
    from ibeis.injest.injest_my_hotspotter_dbs import get_unconverted_hsdbs
    get_unconverted_hsdbs()


@devcmd('convert')
def convert_hsdbs(*args):
    from ibeis.injest.injest_my_hotspotter_dbs import injest_unconverted_hsdbs_in_workdir
    injest_unconverted_hsdbs_in_workdir()


#@utool.indent_decor('[dev]')
@profile
def run_experiments(ibs, qrid_list):
    print('\n')
    print('==========================')
    print('RUN INVESTIGATIONS %s' % ibs.get_dbname())
    print('==========================')
    input_test_list = params.args.tests[:]
    print('input_test_list = %r' % (input_test_list,))
    # fnum = 1

    valid_test_list = []  # build list for printing in case of failure
    valid_test_helpstr_list = []  # for printing

    def intest(*args, **kwargs):
        helpstr = kwargs.get('help', '')
        valid_test_helpstr_list.append('   -t ' + ', '.join(args) + helpstr)
        for testname in args:
            valid_test_list.append(testname)
            ret = testname in input_test_list
            if ret:
                input_test_list.remove(testname)
                print('+===================')
                print('| running testname=%s' % testname)
                return ret
        return False

    valid_test_helpstr_list.append('    # --- Simple Tests ---')

    if intest('info'):
        print(ibs.get_infostr())
    if intest('printcfg'):
        printcfg(ibs)
    if intest('tables'):
        ibs.print_tables()
    if intest('imgtbl'):
        ibs.print_image_table()

    valid_test_helpstr_list.append('    # --- Decor Tests ---')

    # Run decorated functions
    for (func_aliases, func) in DEVCMD_FUNCTIONS:
        if intest(*func_aliases):
            func(ibs, qrid_list)

    valid_test_helpstr_list.append('    # --- Config Tests ---')

    # Allow any testcfg to be in tests like: vsone_1 or vsmany_3
    for test_cfg_name in experiment_configs.TEST_NAMES:
        if intest(test_cfg_name):
            experiment_harness.test_configurations(ibs, qrid_list, [test_cfg_name],  df2.next_fnum())

    valid_test_helpstr_list.append('    # --- Help ---')

    if intest('help'):
        print('valid tests are:')
        print('\n'.join(valid_test_helpstr_list))
        return

    if len(input_test_list) > 0:
        print('valid tests are: \n')
        print('\n'.join(valid_test_list))
        raise Exception('Unknown tests: %r ' % input_test_list)
    return locals()


@profile
def dev_main():
    global back
    print('++dev')
    main_locals = ibeis.main(gui='--gui' in sys.argv)
    ibs  = main_locals['ibs']
    back = main_locals['back']

    fnum = 1
    qrid_list = main_helpers.get_test_qrids(ibs)
    expt_locals = run_experiments(ibs, qrid_list)

    if not '--nopresent' in sys.argv:
        df2.present()
    ipy = (not '--gui' in sys.argv) or ('--cmd' in sys.argv)
    main_execstr = ibeis.main_loop(main_locals, ipy=ipy)
    return locals(), main_execstr

if __name__ == '__main__':
    """
        The Developer Script
            A command line interface to almost everything

            -w     # wait / show the gui / figures are visible
            --cmd  # ipython shell to play with variables
            -t     # run list of tests

            Examples:
                ./dev.py -t query -w
    """
    multiprocessing.freeze_support()  # for win32
    dev_locals, main_execstr = dev_main()
    dev_execstr = utool.execstr_dict(dev_locals, 'dev_locals')
    execstr = dev_execstr + '\n' + main_execstr
    exec(execstr)


"""
Snippets:

rid_list = ibs.get_valid_rids()
gid_list = ibs.get_valid_gids()
nid_list = ibs.get_valid_nids()

"""
