# -*- coding: utf-8 -*-
"""
CommandLine:

    # Run many experiments
    python -m ibeis.scripts.gen_cand_expts --exec-generate_all
    ./overnight_experiments.sh

    # Database information
    python -m ibeis --db PZ_MTEST --dbinfo --postload-exit
    python -m ibeis --db PZ_Master0 --dbinfo --postload-exit

    # Info about configs for a test
    python -m ibeis.experiments.experiment_harness --exec-run_test_configurations2 -t candidacy -a controlled --db PZ_MTEST --acfginfo
    python -m ibeis.experiments.experiment_harness --exec-run_test_configurations2 -t candidacy:sample_size=None -a controlled --db PZ_Master0 --acfginfo
    python -m ibeis.experiments.experiment_harness --exec-run_test_configurations2 -t candidacy -a controlled --db NNP_Master3 --acfginfo
"""
from __future__ import absolute_import, division, print_function
import sys
from os.path import expanduser
import utool as ut


def generate_dbinfo_table():
    """
    python -m ibeis.other.dbinfo --test-latex_dbstats --dblist PZ_Master0 PZ_FlankHack PZ_MTEST NNP_Master3 GZ_ALL NNP_MasterGIRM_core --show

    FIXME: Old database should not be converted to left
    python -m ibeis.other.dbinfo --test-latex_dbstats --dblist PZ_Master0 PZ_FlankHack PZ_MTEST NNP_Master3 GZ_ALL NNP_MasterGIRM_core --show
    python -m ibeis.other.dbinfo --test-latex_dbstats --dblist PZ_MTEST --show
    python -m ibeis.other.dbinfo --test-latex_dbstats --dblist GZ_Master0 --show
    python -m ibeis.other.dbinfo --test-latex_dbstats --dblist GIR_Tanya --show
    python -m ibeis.other.dbinfo --test-latex_dbstats --dblist LF_WEST_POINT_OPTIMIZADAS LF_OPTIMIZADAS_NI_V_E LF_Bajo_bonito --show
    python -m ibeis.other.dbinfo --test-latex_dbstats --dblist JAG_Kieryn JAG_Kelly --show
    """
    pass


TEST_GEN_FUNCS = []


def register_testgen(func):
    global TEST_GEN_FUNCS
    TEST_GEN_FUNCS.append(func)
    return func


if ut.get_argflag('--full'):
    ACFG_NAME_CAND_OPTIONS = ['controlled', 'controlled2']
else:
    ACFG_NAME_CAND_OPTIONS = ['controlled']


#@register_testgen
def precompute_data():
    """
    Generates the experiments we are doing on invariance

    CommandLine:
        python -m ibeis.scripts.gen_cand_expts --exec-precompute_data

    Example:
        >>> from ibeis.scripts.gen_cand_expts import *
        >>> precompute_data()
    """
    #basecmd = 'python -m ibeis.experiments.experiment_printres --exec-print_latexsum --rank-lt-list=1,5,10,100 '
    varydict = ut.odict([
        ('preload_flags', [
            #'--preload-chip',
            #'--preload-feat',
            #'--preload-feeatweight',
            '--preload',
            '--preindex',
        ]),
        ('dbname', get_dbnames()),
        ('acfg_name', ['default:qaids=allgt,species=primary,viewpoint_base=primary,is_known=True']),
        ('cfg_name', ['default', 'candidacy', 'candidacy_invariance']),
    ])
    return make_standard_test_scripts(varydict, 'preload', 'preload')


def generate_all():
    r"""
    CommandLine:
        python -m ibeis.scripts.gen_cand_expts --exec-generate_all --vim
        python -m ibeis.scripts.gen_cand_expts --exec-generate_all
        ./overnight_experiments.sh

    Example:
        >>> from ibeis.scripts.gen_cand_expts import *  # NOQA
        >>> generate_all()
    """
    #script_names = ['sh ' + func()[0] for func in TEST_GEN_FUNCS]
    script_lines = ut.flatten([['\n\n### ' + ut.get_funcname(func)] + func()[2] for func in TEST_GEN_FUNCS])
    fpath, script, line_list = write_script_lines(script_lines, 'overnight_experiments.sh')
    if ut.get_argflag('--vim'):
        ut.editfile(fpath)
    return fpath, script, line_list


@register_testgen
def baseline_experiments():
    """
    Generates the experiments we are doing on invariance

    CommandLine:
        python -m ibeis.scripts.gen_cand_expts --exec-baseline_experiments
        ./experiment_baseline.sh

    Example:
        >>> from ibeis.scripts.gen_cand_expts import *
        >>> baseline_experiments()
    """
    # Invariance Experiments
    varydict = ut.odict([
        #('acfg_name', ['controlled']),
        #('acfg_name', ['controlled', 'controlled2']),
        ('acfg_name', ACFG_NAME_CAND_OPTIONS),
        ('cfg_name', ['candidacy']),
        ('dbname', get_dbnames()),
    ])
    return make_standard_test_scripts(varydict, 'baseline', 'cumhist')


@register_testgen
def invariance_experiments():
    """
    Generates the experiments we are doing on invariance

    CommandLine:
        python -m ibeis.scripts.gen_cand_expts --exec-invariance_experiments

    Example:
        >>> from ibeis.scripts.gen_cand_expts import *
        >>> invariance_experiments()
    """
    # Invariance Experiments
    #static_flags += ' --dpi=512 --figsize=11,4 --clipwhite'
    varydict = ut.odict([
        #('acfg_name', ['controlled']),
        #('acfg_name', ['controlled', 'controlled2']),
        ('acfg_name', ACFG_NAME_CAND_OPTIONS),
        ('cfg_name', ['candidacy_invariance']),
        ('dbname', get_dbnames()),
    ])
    return make_standard_test_scripts(varydict, 'invar', 'cumhist')


@register_testgen
def namescore_experiments():
    """
    Generates the experiments we are doing on invariance

    CommandLine:
        python -m ibeis.scripts.gen_cand_expts --exec-namescore_experiments

    Example:
        >>> from ibeis.scripts.gen_cand_expts import *
        >>> namescore_experiments()
    """
    varydict = ut.odict([
        #('acfg_name', ['controlled', 'controlled2']),
        ('acfg_name', ['controlled', 'varypername']),
        ('cfg_name', ['candidacy_namescore']),
        ('dbname', get_dbnames()),
    ])
    return make_standard_test_scripts(varydict, 'namescore', 'cumhist')


@register_testgen
def k_experiments():
    """
    CommandLine:
        python -m ibeis.scripts.gen_cand_expts --exec-k_experiments

    Example:
        >>> from ibeis.scripts.gen_cand_expts import *
        >>> k_experiments()
    """
    varydict = ut.odict([
        ('acfg_name', ['varysize']),
        ('cfg_name', ['candidacy_k']),
        ('dbname', get_dbnames()),
    ])
    return make_standard_test_scripts(varydict, 'k', 'surface')


@register_testgen
def viewpoint_experiments():
    """
    Generates the experiments we are doing on invariance

    CommandLine:
        python -m ibeis.scripts.gen_cand_expts --exec-viewpoint_experiments

    Example:
        >>> from ibeis.scripts.gen_cand_expts import *
        >>> viewpoint_experiments()
    """
    #basecmd = 'python -m ibeis.experiments.experiment_printres --exec-print_latexsum --rank-lt-list=1,5,10,100 '
    varydict = ut.odict([
        ('dbname', ['NNP_Master3', 'PZ_Master0']),
        ('acfg_name', ['viewpoint_compare']),
        ('cfg_name', ['default']),
    ])
    return make_standard_test_scripts(varydict, 'view', 'cumhist')


# -----------------
# Helpers


def get_results_command(expt_name, media_name):
    """
    Displays results using various media
    """
    plot_fname = expt_name + '_' + media_name + '_{{db}}_a_{{a}}_t_{{t}}'
    static_flags = ''
    dynamic_flags_ = ''
    if media_name == 'table':
        margs = 'ibeis.experiments.experiment_printres --exec-print_latexsum'
        static_flags = '--rank-lt-list=1,5,10,100'
    elif media_name == 'cumhist':
        margs = 'ibeis.experiments.experiment_drawing --exec-draw_rank_cdf'
        static_flags =  ' --save ' + plot_fname + '.png'
        static_flags += ' --dpath=~/code/ibeis/results  --adjust=.15 --dpi=256 --clipwhite'
    elif media_name == 'surface':
        margs = 'ibeis.experiments.experiment_drawing --exec-draw_rank_surface'
        static_flags =  ' --save ' + plot_fname + '.png'
        static_flags += ' --dpath=~/code/ibeis/results'
        static_flags += ' --clipwhite'
        static_flags += ' --dpi=256'
        static_flags += ' --figsize=12,4'
        static_flags += ' --adjust=.02,.02,.4,.02'
    elif media_name == 'preload':
        margs = 'ibeis.experiments.precomputer --exec-precfg'
        dynamic_flags_ = '{preload_flags}'
    else:
        raise NotImplementedError('media_name=%r' % (media_name,))
    basecmd = 'python -m ' + margs
    return basecmd, static_flags, dynamic_flags_
    #shortname = 'Expt' + media_name[0].upper() + media_name[1:]
    #shortscript = shortname + '.sh'
    #ut.write_modscript_alias(shortscript, margs)
    #return 'sh ' + shortscript


def get_dbnames(exclude_list=[]):
    from ibeis.experiments import experiment_configs
    dbnames = experiment_configs.get_candidacy_dbnames()
    dbnames = ut.setdiff_ordered(dbnames, exclude_list)
    return dbnames


def make_standard_test_scripts(varydict, expt_name, media_name):
    basecmd, static_flags, dynamic_flags_ = get_results_command(expt_name, media_name)
    dynamic_flags = '-t {cfg_name} -a {acfg_name} --db {dbname} ' + dynamic_flags_
    cmd_fmtstr = basecmd +  ' ' + dynamic_flags + ' ' + static_flags
    return write_formatted_script_lines(cmd_fmtstr, [varydict], 'experiment_' + expt_name + '.sh')


def write_formatted_script_lines(cmd_fmtstr, varydict_list, fpath):
    cfgdicts_list = list(map(ut.all_dict_combinations, varydict_list))
    line_list = []
    for cfgdicts in cfgdicts_list:
        line_list.append('')
        line_list.extend([cmd_fmtstr.format(**kw) for kw in cfgdicts])
    return write_script_lines(line_list, fpath)


def write_script_lines(line_list, fpath):
    exename = 'python'
    regen_cmd = (exename + ' ' + ' '.join(sys.argv)).replace(expanduser('~'), '~')
    script_lines = []
    script_lines.append('#!/bin/sh')
    script_lines.append('echo << \'EOF\' > /dev/null')
    script_lines.append('RegenCommand:')
    script_lines.append('   ' + regen_cmd)
    script_lines.append('dont forget to tmuxnew')
    script_lines.append('EOF')
    script_lines.extend(line_list)
    script = '\n'.join(script_lines)
    print(script)
    ut.writeto(fpath, script)
    ut.chmod_add_executable(fpath)
    return fpath, script, line_list


def gen_dbranks_tables():
    r"""
    CommandLine:
        python -m ibeis.scripts.gen_cand_expts --exec-gen_dbranks_tables

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.scripts.gen_cand_expts import *  # NOQA
        >>> result = gen_dbranks_tables()
        >>> print(result)
    """
    tex_file = ut.codeblock(  # NOQA
        r'''
        \begin{comment}
        python -c "import utool as ut; ut.write_modscript_alias('ExptPrint.sh', 'ibeis.experiments.experiment_printres --exec-print_latexsum')"
        python -c "import utool as ut; ut.write_modscript_alias('DrawRanks.sh', 'python -m ibeis.experiments.experiment_drawing --exec-draw_rank_cdf')"
        \end{comment}
        ''')

    #gen_table_line =
    #sh ExptPrint.sh -t candidacy --allgt --species=primary --db GZ_ALL --rank-lt-list=1,5,10,100
    pass


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.scripts.gen_cand_expts
        python -m ibeis.scripts.gen_cand_expts --allexamples
        python -m ibeis.scripts.gen_cand_expts --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
