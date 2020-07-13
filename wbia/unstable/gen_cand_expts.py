# -*- coding: utf-8 -*-
"""
CommandLine:

    # Run many experiments
    python -m wbia.scripts.gen_cand_expts --exec-generate_all --full
    ./experiments_overnight.sh

    # Database information
    python -m wbia --db PZ_MTEST --dbinfo --postload-exit
    python -m wbia --db PZ_Master0 --dbinfo --postload-exit

    # Info about configs for a test
    python -m wbia --tf run_expt -t default -a ctrl --db PZ_MTEST --acfginfo
    python -m wbia --tf run_expt -t default:sample_size=None -a ctrl --db PZ_Master0 --acfginfo  # NOQA
    python -m wbia --tf run_expt -t default -a ctrl --db NNP_Master3 --acfginfo

    # Regen Figures
    python -m wbia.scripts.gen_cand_expts --exec-parse_latex_comments_for_commmands
    ./regen_figdef_expt.sh


    # Print all annotation configs that will be used
    python -m wbia.scripts.gen_cand_expts --exec-inspect_annotation_configs --full
    sh experiment_inspect_acfg.sh

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from os.path import expanduser
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


TEST_GEN_FUNCS = []


# def pick_sample_size(popsize, confidence=.99, expected_std=.5,
# margin_of_error=.05, is_finite=True):
#    """
#    Determine a statistically significant sample size

#    References:
#        https://en.wikipedia.org/wiki/Sample_size_determination
#        http://www.surveysystem.com/sample-size-formula.htm
#        http://courses.wcupa.edu/rbove/Berenson/10th%20ed%20CD-ROM%20topics/section8_7.pdf
#    """
#    #import scipy.stats as spstats
#    zscore = {.99: 2.678, .95: 1.96, .9: 1.645}[confidence]
#    ((zscore ** 2) * (expected_std) * (1 - expected_std)) / (margin_of_error)
#    #spstats.norm.ppf([.001])
#    #samplesize =


def register_testgen(func):
    global TEST_GEN_FUNCS
    TEST_GEN_FUNCS.append(func)
    return func


ACFG_OPTION_CONTROLLED = ['controlled']
ACFG_OPTION_VARYSIZE = ['varysize_pzm']
ACFG_OPTION_VARYPERNAME = ['varypername:qsize=500']


def generate_all():
    r"""
    CommandLine:
        python -m wbia.scripts.gen_cand_expts --exec-generate_all --vim
        python -m wbia.scripts.gen_cand_expts --exec-generate_all
        python -m wbia.scripts.gen_cand_expts --exec-generate_all --full
        ./experiments_overnight.sh

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.gen_cand_expts import *  # NOQA
        >>> generate_all()
    """
    # script_names = ['sh ' + func()[0] for func in TEST_GEN_FUNCS]
    script_lines = ut.flatten(
        [
            [
                '\n\n### ' + ut.get_funcname(func),
                '# python -m wbia.scripts.gen_cand_expts --exec-' + ut.get_funcname(func),
            ]
            + make_standard_test_scripts(func())[2]
            for func in TEST_GEN_FUNCS
        ]
    )
    fname, script, line_list = write_script_lines(
        script_lines, 'experiments_overnight.sh'
    )
    if ut.get_argflag('--vim'):
        ut.editfile(fname)
    return fname, script, line_list


def database_intersection_test():
    """
    # PZ_FlankHack is a pure subset of PZ_Master0, but there are minor changes between them
    python -m wbia.dbio.export_subset --exec-check_database_overlap --db1=PZ_FlankHack --db2=PZ_Master0  # NOQA

    # PZ_MTEST is also a subset of PZ_Master0 with minor changes
    python -m wbia.dbio.export_subset --exec-check_database_overlap --db1=PZ_MTEST --db2=PZ_Master0  # NOQA

    # NNP_Master3 and PZ_Master0 are disjoint
    python -m wbia.dbio.export_subset --exec-check_database_overlap --db1=NNP_Master3 --db2=PZ_Master0  # NOQA

    python -m wbia.dbio.export_subset --exec-check_database_overlap --db1=PZ_Master1 --db2=PZ_Master0  # NOQA
    """
    pass


def generate_dbinfo_table():
    """
    python -m wbia.other.dbinfo --test-latex_dbstats --dblist PZ_Master0 PZ_FlankHack PZ_MTEST NNP_Master3 GZ_ALL NNP_MasterGIRM_core --show  # NOQA

    FIXME: Old database should not be converted to left
    python -m wbia.other.dbinfo --test-latex_dbstats --dblist PZ_Master1 --show
    python -m wbia.other.dbinfo --test-latex_dbstats --dblist PZ_Master0 PZ_FlankHack PZ_MTEST NNP_Master3 GZ_ALL NNP_MasterGIRM_core --show  # NOQA
    python -m wbia.other.dbinfo --test-latex_dbstats --dblist PZ_MTEST --show
    python -m wbia.other.dbinfo --test-latex_dbstats --dblist GZ_Master0 --show
    python -m wbia.other.dbinfo --test-latex_dbstats --dblist GIR_Tanya --show
    python -m wbia.other.dbinfo --test-latex_dbstats --dblist LF_WEST_POINT_OPTIMIZADAS LF_OPTIMIZADAS_NI_V_E LF_Bajo_bonito --show  # NOQA
    python -m wbia.other.dbinfo --test-latex_dbstats --dblist JAG_Kieryn JAG_Kelly --show
    """
    pass


def parse_latex_comments_for_commmands():
    r"""
    CommandLine:
        python -m wbia.scripts.gen_cand_expts --exec-parse_latex_comments_for_commmands

    Example:
        >>> # DISABLE_DOCTEST
        >>> # SCRIPT
        >>> from wbia.scripts.gen_cand_expts import *  # NOQA
        >>> parse_latex_comments_for_commmands()
    """
    fname = ut.get_argval('--fname', type_=str, default='figdefexpt.tex')
    text = ut.read_from(ut.truepath('~/latex/crall-candidacy-2015/' + fname))
    # text = ut.read_from(ut.truepath('~/latex/crall-candidacy-2015/figdefindiv.tex'))
    lines = text.split('\n')
    cmd_list = ['']
    in_comment = True
    for line in lines:
        if line.startswith('% ---'):
            # Keep separators
            toadd = line.replace('%', '#')
            if not (len(cmd_list) > 1 and cmd_list[-1].startswith('# ---')):
                cmd_list[-1] += toadd
            else:
                cmd_list.append(toadd)
            cmd_list.append('')

        if line.strip().startswith(r'\begin{comment}'):
            in_comment = True
            continue
        if in_comment:
            line = line.strip()
            if line == '' or line.startswith('#') or line.startswith('%'):
                in_comment = False
            else:
                cmd_list[-1] = cmd_list[-1] + line
                if not line.strip().endswith('\\'):
                    cmd_list[-1] = cmd_list[-1] + ' $@'
                    # cmd_list.append('')
                    # cmd_list.append('#--')
                    cmd_list.append('')
                    in_comment = False
                else:
                    cmd_list[-1] = cmd_list[-1] + '\n'

    cmd_list = [cmd.replace('--render', '').replace('--diskshow', '') for cmd in cmd_list]

    # formatting
    cmd_list2 = []
    for cmd in cmd_list:
        # cmd = cmd.replace(' -t ', ' \\\n    -t ')
        # cmd = cmd.replace('--db', '\\\n    --db')
        # cmd = cmd.replace('python -m wbia.dev', './dev.py')
        cmd = cmd.replace('python -m wbia.dev -e', 'wbia -e')
        cmd_list2.append(cmd)
    cmd_list = cmd_list2

    print('cmd_list = %s' % (ut.repr2(cmd_list),))
    from os.path import splitext

    script_fname = 'regen_' + splitext(fname)[0] + '.sh'
    fname, script, line_list = write_script_lines(cmd_list, script_fname)
    # ut.chmod_add_executable(fname)


def inspect_annotation_configs():
    r"""
    CommandLine:
        python -m wbia.scripts.gen_cand_expts --exec-inspect_annotation_configs
        python -m wbia.scripts.gen_cand_expts --exec-inspect_annotation_configs --full

    Example:
        >>> # SCRIPT
        >>> from wbia.scripts.gen_cand_expts import *  # NOQA
        >>> make_standard_test_scripts(inspect_annotation_configs())
    """
    testdef_list = [func() for func in TEST_GEN_FUNCS]
    acfg_name_list = ut.flatten([tup[0]['acfg_name'] for tup in testdef_list])
    acfg_name_list = list(set(acfg_name_list))
    varydict = ut.odict(
        [
            # ('acfg_name', ['controlled']),
            # ('acfg_name', ['controlled', 'controlled2']),
            ('acfg_name', [' '.join(acfg_name_list)]),
            ('dbname', get_dbnames()),
        ]
    )
    return varydict, 'inspect_acfg', 'inspect_acfg'


# @register_testgen
def precompute_data():
    """
    Ensure features and such are computed
    CommandLine:
        python -m wbia.scripts.gen_cand_expts --exec-precompute_data

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.gen_cand_expts import *
        >>> make_standard_test_scripts(precompute_data())
    """
    # basecmd = 'python -m wbia.expt.experiment_printres
    # --exec-print_latexsum --rank-lt-list=1,5,10,100 '
    varydict = ut.odict(
        [
            (
                'preload_flags',
                [
                    # '--preload-chip',
                    # '--preload-feat',
                    # '--preload-feeatweight',
                    '--preload',
                    '--preindex',
                ],
            ),
            ('dbname', get_dbnames()),
            (
                'acfg_name',
                ['default:qaids=allgt,species=primary,view=primary,is_known=True'],
            ),
            ('cfg_name', ['default', 'candidacy_baseline', 'candidacy_invariance']),
        ]
    )
    return (varydict, 'preload', 'preload')


# --- TEST DEFINITIONS


@register_testgen
def experiments_baseline():
    """
    Generates the experiments we are doing on invariance

    CommandLine:
        python -m wbia.scripts.gen_cand_expts --exec-experiments_baseline
        ./experiment_baseline.sh

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.gen_cand_expts import *
        >>> make_standard_test_scripts(experiments_baseline())
    """
    # Invariance Experiments
    varydict = ut.odict(
        [
            # ('acfg_name', ['controlled']),
            # ('acfg_name', ['controlled', 'controlled2']),
            ('acfg_name', ACFG_OPTION_CONTROLLED),
            ('cfg_name', ['candidacy_baseline']),
            ('dbname', get_dbnames()),
        ]
    )
    return (varydict, 'baseline', 'cumhist')


@register_testgen
def experiments_invariance():
    """
    Generates the experiments we are doing on invariance

    CommandLine:
        python -m wbia.scripts.gen_cand_expts --exec-experiments_invariance
        python -m wbia.scripts.gen_cand_expts --exec-experiments_invariance --full

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.gen_cand_expts import *
        >>> make_standard_test_scripts(experiments_invariance())
    """
    # Invariance Experiments
    # static_flags += ' --dpi=512 --figsize=11,4 --clipwhite'
    varydict = ut.odict(
        [
            # ('acfg_name', ['controlled']),
            # ('acfg_name', ['controlled', 'controlled2']),
            ('acfg_name', ACFG_OPTION_CONTROLLED),
            ('cfg_name', ['candidacy_invariance']),
            ('dbname', get_dbnames()),
        ]
    )
    return (varydict, 'invar', 'cumhist')


@register_testgen
def experiments_namescore():
    """
    Generates the experiments we are doing on invariance

    CommandLine:
        python -m wbia.scripts.gen_cand_expts --exec-experiments_namescore --full
        ./experiment_namescore.sh

        python -m wbia.scripts.gen_cand_expts --exec-experiments_namescore --full
        python -m wbia.expt.experiment_helpers --exec-get_annotcfg_list:0 -a candidacy_namescore --db PZ_Master1  # NOQA

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.gen_cand_expts import *
        >>> make_standard_test_scripts(experiments_namescore())
    """
    varydict = ut.odict(
        [
            # ('acfg_name', ['controlled', 'controlled2']),
            ('acfg_name', ACFG_OPTION_CONTROLLED + ACFG_OPTION_VARYPERNAME),
            ('cfg_name', ['candidacy_namescore', 'candidacy_namescore:K=1']),
            ('dbname', get_dbnames()),
        ]
    )
    return (varydict, 'namescore', 'cumhist')


@register_testgen
def experiments_k():
    """
    CommandLine:
        python -m wbia.scripts.gen_cand_expts --exec-experiments_k
        python -m wbia.scripts.gen_cand_expts --exec-experiments_k --full
        ./experiment_k.sh

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.gen_cand_expts import *
        >>> make_standard_test_scripts(experiments_k())
    """
    varydict = ut.odict(
        [
            ('acfg_name', ACFG_OPTION_VARYSIZE),
            ('cfg_name', ['candidacy_k']),
            ('dbname', get_dbnames(exclude_list=['PZ_FlankHack', 'PZ_MTEST'])),
        ]
    )
    # return (varydict, 'k', ['surface3d', 'surface2d'])
    return (varydict, 'k', ['surface2d'])


@register_testgen
def experiments_viewpoint():
    """
    Generates the experiments we are doing on invariance

    CommandLine:
        python -m wbia.scripts.gen_cand_expts --exec-experiments_viewpoint --full
        ./experiment_view.sh

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.gen_cand_expts import *
        >>> make_standard_test_scripts(experiments_viewpoint())
    """
    # basecmd = 'python -m wbia.expt.experiment_printres
    # --exec-print_latexsum --rank-lt-list=1,5,10,100 '
    varydict = ut.odict(
        [
            ('acfg_name', ['viewpoint_compare']),
            ('cfg_name', ['default']),
            # ('dbname', ['NNP_Master3', 'PZ_Master0']),
            ('dbname', ['PZ_Master1']),
        ]
    )
    return (varydict, 'view', 'cumhist')


# -----------------
# Helpers


def get_results_command(expt_name, media_name):
    """
    Displays results using various media
    """
    dynamic_flags = ' -t {cfg_name} -a {acfg_name} --db {dbname}'
    plot_fname = 'figures/' + expt_name + '_' + media_name + '_{{db}}_a_{{a}}_t_{{t}}'
    output_flags = ''
    static_flags = ''
    # static_flags = ' --diskshow'
    dynamic_flags_ = ''
    dpath = '~/latex/crall-candidacy-2015/'
    if media_name == 'table':
        margs = 'wbia.dev -e print_latexsum'
        static_flags += '--rank-lt-list=1,5,10,100'
    elif media_name == 'cumhist':
        margs = 'wbia.dev -e draw_rank_cmc'
        output_flags += ' --save ' + plot_fname + '.png'
        output_flags += ' --dpath=' + dpath
        static_flags += ' --adjust=.05,.08,.0,.15 --dpi=256 --clipwhite'
    elif media_name == 'surface2d':
        margs = 'wbia.dev -e draw_rank_surface'
        output_flags += ' --save ' + plot_fname + '.png'
        output_flags += ' --dpath=' + dpath
        static_flags += ' --clipwhite'
        static_flags += ' --dpi=256'
        static_flags += ' --figsize=12,4'
        static_flags += ' --adjust=.1,.25,.2,.2'
    elif media_name == 'surface3d':
        margs = 'wbia.dev --e draw_rank_surface --no2dsurf'
        output_flags += ' --save ' + plot_fname + '.png'
        output_flags += ' --dpath=' + dpath
        static_flags += ' --clipwhite'
        static_flags += ' --dpi=256'
        static_flags += ' --figsize=12,4'
        static_flags += ' --adjust=.1,.1,.01,.01'
    elif media_name == 'preload':
        margs = 'wbia.expt.precomputer --exec-precfg'
        dynamic_flags_ = ' {preload_flags}'
    elif media_name == 'inspect_acfg':
        margs = 'wbia.expt.experiment_helpers --exec-get_annotcfg_list:0'
        dynamic_flags = '-a {acfg_name} --db {dbname} '
    else:
        raise NotImplementedError('media_name=%r' % (media_name,))
    static_flags += ' $@'
    dynamic_flags = dynamic_flags + dynamic_flags_
    basecmd = 'python -m ' + margs
    cmd_flaglist = [basecmd, dynamic_flags, output_flags, static_flags]
    return cmd_flaglist
    # shortname = 'Expt' + media_name[0].upper() + media_name[1:]
    # shortscript = shortname + '.sh'
    # ut.write_modscript_alias(shortscript, margs)
    # return 'sh ' + shortscript


def get_dbnames(exclude_list=[]):
    from wbia.expt import experiment_configs

    dbnames = experiment_configs.get_candidacy_dbnames()
    dbnames = ut.setdiff_ordered(dbnames, exclude_list)
    dbnames = ['PZ_Master1']
    return dbnames


def make_standard_test_scripts(*args):
    if len(args) == 1:
        varydict, expt_name, media_name = args[0]
    else:
        varydict, expt_name, media_name = args
    media_names = ut.ensure_iterable(media_name)
    cmd_fmtstr_list = []
    for media_name in media_names:
        cmd_flaglist = get_results_command(expt_name, media_name)
        cmd_fmtstr = ''.join(cmd_flaglist)
        cmd_fmtstr = ' \\\n   '.join(cmd_flaglist)
        cmd_fmtstr_list.append(cmd_fmtstr)
    fname = 'experiment_' + expt_name + '.sh'
    return write_formatted_script_lines(cmd_fmtstr_list, [varydict], fname)


def write_formatted_script_lines(cmd_fmtstr_list, varydict_list, fname):
    cfgdicts_list = list(map(ut.all_dict_combinations, varydict_list))
    line_list = []
    for cmd_fmtstr in cmd_fmtstr_list:
        line_list.append('')
        for cfgdicts in cfgdicts_list:
            line_list.append('')
            try:
                line_list.extend([cmd_fmtstr.format(**kw) for kw in cfgdicts])
            except Exception as ex:
                print(cmd_fmtstr)
                ut.printex(ex, keys=['cmd_fmtstr', 'kw'])
                raise
    return write_script_lines(line_list, fname)


def write_script_lines(line_list, fname):
    exename = 'python'
    regen_cmd = (exename + ' ' + ' '.join(sys.argv)).replace(expanduser('~'), '~')
    script_lines = []
    script_lines.append('#!/bin/sh')
    script_lines.append("echo << 'EOF' > /dev/null")
    script_lines.append('RegenCommand:')
    script_lines.append('   ' + regen_cmd)
    script_lines.append('CommandLine:')
    script_lines.append('   sh ' + fname)
    script_lines.append('dont forget to tmuxnew')
    script_lines.append('EOF')
    script_lines.extend(line_list)
    script = '\n'.join(script_lines)
    print(script)
    import wbia
    from os.path import dirname, join

    dpath = dirname(ut.get_module_dir(wbia))
    fpath = join(dpath, fname)
    if not ut.get_argflag('--dryrun'):
        ut.writeto(fpath, script)
        ut.chmod_add_executable(fpath)
    return fname, script, line_list


def gen_dbranks_tables():
    r"""
    CommandLine:
        python -m wbia.scripts.gen_cand_expts --exec-gen_dbranks_tables

    Example:
        >>> # SCRIPT
        >>> from wbia.scripts.gen_cand_expts import *  # NOQA
        >>> result = gen_dbranks_tables()
        >>> print(result)
    """
    tex_file = ut.codeblock(  # NOQA
        r"""
        \begin{comment}
        python -c "import utool as ut; ut.write_modscript_alias('ExptPrint.sh', 'wbia.expt.experiment_printres --exec-print_latexsum')"
        python -c "import utool as ut; ut.write_modscript_alias('DrawRanks.sh', 'python -m wbia.expt.experiment_drawing --exec-draw_rank_cmc')"
        \end{comment}
        """
    )

    # gen_table_line =
    # sh ExptPrint.sh -t candidacy_baseline --allgt --species=primary --db
    # GZ_ALL --rank-lt-list=1,5,10,100
    pass


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.scripts.gen_cand_expts
        python -m wbia.scripts.gen_cand_expts --allexamples
        python -m wbia.scripts.gen_cand_expts --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
