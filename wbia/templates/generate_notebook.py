# -*- coding: utf-8 -*-
r"""
CommandLine:
    # Generate and start an IPython notebook
    python -m wbia --tf autogen_ipynb --ipynb --db <dbname> [-a <acfg>] [-t <pcfg>]

    python -m wbia --tf autogen_ipynb --ipynb --db seaturtles -a default2:qhas_any=\(left,right\),sample_occur=True,occur_offset=[0,1,2],num_names=1

CommandLine:
    # to connect to a notebook on a remote machine that does not have the
    # appropriate port exposed you must start an SSH tunnel.
    # Typically a jupyter-notebook runs on port 8888.
    # Run this command on your local machine.
    ssh -N -f -L localhost:<local_port>:localhost:<remote_port> <remote_user>@<remote_host>

    E.G.
    ssh -N -f -L localhost:8889:localhost:8888 joncrall@hyrule.cs.rpi.edu
    # Now you can connect locally
    firefox localhost:8889


    # Running a server:
    jupyter-notebook password
    jupyter-notebook --no-browser --NotebookApp.iopub_data_rate_limit=100000000 --NotebookApp.token=


    # To allow remote jupyter-noteobok connections
    jupyter notebook --generate-config

    # Really need to do jupyter hub

    need to set
    c.NotebookApp.port = 8888
    c.NotebookApp.open_browser = False
    c.NotebookApp.ip = '*'


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
from wbia.templates import notebook_cells
from functools import partial


def autogen_ipynb(ibs, launch=None, run=None):
    r"""
    Autogenerates standard IBEIS Image Analysis IPython notebooks.

    CommandLine:
        python -m wbia autogen_ipynb --run --db lynx
        python -m wbia autogen_ipynb --run --db lynx

        python -m wbia autogen_ipynb --ipynb --db PZ_MTEST -p :proot=smk,num_words=64000 default
        python -m wbia autogen_ipynb --ipynb --db PZ_MTEST --asreport
        python -m wbia autogen_ipynb --ipynb --db PZ_MTEST --noexample --withtags
        python -m wbia autogen_ipynb --ipynb --db PZ_MTEST

        python -m wbia autogen_ipynb --ipynb --db STS_SandTigers

        python -m wbia autogen_ipynb --db PZ_MTEST
        # TODO: Add support for dbdir to be specified
        python -m wbia autogen_ipynb --db ~/work/PZ_MTEST

        python -m wbia autogen_ipynb --ipynb --db Oxford -a default:qhas_any=\(query,\),dpername=1,exclude_reference=True,dminqual=good
        python -m wbia autogen_ipynb --ipynb --db PZ_MTEST -a default -t best:lnbnn_normalizer=[None,normlnbnn-test]

        python -m wbia.templates.generate_notebook --exec-autogen_ipynb --db wd_peter_blinston --ipynb

        python -m wbia autogen_ipynb --db PZ_Master1 --ipynb
        python -m wbia autogen_ipynb --db PZ_Master1 -a timectrl:qindex=0:100 -t best best:normsum=True --ipynb --noexample
        python -m wbia autogen_ipynb --db PZ_Master1 -a timectrl --run
        jupyter-notebook Experiments-lynx.ipynb
        killall python

        python -m wbia autogen_ipynb --db humpbacks --ipynb -t default:proot=BC_DTW -a default:has_any=hasnotch
        python -m wbia autogen_ipynb --db humpbacks --ipynb -t default:proot=BC_DTW default:proot=vsmany -a default:has_any=hasnotch,mingt=2,qindex=0:50 --noexample

        python -m wbia autogen_ipynb --db testdb_curvrank --ipynb -t default:proot=CurvRankDorsal
        python -m wbia autogen_ipynb --db testdb_curvrank --ipynb -t default:proot=CurvRankFluke
        python -m wbia autogen_ipynb --db PW_Master --ipynb -t default:proot=CurvRankDorsal

        python -m wbia autogen_ipynb --db testdb_identification --ipynb -t default:proot=Deepsense

    Ignore:
        python -m wbia autogen_ipynb --db WS_ALL

    Example:
        >>> # SCRIPT
        >>> from wbia.templates.generate_notebook import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> result = autogen_ipynb(ibs)
        >>> print(result)
    """
    dbname = ibs.get_dbname()
    fname = 'Experiments-' + dbname
    nb_fpath = fname + '.ipynb'
    if ut.get_argflag('--cells'):
        notebook_cells = make_wbia_cell_list(ibs)
        print('\n# ---- \n'.join(notebook_cells))
        return
    # TODO: Add support for dbdir to be specified
    notebook_str = make_wbia_notebook(ibs)
    ut.writeto(nb_fpath, notebook_str)
    run = ut.get_argflag('--run') if run is None else run
    launch = launch if launch is not None else ut.get_argflag('--ipynb')
    if run:
        run_nb = ut.run_ipython_notebook(notebook_str)
        output_fpath = ut.export_notebook(run_nb, fname)
        ut.startfile(output_fpath)
    elif launch:
        command = ' '.join(
            [
                'jupyter-notebook',
                '--NotebookApp.iopub_data_rate_limit=10000000',
                '--NotebookApp.token=',
                nb_fpath,
            ]
        )
        ut.cmd2(command, detatch=True, verbose=True)
    else:
        print('notebook_str =\n%s' % (notebook_str,))


def get_default_cell_template_list(ibs):
    """
    Defines the order of ipython notebook cells
    """
    cells = notebook_cells

    noexample = not ut.get_argflag('--examples')
    asreport = ut.get_argflag('--asreport')
    withtags = ut.get_argflag('--withtags')

    cell_template_list = []

    info_cells = [
        cells.pipe_config_info,
        cells.annot_config_info,
        # cells.per_encounter_stats,
        cells.timestamp_distribution,
    ]

    dev_analysis = [
        cells.config_overlap,
        # cells.dbsize_expt,
        # None if ibs.get_dbname() == 'humpbacks' else cells.feat_score_sep,
        cells.all_annot_scoresep,
        cells.success_annot_scoresep,
    ]

    cell_template_list += [
        cells.introduction if asreport else None,
        cells.nb_init,
        cells.db_init,
        None if ibs.get_dbname() != 'humpbacks' else cells.fluke_select,
    ]

    if not asreport:
        cell_template_list += info_cells

    if not noexample:
        cell_template_list += [
            cells.example_annotations,
            cells.example_names,
        ]

    cell_template_list += [
        cells.per_annotation_accuracy,
        cells.per_name_accuracy,
        cells.easy_success_cases,
        cells.hard_success_cases,
        cells.failure_type1_cases,
        cells.failure_type2_cases,
        cells.total_failure_cases,
        cells.timedelta_distribution,
    ]

    if withtags:
        cell_template_list += [
            cells.investigate_specific_case,
            cells.view_intereseting_tags,
        ]

    if asreport:
        # Append our debug stuff at the bottom
        cell_template_list += [cells.IGNOREAFTER]
        cell_template_list += info_cells

    cell_template_list += dev_analysis

    cell_template_list += [
        cells.config_disagree_cases,
    ]

    cell_template_list = ut.filter_Nones(cell_template_list)

    cell_template_list = ut.lmap(ut.normalize_cells, cell_template_list)

    if not asreport:
        # Remove all of the extra fluff
        cell_template_list = [
            (header.split('\n')[0], code, None)
            for (header, code, footer) in cell_template_list
        ]

    return cell_template_list


def make_wbia_notebook(ibs):
    r"""
    Args:
        ibs (wbia.IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.templates.generate_notebook --exec-make_wbia_notebook --db wd_peter_blinston --asreport
        python -m wbia --tf --exec-make_wbia_notebook
        python -m wbia --tf make_wbia_notebook --db lynx
        jupyter-notebook tmp.ipynb
        runipy tmp.ipynb --html report.html
        runipy --pylab tmp.ipynb tmp2.ipynb
        sudo pip install runipy
        python -c "import runipy; print(runipy.__version__)"

    Example:
        >>> # SCRIPT
        >>> from wbia.templates.generate_notebook import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> notebook_str = make_wbia_notebook(ibs)
        >>> print(notebook_str)
    """
    cell_list = make_wbia_cell_list(ibs)
    notebook_str = ut.make_notebook(cell_list)
    return notebook_str


def make_wbia_cell_list(ibs):
    cell_template_list = get_default_cell_template_list(ibs)
    autogen_str = '# python -m wbia autogen_ipynb --launch --dbdir %r' % (ibs.get_dbdir())
    # autogen_str = ut.make_autogen_str()
    dbname = ibs.get_dbname()
    dbdir = ibs.dbdir
    default_acfgstr = ut.get_argval('-a', type_=str, default='default:is_known=True')

    asreport = ut.get_argflag('--asreport')

    default_pcfgstr_list = ut.get_argval(('-t', '-p'), type_=list, default='default')
    default_pcfgstr = ut.repr3(default_pcfgstr_list, nobr=True)

    if asreport:
        annotconfig_list_body = ut.codeblock(ut.repr2(default_acfgstr))
        pipeline_list_body = ut.codeblock(default_pcfgstr)
    else:
        annotconfig_list_body = ut.codeblock(
            ut.repr2(default_acfgstr)
            + '\n'
            + ut.codeblock(
                """
            #'default:has_any=(query,),dpername=1,exclude_reference=True',
            #'default:is_known=True',
            #'default:is_known=True,minqual=good,require_timestamp=True,dcrossval_enc=1,view=left'
            #'default:qsame_imageset=True,been_adjusted=True,excluderef=True,qsize=10,dsize=20',
            #'default:require_timestamp=True,min_timedelta=3600',
            #'default:species=primary',
            #'unctrl:been_adjusted=True',
            #'timectrl:',
            #'timectrl:view=primary,minqual=good',

            #'default:minqual=good,require_timestamp=True,view=left,dcrossval_enc=1,joinme=1',
            #'default:minqual=good,require_timestamp=True,view=right,dcrossval_enc=1,joinme=1',
            #'default:minqual=ok,require_timestamp=True,view=left,dcrossval_enc=1,joinme=2',
            #'default:minqual=ok,require_timestamp=True,view=right,dcrossval_enc=1,joinme=2',

            """
            )
        )
        pipeline_list_body = ut.codeblock(
            default_pcfgstr
            + '\n'
            + ut.codeblock(
                """
            #'default',
            #'default:K=1,AI=False,QRH=True',
            #'default:K=1,RI=True,AI=False',
            #'default:K=1,adapteq=True',
            #'default:fg_on=[True,False]',
            """
            )
        )

    locals_ = locals()
    _format = partial(ut.format_cells, locals_=locals_)
    cell_list = ut.flatten(map(_format, cell_template_list))
    return cell_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.templates.generate_notebook
        python -m wbia.templates.generate_notebook --allexamples
        python -m wbia.templates.generate_notebook --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
