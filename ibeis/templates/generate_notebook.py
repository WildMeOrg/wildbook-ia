# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut


class NotebookCells(object):
    initialize_cell = ut.codeblock(
        r'''
        # STARTBLOCK
        # Set global utool flags
        import utool as ut
        ut.util_io.__PRINT_WRITES__ = False
        ut.util_io.__PRINT_READS__ = False
        ut.util_parallel.__FORCE_SERIAL__ = True
        ut.util_cache.VERBOSE_CACHE = False
        ut.NOT_QUIET = False

        # Matplotlib stuff
        %pylab inline
        %load_ext autoreload
        %autoreload

        draw_case_kw = dict(show_in_notebook=True, annot_modes=[0, 1])

        # Setup database specific parameter configurations
        db = '{dbname}'
        #a = ['default:qsame_encounter=True,been_adjusted=True,excluderef=True']
        a = ['default:is_known=True']
        t = ['default:K=1']

        # Load database for this test run
        import ibeis
        ibs = ibeis.opendb(db=db)
        # ENDBLOCK
        ''')

    timestamp_distribution = ut.codeblock(
        r'''
        # STARTBLOCK
        ibeis.other.dbinfo.show_image_time_distributions(ibs, ibs.get_valid_gids())
        # ENDBLOCK
        ''')

    detection_summary = ut.codeblock(
        r'''
        # STARTBLOCK
        # Get a sample of images
        if False:
            gids = ibs.get_valid_gids()
        else:
            from ibeis.init.filter_annots import expand_single_acfg
            from ibeis.expt import experiment_helpers
            acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, [a[0]], use_cache=False)
            qaids, daids = expanded_aids_list[0]
            all_aids = ut.flatten([qaids, daids])
            gids = ut.unique_keep_order2(ibs.get_annot_gids(all_aids))

        aids = ibs.get_image_aids(gids)

        nAids_list = list(map(len, aids))
        gids_sorted = ut.sortedby(gids, nAids_list)[::-1]
        samplex = list(range(5))
        print(samplex)
        gids_sample = ut.list_take(gids_sorted, samplex)

        import ibeis.viz
        for gid in ut.ProgressIter(gids_sample, lbl='drawing image'):
            ibeis.viz.show_image(ibs, gid)
        # ENDBLOCK
    ''')

    per_annotation_accuracy = ut.codeblock(
        r'''
        # STARTBLOCK
        testres = ibeis.run_experiment(
            e='rank_cdf',
            db=db, a=a, t=t)
        #testres.print_unique_annot_config_stats()
        _ = testres.draw_func()
        # ENDBLOCK
        '''
    )

    success_scores = ut.codeblock(
        r'''
        # STARTBLOCK
        testres = ibeis.run_experiment(
            e='scores',
            db=db, a=a, t=t,
            f=[':fail=False,min_gf_timedelta=None'],
        )
        _ = testres.draw_func()
        # ENDBLOCK
        ''')

    all_scores = ut.codeblock(
        r'''
        # STARTBLOCK
        testres = ibeis.run_experiment(
            e='scores',
            db=db, a=a, t=t,
            f=[':fail=None,min_gf_timedelta=None']
        )
        _ = testres.draw_func()
        # ENDBLOCK
        ''')

    success_cases = ut.codeblock(
        r'''
        # STARTBLOCK
        testres = ibeis.run_experiment(
            e='draw_cases',
            db=db, a=a, t=t,
            f=[':fail=False,index=0:3,sortdsc=gtscore,without_gf_tag=Photobomb,max_pername=1'],
            **draw_case_kw)
        _ = testres.draw_func()
        # ENDBLOCK
        ''')

    failure_type2_cases = ut.codeblock(
        r'''
        # STARTBLOCK
        testres = ibeis.run_experiment(
            e='draw_cases',
            db=db, a=a, t=t,
            f=[':fail=True,index=0:3,sortdsc=gtscore,max_pername=1'],
            **draw_case_kw)
        _ = testres.draw_func()
        # ENDBLOCK
        ''')

    failure_type1_cases = ut.codeblock(
        r'''
        # STARTBLOCK
        testres = ibeis.run_experiment(
        e='draw_cases',
        db=db, a=a, t=t,
        f=[':fail=True,index=0:3,sortdsc=gfscore,max_pername=1'],
        **draw_case_kw)
        _ = testres.draw_func()
        # ENDBLOCK
        ''')


def make_default_notebook(ibs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.templates.generate_notebook --exec-make_default_notebook
        jupyter-notebook tmp.ipynb

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.templates.generate_notebook import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> result = make_default_notebook(ibs)
        >>> print(result)
    """
    dbname = ibs.get_dbname()
    cell_list = [
        #None,
        #'',
        code_cell(NotebookCells.initialize_cell.format(dbname=dbname)),
        markdown_cell('# Timestamp Distribution'),
        code_cell(NotebookCells.timestamp_distribution),
        markdown_cell('# Query Accuracy (% correct annotations)'),
        code_cell(NotebookCells.per_annotation_accuracy),
        markdown_cell('# Score Distribution'),
        code_cell(NotebookCells.all_scores),
        markdown_cell('# Success Cases'),
        code_cell(NotebookCells.success_cases),
        markdown_cell('# Failure Cases Cases (false pos)'),
        code_cell(NotebookCells.failure_type1_cases),
        markdown_cell('# Failure Cases Cases (false neg)'),
        code_cell(NotebookCells.failure_type2_cases),
    ]

    make_notebook(cell_list)


def code_cell(sourcecode):
    from ibeis.templates.template_generator import remove_sentinals
    sourcecode = remove_sentinals(sourcecode)
    cell_header = ut.codeblock(
        """
        {
         "cell_type": "code",
         "execution_count": null,
         "metadata": {
          "collapsed": true
         },
         "outputs": [],
         "source":
        """)
    cell_footer = ut.codeblock(
        """
        }
        """)
    if sourcecode is None:
        source_line_repr = ' []\n'
    else:
        lines = sourcecode.split('\n')
        line_list = [line + '\n' if count < len(lines) else line for count, line in enumerate(lines, start=1)]
        repr_line_list = [repr_single(line) for line in line_list]
        source_line_repr = ut.indent(',\n'.join(repr_line_list), ' ' * 2)
        source_line_repr = ' [\n' + source_line_repr + '\n ]\n'
    return (cell_header + source_line_repr + cell_footer)


def markdown_cell(markdown):
    markdown_header = ut.codeblock(
        """
          {
           "cell_type": "markdown",
           "metadata": {},
           "source": [
        """
    )
    markdown_footer = ut.codeblock(
        """
           ]
          }
        """
    )
    return (markdown_header + '\n' + ut.indent(repr_single(markdown), ' ' * 2) + '\n' + markdown_footer)


def make_notebook(cell_list):
    header = ut.codeblock(
        """
        {
         "cells": [
        """
    )

    footer = ut.codeblock(
        """
         ],
         "metadata": {
          "kernelspec": {
           "display_name": "Python 2",
           "language": "python",
           "name": "python2"
          },
          "language_info": {
           "codemirror_mode": {
            "name": "ipython",
            "version": 2
           },
           "file_extension": ".py",
           "mimetype": "text/x-python",
           "name": "python",
           "nbconvert_exporter": "python",
           "pygments_lexer": "ipython2",
           "version": "2.7.6"
          }
         },
         "nbformat": 4,
         "nbformat_minor": 0
        }
        """)

    cell_body = ut.indent(',\n'.join(cell_list), '  ')
    nodebook_str = header + '\n' + cell_body +  '\n' +  footer
    ut.writeto('tmp.ipynb', nodebook_str)
    print('nodebook_str =\n%s' % (nodebook_str,))


def repr_single(s):
    return "\"" + repr('\'' + s)[2:]


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.templates.generate_notebook
        python -m ibeis.templates.generate_notebook --allexamples
        python -m ibeis.templates.generate_notebook --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
