# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join, dirname, split, basename, splitext
import re
import utool as ut
from six.moves import map, range

print, rrr, profile = ut.inject2(__name__)


class IndividualResultsCopyTaskQueue(object):
    def __init__(self):
        self.cp_task_list = []

    def append_copy_task(self, fpath_orig, dstdir=None):
        """ helper which copies a summary figure to root dir """
        fname_orig, ext = splitext(basename(fpath_orig))
        outdir = dirname(fpath_orig)
        fdir_clean, cfgdir = split(outdir)
        if dstdir is None:
            dstdir = fdir_clean
        # aug = cfgdir[0:min(len(cfgdir), 10)]
        aug = cfgdir
        fname_fmt = '{aug}_{fname_orig}{ext}'
        fmt_dict = {'aug': aug, 'fname_orig': fname_orig, 'ext': ext}
        fname_clean = ut.long_fname_format(
            fname_fmt, fmt_dict, ['fname_orig'], max_len=128
        )
        fdst_clean = join(dstdir, fname_clean)
        self.cp_task_list.append((fpath_orig, fdst_clean))

    def flush_copy_tasks(self):
        # Execute all copy tasks and empty the lists
        if ut.NOT_QUIET:
            print('[DRAW_RESULT] copying %r summaries' % (len(self.cp_task_list)))
        for src, dst in self.cp_task_list:
            ut.copy(src, dst, verbose=False)
        del self.cp_task_list[:]


def make_individual_latex_figures(
    ibs, fpaths_list, flat_case_labels, cfgx2_shortlbl, case_figdir, analysis_fpath_list
):
    # HACK MAKE LATEX CONVINENCE STUFF
    # print('LATEX HACK')
    if len(fpaths_list) == 0:
        print('nothing to render')
        return
    RENDER = ut.get_argflag('--render')
    DUMP_FIGDEF = ut.get_argflag(('--figdump', '--dump-figdef', '--figdef'))

    if not (DUMP_FIGDEF or RENDER):  # HACK
        return

    latex_code_blocks = []
    latex_block_keys = []

    caption_prefix = ut.get_argval('--cappref', type_=str, default='')
    caption_suffix = ut.get_argval('--capsuf', type_=str, default='')
    cmdaug = ut.get_argval('--cmdaug', type_=str, default='custom')

    selected = None

    for case_idx, (fpaths, labels) in enumerate(zip(fpaths_list, flat_case_labels)):
        if labels is None:
            labels = [cmdaug]
        if len(fpaths) < 4:
            nCols = len(fpaths)
        else:
            nCols = 2

        _cmdname = ibs.get_dbname() + ' Case ' + ' '.join(labels) + '_' + str(case_idx)
        # print('_cmdname = %r' % (_cmdname,))
        cmdname = ut.latex_sanitize_command_name(_cmdname)
        label_str = cmdname
        if len(caption_prefix) == 0:
            caption_str = ut.escape_latex(
                'Casetags: '
                + ut.repr2(labels, nl=False, strvals=True)
                + ', db='
                + ibs.get_dbname()
                + '. '
            )
        else:
            caption_str = ''

        use_sublbls = len(cfgx2_shortlbl) > 1
        if use_sublbls:
            caption_str += ut.escape_latex(
                'Each figure shows a different configuration: '
            )
            sublbls = [
                '(' + chr(97 + count) + ') ' for count in range(len(cfgx2_shortlbl))
            ]
        else:
            # caption_str += ut.escape_latex('This figure depicts correct and
            # incorrect matches from configuration: ')
            sublbls = [''] * len(cfgx2_shortlbl)

        def wrap_tt(text):
            return r'{\tt ' + text + '}'

        _shortlbls = cfgx2_shortlbl
        _shortlbls = list(map(ut.escape_latex, _shortlbls))
        # Adjust spacing for breaks
        # tex_small_space = r''
        tex_small_space = r'\hspace{0pt}'
        # Remove query specific config flags in individual results
        _shortlbls = [re.sub('\\bq[^,]*,?', '', shortlbl) for shortlbl in _shortlbls]
        # Let config strings be broken over newlines
        _shortlbls = [
            re.sub('\\+', tex_small_space + '+' + tex_small_space, shortlbl)
            for shortlbl in _shortlbls
        ]
        _shortlbls = [
            re.sub(', *', ',' + tex_small_space, shortlbl) for shortlbl in _shortlbls
        ]
        _shortlbls = list(map(wrap_tt, _shortlbls))
        cfgx2_texshortlbl = [
            '\n    ' + lbl + shortlbl for lbl, shortlbl in zip(sublbls, _shortlbls)
        ]

        caption_str += ut.conj_phrase(cfgx2_texshortlbl, 'and') + '.\n    '
        caption_str = '\n    ' + caption_prefix + caption_str + caption_suffix
        caption_str = caption_str.rstrip()
        figure_str = ut.get_latex_figure_str(
            fpaths,
            nCols=nCols,
            label_str=label_str,
            caption_str=caption_str,
            use_sublbls=None,
            use_frame=True,
        )
        latex_block = ut.latex_newcommand(cmdname, figure_str)
        latex_block = '\n%----------\n' + latex_block
        latex_code_blocks.append(latex_block)
        latex_block_keys.append(cmdname)

    # HACK
    remove_fpath = ut.truepath('~/latex/crall-candidacy-2015') + '/'

    latex_fpath = join(case_figdir, 'latex_cases.tex')

    if selected is not None:
        selected_keys = selected
    else:
        selected_keys = latex_block_keys

    selected_blocks = ut.dict_take(
        dict(zip(latex_block_keys, latex_code_blocks)), selected_keys
    )

    figdef_block = '\n'.join(selected_blocks)
    figcmd_block = '\n'.join(['\\' + key for key in latex_block_keys])

    selected_block = figdef_block + '\n\n' + figcmd_block

    # HACK: need full paths to render
    selected_block_renderable = selected_block
    selected_block = selected_block.replace(remove_fpath, '')
    if RENDER:
        ut.render_latex_text(selected_block_renderable)

    if DUMP_FIGDEF:
        ut.writeto(latex_fpath, selected_block)

    # if NOT DUMP AND NOT RENDER:
    #    print('STANDARD LATEX RESULTS')
    #    cmdname = ibs.get_dbname() + 'Results'
    #    latex_block  = ut.get_latex_figure_str2(analysis_fpath_list, cmdname, nCols=1)
    #    ut.print_code(latex_block, 'latex')
    if DUMP_FIGDEF or RENDER:
        ut.print_code(selected_block, 'latex')
