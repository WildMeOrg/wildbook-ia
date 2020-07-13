# -*- coding: utf-8 -*-
"""
Matplotlib interface for name interactions. Allows for relatively fine grained
control of splitting and merging.

DEPRICATE

CommandLine:
    python -m wbia.viz.interact.interact_name --test-ishow_name --show
    python -m wbia.viz.interact.interact_name --test-testsdata_match_verification --show --db PZ_MTEST --aid1 1 --aid2 30
    python -m wbia.viz.interact.interact_name --test-testsdata_match_verification --show --db PZ_MTEST --aid1 30 --aid2 32


"""
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
from six.moves import zip
from wbia.plottool import interact_helpers as ih
import functools
import wbia.plottool as pt
from wbia import viz
from wbia import constants as const
from wbia.viz import viz_helpers as vh
from wbia.other import ibsfuncs
from wbia.viz import viz_chip
from wbia.plottool.abstract_interaction import AbstractInteraction

(print, rrr, profile) = ut.inject2(__name__, '[interact_name]', DEBUG=False)


# ==========================
# Name Interaction
# ==========================

MAX_COLS = 3


def build_name_context_options(ibs, nids):
    print('build_name_context_options nids = %r' % (nids,))
    callback_list = []
    from wbia.viz import viz_graph2

    callback_list.extend(
        [
            (
                'New Split Interact (Name)',
                functools.partial(viz_graph2.make_qt_graph_interface, ibs, nids=nids),
            ),
        ]
    )
    return callback_list


def ishow_name(
    ibs, nid, sel_aids=[], select_aid_callback=None, fnum=5, dodraw=True, **kwargs
):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        nid (?):
        sel_aids (list):
        select_aid_callback (None):
        fnum (int):  figure number

    CommandLine:
        python -m wbia.viz.interact.interact_name --test-ishow_name --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.interact.interact_name import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> nid = ut.get_argval('--nid', int, default=1)
        >>> sel_aids = []
        >>> select_aid_callback = None
        >>> fnum = 5
        >>> dodraw = ut.show_was_requested()
        >>> # execute function
        >>> result = ishow_name(ibs, nid, sel_aids, select_aid_callback, fnum, dodraw)
        >>> # verify results
        >>> pt.show_if_requested()
        >>> print(result)
    """
    if fnum is None:
        fnum = pt.next_fnum()
    fig = ih.begin_interaction('name', fnum)

    def _on_name_click(event):
        ax = event.inaxes
        if ih.clicked_inside_axis(event):
            viztype = vh.get_ibsdat(ax, 'viztype')
            if viztype == 'chip':
                aid = vh.get_ibsdat(ax, 'aid')
                print('... aid=%r' % aid)
                if event.button == 3:  # right-click
                    from wbia import guitool
                    from wbia.viz.interact import interact_chip

                    height = fig.canvas.geometry().height()
                    qpoint = guitool.newQPoint(event.x, height - event.y)
                    refresh_func = functools.partial(
                        viz.show_name, ibs, nid, fnum=fnum, sel_aids=sel_aids
                    )
                    interact_chip.show_annot_context_menu(
                        ibs,
                        aid,
                        fig.canvas,
                        qpoint,
                        refresh_func=refresh_func,
                        with_interact_name=False,
                    )
                else:
                    viz.show_name(ibs, nid, fnum=fnum, sel_aids=[aid], in_image=True)
                    if select_aid_callback is not None:
                        select_aid_callback(aid)
        viz.draw()

    viz.show_name(ibs, nid, fnum=fnum, sel_aids=sel_aids, in_image=True)
    if dodraw:
        viz.draw()
    ih.connect_callback(fig, 'button_press_event', _on_name_click)
    pass


def testsdata_match_verification(defaultdb='testdb1', aid1=1, aid2=2):
    r"""
    CommandLine:
        main.py --imgsetid 2
        main.py --imgsetid 13 --db PZ_MUGU_19

    CommandLine:
        python -m wbia.viz.interact.interact_name --test-testsdata_match_verification --show
        python -m wbia.viz.interact.interact_name --test-testsdata_match_verification --aid1 2 --aid2 3 --show

        # Merge case
        python -m wbia.viz.interact.interact_name --test-testsdata_match_verification --show --db PZ_MTEST --aid1 1 --aid2 30

        # Split case
        python -m wbia.viz.interact.interact_name --test-testsdata_match_verification --show --db PZ_MTEST --aid1 30 --aid2 32

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.viz.interact.interact_name import *  # NOQA
        >>> self = testsdata_match_verification()
        >>> # verify results
        >>> ut.quit_if_noshow()
        >>> self.show_page()
        >>> ut.show_if_requested()
    """
    # from wbia.viz.interact.interact_name import *  # NOQA
    import wbia

    # ibs = wbia.opendb(defaultdb='PZ_Master0')
    ibs = wbia.opendb(defaultdb=defaultdb)
    # aid1 = ut.get_argval('--aid1', int, 14)
    # aid2 = ut.get_argval('--aid2', int, 5545)
    aid1 = ut.get_argval('--aid1', int, aid1)
    aid2 = ut.get_argval('--aid2', int, aid2)
    self = MatchVerificationInteraction(ibs, aid1, aid2, dodraw=False)
    return self


class MatchVerificationInteraction(AbstractInteraction):
    def __init__(
        self,
        ibs,
        aid1,
        aid2,
        update_callback=None,
        backend_callback=None,
        dodraw=True,
        max_cols=MAX_COLS,
        **kwargs,
    ):
        if ut.VERBOSE:
            print('[matchver] __init__')
        if ut.VERBOSE or ut.is_developer():
            print('[matchver] __init__ aid1=%r, aid2=%r ' % (aid1, aid2))
        super(MatchVerificationInteraction, self).__init__(**kwargs)
        self.ibs = ibs
        self.max_cols = max_cols
        self.aid1 = aid1
        self.aid2 = aid2
        self.col_offset_list = [0, 0]
        # ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])

        def _nonefn():
            return None

        if update_callback is None:
            update_callback = _nonefn
        if backend_callback is None:
            backend_callback = _nonefn
        self.update_callback = (
            update_callback  # if something like qt needs a manual refresh on change
        )
        self.backend_callback = backend_callback
        self.qres_callback = kwargs.get('qres_callback', None)
        self.cm = kwargs.get('cm', None)
        self.qreq_ = kwargs.get('qreq_', None)
        if self.cm is not None:
            from wbia.algo.hots import chip_match

            assert isinstance(self.cm, chip_match.ChipMatch)
            assert self.qreq_ is not None
        self.infer_data()
        if dodraw:
            self.show_page(bring_to_front=True)

    def infer_data(self):
        """ Initialize data related to the input aids
        """
        ibs = self.ibs
        # The two matching aids
        self.aid_pair = (self.aid1, self.aid2)
        (aid1, aid2) = self.aid_pair
        self.match_text = ibs.get_match_text(self.aid1, self.aid2)
        # The names of the matching annotations
        self.nid1, self.nid2 = ibs.get_annot_name_rowids((aid1, aid2))
        self.name1, self.name2 = ibs.get_annot_names((aid1, aid2))
        self.other_valid_nids = []
        # The other annotations that belong to these two names
        self.gts_list = ibs.get_annot_groundtruth((aid1, aid2))
        self.gt1, self.gt2 = self.gts_list
        # A flat list of all the aids we are looking at
        self.is_split_case = self.nid1 == self.nid2
        self.all_aid_list = ut.unique_ordered([aid1, aid2] + self.gt1 + self.gt2)
        self.all_nid_list_orig = ibs.get_annot_name_rowids(self.all_aid_list)
        self.other_aids = list(set(self.all_aid_list) - set([self.aid1, self.aid2]))

        if self.is_split_case:
            # Split case
            self.nCols = max(2, len(self.other_aids))
            self.nRows = 2 if len(self.other_aids) > 0 else 1
        else:
            # Merge/New Match case
            self.nCols = max(len(self.gt1) + 1, len(self.gt2) + 1)
            self.nRows = 2
        self.nCols = min(self.max_cols, self.nCols)

        # Grab not just the exemplars

        if ut.VERBOSE or ut.is_developer():
            print('[matchver] __init__ nid1=%r, nid2=%r ' % (self.nid1, self.nid2))
            print('[matchver] __init__ self.gts_list=%r ' % (self.gts_list))

        if ut.VERBOSE or ut.is_developer():
            print('[matchver] __init__ nid1=%r, nid2=%r ' % (self.nid1, self.nid2))
            print('[matchver] __init__ self.gts_list=%r ' % (self.gts_list))

    def get_other_nids(self):
        ibs = self.ibs
        all_nid_list = ibs.get_annot_name_rowids(self.all_aid_list)
        unique_nid_list = ut.unique_ordered(all_nid_list)
        is_unknown = ibs.is_nid_unknown(unique_nid_list)
        is_name1 = [nid == self.nid1 for nid in unique_nid_list]
        is_name2 = [nid == self.nid2 for nid in unique_nid_list]
        is_other = ut.and_lists(
            *tuple(map(ut.not_list, (is_name1, is_name2, is_unknown)))
        )
        other_nid_list = ut.compress(unique_nid_list, is_other)
        return other_nid_list

    def get_rotating_columns(self, rowx):
        if self.is_split_case:
            if rowx == 0:
                return []
            else:
                return self.other_aids
        else:
            if rowx == 0:
                return self.gt1
            else:
                return self.gt2

    def get_non_rotating_columns(self, rowx):
        if self.is_split_case:
            if rowx == 0:
                return [self.aid1, self.aid2]
            else:
                return []
        else:
            if rowx == 0:
                return [self.aid1]
            else:
                return [self.aid2]

    def get_row_aids_list(self):
        r"""
        Args:

        Returns:
            list: row_aids_list

        CommandLine:
            python -m wbia.viz.interact.interact_name --test-get_row_aids_list

        CommandLine:
            python -m wbia.viz.interact.interact_name --test-get_row_aids_list
            python -m wbia.viz.interact.interact_name --test-get_row_aids_list --aid1 2 --aid2 3
            # Merge case
            python -m wbia.viz.interact.interact_name --test-get_row_aids_list --db PZ_MTEST --aid1 1 --aid2 30
            # Split case
            python -m wbia.viz.interact.interact_name --test-get_row_aids_list --db PZ_MTEST --aid1 30 --aid2 32

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.viz.interact.interact_name import *  # NOQA
            >>> # build test data
            >>> self = testsdata_match_verification('PZ_MTEST', 30, 32)
            >>> # execute function
            >>> row_aids_list = self.get_row_aids_list()
            >>> # verify results
            >>> result = str(row_aids_list)
            >>> print(result)
            >>> ut.quit_if_noshow()
            >>> self.show_page()
            >>> ut.show_if_requested()
        """

        def get_row(rowx):
            row_offset = self.col_offset_list[rowx]
            row_nonrotate_part = self.get_non_rotating_columns(rowx)
            row_rotate_part_ = self.get_rotating_columns(rowx)
            row_rotate_part = ut.list_roll(row_rotate_part_, -row_offset)
            row = row_nonrotate_part + row_rotate_part
            return row

        row_aids_list_ = [get_row(rowx) for rowx in range(self.nRows)]
        row_aids_list = list(filter(lambda x: len(x) > 0, row_aids_list_))
        return row_aids_list

    def rotate_row(self, event=None, rowx=None):
        """
        shows the next few annotations in this row
        (implicitly rotates the row's columns the rows columns)
        """
        modbase = len(self.get_rotating_columns(rowx))
        self.col_offset_list[rowx] = (1 + self.col_offset_list[rowx]) % modbase
        # self.gts_list[rowx] = list_roll(self.gts_list[rowx], -(self.nCols - 1))
        self.show_page(onlyrows=[rowx], fulldraw=False)

    def prepare_page(self, fulldraw=True):
        figkw = {
            'fnum': self.fnum,
            'doclf': fulldraw,
            'docla': fulldraw,
        }
        if fulldraw:
            self.fig = pt.figure(**figkw)
        ih.disconnect_callback(self.fig, 'button_press_event')
        ih.disconnect_callback(self.fig, 'key_press_event')
        ih.connect_callback(self.fig, 'button_press_event', self.figure_clicked)
        ih.connect_callback(self.fig, 'key_press_event', self.on_key_press)

    def show_page(self, bring_to_front=False, onlyrows=None, fulldraw=True):
        """ Plots all subaxes on a page

        onlyrows is a hack to only draw a subset of the data again
        """
        if ut.VERBOSE:
            if not fulldraw:
                print(
                    '[matchver] show_page(fulldraw=%r, onlyrows=%r)'
                    % (fulldraw, onlyrows)
                )
            else:
                print('[matchver] show_page(fulldraw=%r)' % (fulldraw))
        self.prepare_page(fulldraw=fulldraw)
        # Variables we will work with to paint a pretty picture
        ibs = self.ibs
        nRows = self.nRows
        colpad = 1 if self.cm is not None else 0
        nCols = self.nCols + colpad

        # Distinct color for every unique name
        unique_nids = ut.unique_ordered(
            ibs.get_annot_name_rowids(self.all_aid_list, distinguish_unknowns=False)
        )
        unique_colors = pt.distinct_colors(
            len(unique_nids), brightness=0.7, hue_range=(0.05, 0.95)
        )
        self.nid2_color = dict(zip(unique_nids, unique_colors))

        row_aids_list = self.get_row_aids_list()

        if self.cm is not None:
            print('DRAWING QRES')
            pnum = (1, nCols, 1)
            if not fulldraw:
                # not doing full draw so we have to clear any axes
                # that are here already manually
                ax = self.fig.add_subplot(*pnum)
                self.clear_parent_axes(ax)
            self.cm.show_single_annotmatch(
                self.qreq_,
                self.aid2,
                fnum=self.fnum,
                pnum=pnum,
                draw_fmatch=True,
                colorbar_=False,
            )

        # For each row
        for rowx, aid_list in enumerate(row_aids_list):
            offset = rowx * nCols + 1
            if onlyrows is not None and rowx not in onlyrows:
                continue
            # ibsfuncs.assert_valid_aids(ibs, groundtruth)
            # For each column
            for colx, aid in enumerate(aid_list, start=colpad):
                if colx >= nCols:
                    break
                try:
                    nid = ibs.get_annot_name_rowids(aid)
                    if ibsfuncs.is_nid_unknown(ibs, [nid])[0]:
                        color = const.UNKNOWN_PURPLE_RGBA01
                    else:
                        color = self.nid2_color[nid]
                except Exception as ex:
                    ut.printex(ex)
                    print('nid = %r' % (nid,))
                    print('self.nid2_color = %s' % (ut.repr2(self.nid2_color),))
                    raise
                px = colx + offset
                ax = self.plot_chip(
                    int(aid), nRows, nCols, px, color=color, fulldraw=fulldraw
                )
                # If there are still more in this row to display
                if colx + 1 < len(aid_list) and colx + 1 >= nCols:
                    total_indices = len(aid_list)
                    current_index = self.col_offset_list[rowx] + 1
                    next_text = 'next\n%d/%d' % (current_index, total_indices)
                    next_func = functools.partial(self.rotate_row, rowx=rowx)
                    self.append_button(
                        next_text,
                        callback=next_func,
                        location='right',
                        size='33%',
                        ax=ax,
                    )

        if fulldraw:
            self.show_hud()
            hspace = 0.05 if (self.nCols) > 1 else 0.1
            subplotspar = {
                'left': 0.1,
                'right': 0.9,
                'top': 0.85,
                'bottom': 0.1,
                'wspace': 0.3,
                'hspace': hspace,
            }
            pt.adjust_subplots(**subplotspar)
        self.draw()
        self.show()
        if bring_to_front:
            self.bring_to_front()
        # self.update()

    def plot_chip(self, aid, nRows, nCols, px, fulldraw=True, **kwargs):
        """ Plots an individual chip in a subaxis """
        ibs = self.ibs
        if aid in [self.aid1, self.aid2]:
            # Bold color for the matching chips
            lw = 5
            text_color = np.array((135, 206, 235, 255)) / 255.0
        else:
            lw = 2
            text_color = None

        pnum = (nRows, nCols, px)
        if not fulldraw:
            # not doing full draw so we have to clear any axes
            # that are here already manually
            ax = self.fig.add_subplot(*pnum)
            self.clear_parent_axes(ax)
            # ut.embed()
            # print(subax)

        viz_chip_kw = {
            'fnum': self.fnum,
            'pnum': pnum,
            'nokpts': True,
            'show_name': True,
            'show_gname': False,
            'show_aidstr': True,
            'notitle': True,
            'show_num_gt': False,
            'text_color': text_color,
        }
        if False and ut.is_developer():
            enable_chip_title_prefix = True
            viz_chip_kw.update(
                {
                    'enable_chip_title_prefix': enable_chip_title_prefix,
                    'show_name': True,
                    'show_aidstr': True,
                    'show_viewcode': True,
                    'show_num_gt': True,
                    'show_quality_text': True,
                }
            )

        viz_chip.show_chip(ibs, aid, **viz_chip_kw)
        ax = pt.gca()
        pt.draw_border(ax, color=kwargs.get('color'), lw=lw)
        if kwargs.get('make_buttons', True):
            # divider = pt.ensure_divider(ax)
            butkw = {
                # 'divider': divider,
                'ax': ax,
                'size': '13%'
                # 'size': '15%'
            }
        # Chip matching/naming options
        nid = ibs.get_annot_name_rowids(aid)
        annotation_unknown = ibs.is_nid_unknown([nid])[0]
        if not annotation_unknown:
            # remove name
            callback = functools.partial(self.unname_annotation, aid)
            self.append_button(
                'remove name (' + ibs.get_name_texts(nid) + ')',
                callback=callback,
                **butkw,
            )
        else:
            # new name
            callback = functools.partial(self.mark_annotation_as_new_name, aid)
            self.append_button('mark as new name', callback=callback, **butkw)
        if (
            nid != self.nid2
            and not ibs.is_nid_unknown([self.nid2])[0]
            and not self.is_split_case
        ):
            # match to nid2
            callback = functools.partial(self.rename_annotation, aid, self.nid2)
            text = 'match to name2: ' + ibs.get_name_texts(self.nid2)
            self.append_button(text, callback=callback, **butkw)
        if nid != self.nid1 and not ibs.is_nid_unknown([self.nid1])[0]:
            # match to nid1
            callback = functools.partial(self.rename_annotation, aid, self.nid1)
            text = 'match to name1: ' + ibs.get_name_texts(self.nid1)
            self.append_button(text, callback=callback, **butkw)

        other_nid_list = self.get_other_nids()
        for other_nid in other_nid_list:
            if other_nid == nid:
                continue
            # rename nid2
            callback = functools.partial(self.rename_annotation, aid, other_nid)
            text = 'match to: ' + ibs.get_name_texts(other_nid)
            self.append_button(text, callback=callback, **butkw)
        return ax

    def show_hud(self):
        """ Creates heads up display

        button bar on bottom and title string

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.viz.interact.interact_name import *  # NOQA
            >>> # build test data
            >>> self = testsdata_match_verification('PZ_MTEST', 30, 32)
            >>> # execute function
            >>> result = self.show_hud()
            >>> # verify results
            >>> print(result)
            >>> ut.quit_if_noshow()
            >>> self.show_page()
            >>> pt.show_if_requested()
        """
        # Button positioners
        hl_slot, hr_slot = pt.make_bbox_positioners(
            y=0.02, w=0.15, h=0.063, xpad=0.02, startx=0, stopx=1
        )
        # hack make a second bbox positioner to get different sized buttons on #
        # the left
        hl_slot2, hr_slot2 = pt.make_bbox_positioners(
            y=0.02, w=0.08, h=0.05, xpad=0.015, startx=0, stopx=1
        )

        def next_rect(accum=[-1]):
            accum[0] += 1
            return hr_slot(accum[0])

        def next_rect2(accum=[-1]):
            accum[0] += 1
            return hl_slot2(accum[0])

        ibs = self.ibs
        name1, name2 = self.name1, self.name2
        nid1_is_known = not ibs.is_nid_unknown(self.nid1)
        nid2_is_known = not ibs.is_nid_unknown(self.nid2)
        all_nid_list = ibs.get_annot_name_rowids(self.all_aid_list)
        is_unknown = ibs.is_nid_unknown(all_nid_list)
        is_name1 = [nid == self.nid1 for nid in all_nid_list]
        is_name2 = [nid == self.nid2 for nid in all_nid_list]

        # option to remove all names only if at least one name exists
        if not all(is_unknown):
            unname_all_text = 'remove all names'
            self.append_button(
                unname_all_text, callback=self.unname_all, rect=next_rect()
            )
        # option to merge all into a new name if all are unknown
        if all(is_unknown) and not nid1_is_known and not nid2_is_known:
            joinnew_text = 'match all (nonjunk)\n to a new name'
            self.append_button(
                joinnew_text, callback=self.merge_nonjunk_into_new_name, rect=next_rect(),
            )
        # option dismiss all and give new names to all nonjunk images
        if any(is_unknown):
            self.append_button(
                'mark all unknowns\nas not matching',
                callback=self.dismiss_all,
                rect=next_rect(),
            )
        # merges all into the first name
        if nid1_is_known and not all(is_name1):
            join1_text = 'match all to name1:\n{name1}'.format(name1=name1)
            callback = functools.partial(self.merge_all_into_nid, self.nid1)
            self.append_button(join1_text, callback=callback, rect=next_rect())
        # merges all into the seoncd name
        if name1 != name2 and nid2_is_known and not all(is_name2):
            join2_text = 'match all to name2:\n{name2}'.format(name2=name2)
            callback = functools.partial(self.merge_all_into_nid, self.nid2)
            self.append_button(join2_text, callback=callback, rect=next_rect())
        ###
        self.append_button('close', callback=self.close_, rect=next_rect2())
        if self.qres_callback is not None:
            self.append_button('review', callback=self.review, rect=next_rect2())
        self.append_button('reset', callback=self.reset_all_names, rect=next_rect2())
        self.dbname = ibs.get_dbname()
        self.vsstr = 'qaid%d-vs-aid%d' % (self.aid1, self.aid2)
        figtitle_fmt = """
        Match Review Interface - {dbname}
        {match_text}:
        {vsstr}
        """
        figtitle = figtitle_fmt.format(**self.__dict__)  # sexy: using obj dict as fmtkw
        pt.set_figtitle(figtitle)

    def on_close(self, event=None):
        super(MatchVerificationInteraction, self).on_close(event)
        pass

    def unname_annotation(self, aid, event=None):
        if ut.VERBOSE:
            print('remove name')
        self.ibs.delete_annot_nids([aid])
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def mark_annotation_as_new_name(self, aid, event=None):
        if ut.VERBOSE:
            print('new name')
        self.ibs.set_annot_names_to_same_new_name([aid])
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def rename_annotation(self, aid, nid, event=None):
        if ut.VERBOSE:
            print('rename nid1')
        self.ibs.set_annot_name_rowids([aid], [nid])
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def reset_all_names(self, event=None):
        self.ibs.set_annot_name_rowids(self.all_aid_list, self.all_nid_list_orig)
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def review(self, event=None):
        if ut.VERBOSE:
            print('review pressed')
        if self.qres_callback is not None:
            self.qres_callback()
        else:
            print('Warning: no review callback connected.')

    def close_(self, event=None):
        # closing this gui with the button means you have reviewed the annotation.
        # self.ibs.set_annot_pair_as_reviewed(self.aid1, self.aid2)
        self.close()

    def unname_all(self, event=None):
        if ut.VERBOSE:
            print('unname_all')
        self.ibs.delete_annot_nids(self.all_aid_list)
        self.show_page()

    def merge_all_into_nid(self, nid, event=None):
        """ All the annotations are given nid """
        aid_list = self.all_aid_list
        self.ibs.set_annot_name_rowids(aid_list, [nid] * len(aid_list))
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def merge_nonjunk_into_new_name(self, event=None):
        """ All nonjunk annotations are given the SAME new name """
        # Delete all original names
        aid_list = self.all_aid_list
        aid_list_filtered = ut.filterfalse_items(
            aid_list, self.ibs.get_annot_isjunk(aid_list)
        )
        # Rename annotations
        self.ibs.set_annot_names_to_same_new_name(aid_list_filtered)
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def dismiss_all(self, event=None):
        """ All unknown annotations are given DIFFERENT new names """
        # Delete all original names
        ibs = self.ibs
        aid_list = self.all_aid_list
        is_unknown = ibs.is_aid_unknown(aid_list)
        aid_list_filtered = ut.compress(aid_list, is_unknown)
        # Rename annotations
        ibs.set_annot_names_to_different_new_names(aid_list_filtered)
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def on_key_press(self, event=None):
        if event.key == 'escape':
            from wbia import guitool

            if guitool.are_you_sure():
                self.close()

    def figure_clicked(self, event=None):
        ax = event.inaxes
        if ih.clicked_inside_axis(event):
            viztype = vh.get_ibsdat(ax, 'viztype')
            if viztype == 'chip':
                aid = vh.get_ibsdat(ax, 'aid')
                # print('... aid=%r' % aid)
                if event.button == 3:  # right-click
                    # import wbia.guitool
                    # height = self.fig.canvas.geometry().height()
                    # qpoint = guitool.newQPoint(event.x, height - event.y)
                    # ibs = self.ibs
                    # is_exemplar = ibs.get_annot_exemplar_flags(aid)
                    # def context_func():
                    #    ibs.set_annot_exemplar_flags(aid, not is_exemplar)
                    #    self.show_page()
                    # guitool.popup_menu(self.fig.canvas, pt, [
                    #    ('unset as exemplar' if is_exemplar else 'set as exemplar', context_func),
                    # ])
                    # TODO USE ABSTRACT INTERACTION
                    from wbia.viz.interact import interact_chip

                    options = interact_chip.build_annot_context_options(
                        self.ibs, aid, refresh_func=self.show_page
                    )
                    self.show_popup_menu(options, event)
                    # interact_chip.show_annot_context_menu(
                    #    self.ibs, aid, self.fig.canvas, qpoint, refresh_func=self.show_page)
                    # ibs.print_annotation_table()
                # print(ut.repr2(event.__dict__))
            elif viztype == 'matches':
                self.cm.ishow_single_annotmatch(self.qreq_, self.aid2, fnum=None, mode=0)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.interact.interact_name --test-ishow_name --show
        python -m wbia.viz.interact.interact_name --test-testsdata_match_verification --show --db PZ_MTEST --aid1 1 --aid2 30

        python -m wbia.viz.interact.interact_name
        python -m wbia.viz.interact.interact_name --allexamples
        python -m wbia.viz.interact.interact_name --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
