from __future__ import absolute_import, division, print_function
import utool as ut
from six.moves import zip
from plottool import interact_helpers as ih
from plottool import draw_func2 as df2
import plottool as pt
from ibeis import viz
from ibeis import constants as const
from ibeis.viz import viz_helpers as vh
from ibeis import ibsfuncs
from functools import partial
from guitool import guitool_dialogs
from ibeis.viz import viz_chip
from plottool.abstract_interaction import AbstractInteraction
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[interact_name]', DEBUG=False)


#==========================
# Name Interaction
#==========================


def ishow_name(ibs, nid, sel_aids=[], select_aid_callback=None, fnum=5, **kwargs):
    fig = ih.begin_interaction('name', fnum)

    def _on_name_click(event):
        print_('[inter] clicked name')
        ax = event.inaxes
        if ih.clicked_inside_axis(event):
            viztype = vh.get_ibsdat(ax, 'viztype')
            print_(' viztype=%r' % viztype)
            if viztype == 'chip':
                aid = vh.get_ibsdat(ax, 'aid')
                print('... aid=%r' % aid)
                viz.show_name(ibs, nid, fnum=fnum, sel_aids=[aid])
                if select_aid_callback is not None:
                    select_aid_callback(aid)
        viz.draw()

    viz.show_name(ibs, nid, fnum=fnum, sel_aids=sel_aids)
    viz.draw()
    ih.connect_callback(fig, 'button_press_event', _on_name_click)
    pass


def testsdata_match_verification():
    r"""
    CommandLine:
        main.py --eid 2
        main.py --eid 13 --db PZ_MUGU_19

    CommandLine:
        python -m ibeis.viz.interact.interact_name --test-testsdata_match_verification --show
        python -m ibeis.viz.interact.interact_name --test-testsdata_match_verification --show --db PZ_MUGU_19 --aid1 297 --aid2 267
        python -m ibeis.viz.interact.interact_name --test-testsdata_match_verification --show --db PZ_MUGU_19 --aid1 159 --aid2 154

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.viz.interact.interact_name import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> result = testsdata_match_verification()
        >>> # verify results
        >>> print(result)
    """
    from ibeis.viz.interact.interact_name import *  # NOQA
    import ibeis
    #ibs = ibeis.opendb(defaultdb='PZ_Master0')
    ibs = ibeis.opendb(defaultdb='testdb1')
    #aid1 = ut.get_argval('--aid1', int, 14)
    #aid2 = ut.get_argval('--aid2', int, 5545)
    aid1 = ut.get_argval('--aid1', int, 1)
    aid2 = ut.get_argval('--aid2', int, 2)
    self = MatchVerificationInteraction(ibs, aid1, aid2, dodraw=False)
    if ut.show_was_requested():
        self.show_page()
        pt.show_if_requested()


class MatchVerificationInteraction(AbstractInteraction):
    def __init__(self, ibs, aid1, aid2, update_callback=None,
                 backend_callback=None, dodraw=True, **kwargs):
        if ut.VERBOSE:
            print('[matchver] __init__')
        if ut.VERBOSE or ut.is_developer():
            print('[matchver] __init__ aid1=%r, aid2=%r ' % (aid1, aid2))
        super(MatchVerificationInteraction, self).__init__(**kwargs)
        self.ibs = ibs
        self.aid1 = aid1
        self.aid2 = aid2
        #ibsfuncs.assert_valid_aids(ibs, [aid1, aid2])
        if update_callback is None:
            update_callback = lambda: None
        if backend_callback is None:
            backend_callback = lambda: None
        self.update_callback = update_callback  # if something like qt needs a manual refresh on change
        self.backend_callback = backend_callback
        self.qres_callback = kwargs.get('qres_callback', None)
        self.infer_data()
        if dodraw:
            self.show_page(bring_to_front=True)

    def infer_data(self):
        """ Initialize data related to the input aids
        """
        ibs = self.ibs
        # The two matching aids
        (aid1, aid2) = (self.aid1, self.aid2)
        self.match_text = ibs.get_match_text(aid1, aid2)
        # The names of the matching annotations
        self.nid1, self.nid2 = ibs.get_annot_name_rowids((aid1, aid2))
        self.name1, self.name2 = ibs.get_annot_names((aid1, aid2))
        # The other annotations that belong to these two names
        self.gts_list  = ibs.get_annot_groundtruth((aid1, aid2), is_exemplar=None)
        #self.gts_list = [sorted(set(gt + [aid])) for gt, aid in zip(groundtruth_list, (aid1, aid2))]
        # A flat list of all the aids we are looking at
        self.aid_list = ut.unique_ordered(ut.flatten(self.gts_list))

        # Grab not just the exemplars
        # <HACK>
        all_groundtruth_list = ibs.get_annot_groundtruth((aid1, aid2))
        all_gt_list = [sorted(set(gt + [aid])) for gt, aid in zip(all_groundtruth_list, (aid1, aid2))]
        self.all_aid_list = ut.unique_ordered(ut.flatten(all_gt_list))
        # </HACK>

        # Original sets of groundtruth we are working with
        self.gt1, self.gt2 = self.gts_list
        # Grid that will fit all the names we need to display
        MAX_COLS = 3
        max_num_gt = max(map(len, all_gt_list))
        self.nCols = max_num_gt
        self.nCols = min(max_num_gt, MAX_COLS)
        self.nRows = len(all_gt_list)

        if ut.VERBOSE or ut.is_developer():
            print('[matchver] __init__ nid1=%r, nid2=%r ' % (self.nid1, self.nid2))
            print('[matchver] __init__ all_gt_list=%r ' % (all_gt_list))
            print('[matchver] __init__ self.gts_list=%r ' % (self.gts_list))

        if self.nid1 == self.nid2:
            self.nRows = 1
            self.gts_list = self.gts_list[0:1]  # remove redundant aids

        if ut.VERBOSE or ut.is_developer():
            print('[matchver] __init__ nid1=%r, nid2=%r ' % (self.nid1, self.nid2))
            print('[matchver] __init__ self.gts_list=%r ' % (self.gts_list))

    def prepare_page(self):
        figkw = {'fnum': self.fnum,
                 'doclf': True,
                 'docla': True, }
        self.fig = df2.figure(**figkw)
        ih.disconnect_callback(self.fig, 'button_press_event')
        ih.connect_callback(self.fig, 'button_press_event', self.figure_clicked)

    def show_page(self, bring_to_front=False):
        """ Plots all subaxes on a page """
        print('[matchver] show_page()')
        self.prepare_page()
        # Variables we will work with to paint a pretty picture
        ibs = self.ibs
        nRows = self.nRows
        nCols = self.nCols

        # Distinct color for every unique name
        nid_list = ibs.get_annot_name_rowids(self.all_aid_list)
        unique_nids = ut.unique_ordered(nid_list)
        unique_colors = df2.distinct_colors(len(unique_nids) + 2)
        self.nid2_color = dict(zip(unique_nids, unique_colors))

        if len(self.gts_list) == 2:
            list_ = list(map(ut.flatten, zip(([self.aid1], [self.aid2]), self.gts_list)))
        elif len(self.gts_list) == 1:
            # so hacky
            hack_aids_list = [_ for _ in self.gts_list[0] if _ not in [self.aid1, self.aid2]]
            list_ = [[self.aid1, self.aid2] + hack_aids_list]
        else:
            raise AssertionError('[interact_name] unknown hacked case')
        # For each row
        for rowx, aid_list in enumerate(list_):
            offset = rowx * nCols + 1
            #ibsfuncs.assert_valid_aids(ibs, groundtruth)
            # For each column
            for colx, aid in enumerate(aid_list):
                if colx >= self.nCols:
                    break
                try:
                    nid = ibs.get_annot_name_rowids(aid)
                    color = self.nid2_color[nid]
                except Exception as ex:
                    ut.printex(ex)
                    print('nid = %r' % (nid,))
                    print('self.nid2_color = %s' % (ut.dict_str(self.nid2_color),))
                    raise
                if ibsfuncs.is_nid_unknown(ibs, [nid])[0]:
                    color = const.UNKNOWN_PURPLE_RGBA01
                px = colx + offset
                #print('rowx=%r, colx=%r' % (rowx, colx))
                #print('offset=%rr' % (offset))
                ax = self.plot_chip(int(aid), nRows, nCols, px, color=color)
                if len(self.gts_list) == 2:
                    # OH MY GOD THE HACKYNESS. WHY SO STATEFUL?
                    if (colx + 1) >= self.nCols and colx < (len(aid_list) - 1):
                        next_text = 'next\n%d/%d' % (self.nCols - 1, len(aid_list) - 1)
                        next_func = partial(self.show_more, rowx=rowx)
                        self.append_button(next_text, callback=next_func,
                                           location='right', size='33%', ax=ax)
                elif len(self.gts_list) == 1:
                    # OH MY GOD THE HACKYNESS. WHY SO STATEFUL?
                    if (colx + 1) >= self.nCols and colx < (len(hack_aids_list) - 1):
                        next_text = 'next\n%d/%d' % (self.nCols - 2, len(hack_aids_list) - 1)
                        next_func = partial(self.show_more, rowx=rowx)
                        self.append_button(next_text, callback=next_func,
                                           location='right', size='33%', ax=ax)
                else:
                    raise AssertionError('[interact_name] unknown hacked case')

        self.show_hud()
        #df2.adjust_subplots_safe(top=0.85, hspace=0.03)
        df2.adjust_subplots_safe(top=0.85, hspace=0.05)
        self.draw()
        self.show()
        if bring_to_front:
            self.bring_to_front()
        #self.update()

    def plot_chip(self, aid, nRows, nCols, px, **kwargs):
        """ Plots an individual chip in a subaxis """
        #print('[plot_chip] %d %d %d' % (nRows, nCols, px))
        ibs = self.ibs
        nid = ibs.get_annot_name_rowids(aid)
        annotation_unknown = ibs.is_nid_unknown([nid])[0]
        if aid in [self.aid1, self.aid2]:
            lw = 5
            import numpy as np
            text_color = np.array((135, 206, 235, 255)) / 255.0
        else:
            lw = 2
            text_color = None
        viz_chip_kw = {
            'fnum': self.fnum,
            'pnum': (nRows, nCols, px),
            'nokpts': True,
            'show_name': True,
            'show_gname': False,
            'show_aidstr': True,
            'notitle': True,
            'show_num_gt': False,
            'text_color': text_color,
        }
        if ut.is_developer():
            enable_chip_title_prefix = True
            viz_chip_kw.update(
                {
                    'enable_chip_title_prefix': enable_chip_title_prefix,
                    'show_name': True,
                    'show_aidstr': True,
                    'show_yawtext': True,
                    'show_num_gt': True,
                    'show_quality_text': True,
                }
            )

        viz_chip.show_chip(ibs, aid, **viz_chip_kw)
        ax = df2.gca()
        df2.draw_border(ax, color=kwargs.get('color'), lw=lw)
        if kwargs.get('make_buttons', True):
            divider = df2.ensure_divider(ax)
            butkw = {
                'divider': divider,
                'size': '13%'
            }
        # Chip options
        flag = True
        if not annotation_unknown:
            callback = partial(self.unname_annotation, aid)
            self.append_button('unname', callback=callback, **butkw)
        if nid != self.nid1 and not ibs.is_nid_unknown([self.nid1])[0]:
            callback = partial(self.rename_annotation_nid1, aid)
            text = 'rename to: ' + ibs.get_name_texts(self.nid1)
            self.append_button(text, callback=callback, **butkw)
            flag = self.nid1 != self.nid2
        if nid != self.nid2 and not ibs.is_nid_unknown([self.nid2])[0] and flag:
            callback = partial(self.rename_annotation_nid2, aid)
            text = 'rename to: ' + ibs.get_name_texts(self.nid2)
            self.append_button(text, callback=callback, **butkw)
        return ax

    def unname_annotation(self, aid, event=None):
        print('remove name')
        self.ibs.delete_annot_nids([aid])
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def rename_annotation_nid1(self, aid, event=None):
        print('rename nid1')
        self.ibs.set_annot_name_rowids([aid], [self.nid1])
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def rename_annotation_nid2(self, aid, event=None):
        print('rename nid2')
        self.ibs.set_annot_name_rowids([aid], [self.nid2])
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def show_hud(self):
        """ Creates heads up display """
        # Button positioners
        hl_slot, hr_slot = df2.make_bbox_positioners(y=.02, w=.16,
                                                     h=3 * ut.PHI_B ** 4,
                                                     xpad=.02, startx=0, stopx=1)
        def next_rect(accum=[-1]):
            accum[0] += 1
            return hr_slot(accum[0])

        ibs = self.ibs
        name1, name2 = self.name1, self.name2
        nid_list = ibs.get_annot_name_rowids(self.all_aid_list)
        is_unknown = ibs.is_nid_unknown(nid_list)

        # option to remove all names only if at least one name exists
        if not all(is_unknown):
            self.append_button('remove all names', callback=self.unname_all,
                               rect=next_rect())

        # option to merge all into a new name if all are unknown
        if all(is_unknown):
            self.append_button('merge all\n into a NEW NAME',
                               callback=self.merge_all_into_next_name, rect=next_rect())

        # option dismiss all and give new names to all nonjunk images
        if any(is_unknown):
            self.append_button('dismiss all', callback=self.dismiss_all, rect=next_rect())

        # merges all into the first name
        if not name1.startswith('____'):
            self.append_button('join all\n into name1=%s' % name1, callback=self.merge_all_into_nid1, rect=next_rect())

        # merges all into the seoncd name
        if name1 != name2 and not name2.startswith('____') and not all([False]):
            self.append_button('join all\n into name2=%s' % name2,
                               callback=self.merge_all_into_nid2, rect=next_rect())

        ###
        #self.append_button('confirm', callback=self.confirm, rect=hl_slot(0))
        self.append_button('close', callback=self.close_, rect=hl_slot(0))
        self.append_button('review', callback=self.review, rect=hl_slot(1))
        self.vsstr = ibsfuncs.vsstr(self.aid1, self.aid2)
        figtitle_fmt = '''
        Match Review Interface
        {match_text}
        {vsstr}
        '''
        figtitle = figtitle_fmt.format(**self.__dict__)  # sexy: using obj dict as fmtkw
        df2.set_figtitle(figtitle)

    def review(self, event=None):
        if self.qres_callback is not None:
            self.qres_callback()
        print('review pressed')

    def close_(self, event=None):
        self.close()

    def confirm(self, event=None):
        print('confirm')
        ans = guitool_dialogs.user_option(parent=self.fig.canvas, msg='Are you sure?', title='Confirmation',
                                          options=['Confirm'], use_cache=False)
        print('ans = %r' % ans)
        #if ans == 'Confirm':
        #    #alrids_list = ibs.get_annot_alrids_oftype(self.aid_list, ibs.lbltype_ids[const.INDIVIDUAL_KEY], configid=ibs.MANUAL_CONFIGID)
        #    #alrid_list = ut.flatten(alrids_list)
        #    #ibs.set_alr_confidence(alrid_list, [1.0] * len(alrid_list))
        #    self.close()
        #    #print(ut.dict_str(locals()))
        #ibs.print_alr_table()
        self.infer_data()

    def unname_all(self, event=None):
        print('remove name')
        self.ibs.delete_annot_nids(self.all_aid_list)
        self.show_page()

    def merge_all_into_nid1(self, event=None):
        """ All the annotations are given nid1 """
        aid_list = self.all_aid_list
        self.ibs.set_annot_name_rowids(aid_list, [self.nid1] * len(aid_list))
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def merge_all_into_nid2(self, event=None):
        """ All the annotations are given nid2 """
        aid_list = self.all_aid_list
        self.ibs.set_annot_name_rowids(aid_list, [self.nid2] * len(aid_list))
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def merge_all_into_next_name(self, event=None):
        """ All nonjunk? annotations are given the SAME new name """
        # Delete all original names
        aid_list    = self.all_aid_list
        self.ibs.delete_annot_nids(aid_list)
        aid_list_filtered = ut.filterfalse_items(aid_list, self.ibs.get_annot_isjunk(aid_list))
        # Get next nagge from the controller
        next_name = ibsfuncs.make_next_name(self.ibs)
        # Readd the new names to all aids
        print('Setting aids=%r to have name=%r' % (aid_list_filtered, next_name))
        self.ibs.set_annot_names(aid_list_filtered, [next_name] * len(aid_list_filtered))
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def dismiss_all(self, event=None):
        """ All unknown nonjunk? annotations are given DIFFERENT new names """
        # Delete all original names
        ibs = self.ibs
        aid_list    = self.all_aid_list
        # ibs.delete_annot_nids(aid_list)
        # Get next name from the controller
        nid_list    = ibs.get_annot_name_rowids(aid_list)
        is_unknown  = ibsfuncs.is_nid_unknown(ibs, nid_list)
        aid_list_filtered = ut.filter_items(aid_list, is_unknown)
        #aid_list_filtered = ut.filterfalse_items(_aid_list_filtered, ibs.get_annot_isjunk(_aid_list_filtered))
        next_names = ibsfuncs.make_next_name(ibs, num=len(aid_list_filtered))
        # Readd the new names to all aids
        ibs.set_annot_names(aid_list_filtered, next_names)
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def show_more(self, event=None, rowx=None):
        def rotate_list(list_, n):
            """
            References:
                http://stackoverflow.com/questions/9457832/python-list-rotation
            """
            return list_[n:] + list_[:n]
        self.gts_list[rowx] = rotate_list(self.gts_list[rowx], self.nCols - 1)
        self.show_page()

    def figure_clicked(self, event=None):
        print_('[inter] clicked name')
        ax = event.inaxes
        if ih.clicked_inside_axis(event):
            viztype = vh.get_ibsdat(ax, 'viztype')
            print_(' viztype=%r' % viztype)
            if viztype == 'chip':
                aid = vh.get_ibsdat(ax, 'aid')
                print('... aid=%r' % aid)
                if event.button == 3:   # right-click
                    import guitool
                    height = self.fig.canvas.geometry().height()
                    qpoint = guitool.newQPoint(event.x, height - event.y)
                    #ibs = self.ibs
                    #is_exemplar = ibs.get_annot_exemplar_flags(aid)
                    #def context_func():
                    #    ibs.set_annot_exemplar_flags(aid, not is_exemplar)
                    #    self.show_page()
                    #guitool.popup_menu(self.fig.canvas, pt, [
                    #    ('unset as exemplar' if is_exemplar else 'set as exemplar', context_func),
                    #])
                    from ibeis.viz.interact import interact_chip
                    interact_chip.show_annot_context_menu(
                        self.ibs, aid, self.fig.canvas, qpoint, refresh_func=self.show_page)
                    #ibs.print_annotation_table()
                #print(ut.dict_str(event.__dict__))

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.interact.interact_name
        python -m ibeis.viz.interact.interact_name --allexamples
        python -m ibeis.viz.interact.interact_name --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
