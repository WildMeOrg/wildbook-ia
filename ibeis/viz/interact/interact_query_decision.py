from __future__ import absolute_import, division, print_function
import utool
import utool as ut
import ibeis
from plottool import interact_helpers as ih
from plottool import draw_func2 as df2
from ibeis.viz.interact import interact_matches as im
# from ibeis.gui import guiback
from functools import partial
import guitool
from ibeis.viz import viz_chip
from plottool.abstract_interaction import AbstractInteraction
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_name]', DEBUG=False)


#==========================
# query interaction
#==========================

def test_QueryVerificationInteraction():
    """
    CommandLine:
        python -m ibeis.viz.interact.interact_query_decision --test-test_QueryVerificationInteraction

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.viz.interact.interact_query_decision import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> result = test_QueryVerificationInteraction()
        >>> # verify results
        >>> print(result)
    """
    ibs = ibeis.opendb('testdb1')
    valid_aids = ibs.get_valid_aids()
    qaids = valid_aids[0:1]
    daids = valid_aids[1:]
    qres = ibs.query_chips(qaids, daids)[0]
    comp_aids = qres.get_top_aids(ibs=ibs, name_scoring=True)[0:3]
    suggest_aid = comp_aids[1]
    qvi = QueryVerificationInteraction(ibs, qres, comp_aids, suggest_aid)
    qvi.fig.show()
    exec(df2.present())


class QueryVerificationInteraction(AbstractInteraction):
    def __init__(self, ibs, qres, comp_aids, suggest_aid, update_callback=None,
                 backend_callback=None, decision_callback=None, **kwargs):
        print('[matchver] __init__')
        super(QueryVerificationInteraction, self).__init__(**kwargs)
        self.ibs = ibs
        self.qres = qres
        self.query_aid = self.qres.get_qaid()
        assert(len(comp_aids) <= 3)
        self.comp_aids = comp_aids
        self.suggest_aid = suggest_aid
        if update_callback is None:
            update_callback = lambda: None
        if backend_callback is None:
            backend_callback = lambda: None
        if decision_callback is None:
            decision_callback = lambda aids: None
        self.update_callback = update_callback  # if something like qt needs a manual refresh on change
        self.backend_callback = backend_callback
        self.decision_callback = decision_callback
        self.checkbox_states = {}
        self.qres_callback = kwargs.get('qres_callback', None)
        self.infer_data()
        self.show_page(bring_to_front=True)

    def infer_data(self):
        """ Initialize data related to the input aids """
        ibs = self.ibs

        self.query_nid = ibs.get_annot_name_rowids(self.query_aid)
        self.comp_nids = ibs.get_annot_name_rowids(self.comp_aids)
        self.query_name = ibs.get_annot_names(self.query_aid)
        self.comp_names = ibs.get_annot_names(self.comp_aids)

        self.aid_list = [self.query_aid] + self.comp_aids

        # qres = ibs.query_chips(query_aid,)

        #HACK: make sure that comp_aids is of length 3
        if len(self.comp_aids) != 3:
            self.comp_aids += [None for i in range(3 - len(self.comp_aids))]

        #column for each comparasion + the none button
        #row for the query, row for the comparasions
        self.nCols = len(self.comp_aids)
        self.nRows = 2

    def prepare_page(self):
        figkw = {
            'fnum': self.fnum,
            'doclf': True,
            'docla': True,
        }
        self.fig = df2.figure(**figkw)
        ih.disconnect_callback(self.fig, 'button_press_event')
        # ih.connect_callback(self.fig, 'button_press_event', self.figure_clicked)

    # def figure_clicked(self, event=None):
    #     print_('[inter] clicked name')
    #     ax = event.inaxes
    #     if ih.clicked_inside_axis(event):
    #         viztype = vh.get_ibsdat(ax, 'viztype')
    #         print_(' viztype=%r' % viztype)
    #         if viztype == 'chip':
    #             aid = vh.get_ibsdat(ax, 'aid')
    #             print('... aid=%r' % aid)
    #             if event.button == 3:   # right-click
    #                 import guitool
    #                 ibs = self.ibs
    #                 is_exemplar = ibs.get_annot_exemplar_flags(aid)
    #                 def context_func():
    #                     ibs.set_annot_exemplar_flags(aid, not is_exemplar)
    #                     self.show_page()
    #                 guitool.popup_menu(self.fig.canvas, guitool.newQPoint(event.x, event.y), [
    #                     ('unset as exemplar' if is_exemplar else 'set as exemplar', context_func),
    #                 ])
    #                 #ibs.print_annotation_table()
    #             print(utool.dict_str(event.__dict__))

    def show_page(self, bring_to_front=False):
        """ Plots all subaxes on a page """
        print('[querydec] show_page()')
        self.prepare_page()
        # Variables we will work with to paint a pretty picture
        #ibs = self.ibs
        nRows = self.nRows
        nCols = self.nCols

        #Plot the Comparisions
        for count, c_aid in enumerate(self.comp_aids):
            if c_aid is not None:
                if self.suggest_aid == c_aid:
                    self.plot_chip(int(c_aid), nRows, nCols, nCols + count + 1, title_suffix="SUGGESTED BY IBEIS")
                else:
                    self.plot_chip(int(c_aid), nRows, nCols, nCols + count + 1)
            else:
                df2.imshow_null(fnum=self.fnum, pnum=(nRows, nCols, nCols + count + 1))

        #Plot the Query Chip last
        self.plot_chip(int(self.query_aid), nRows, 1, 1, title_suffix="QUERY RESULT")

        self.show_hud()
        # df2.adjust_subplots_safe(top=0.85, hspace=0.03)
        self.draw()
        self.show()
        if bring_to_front:
            self.bring_to_front()
        #self.update()

    def plot_chip(self, aid, nRows, nCols, px, **kwargs):
        """ Plots an individual chip in a subaxis """
        ibs = self.ibs
        #nid = ibs.get_annot_name_rowids(aid)
        viz_chip_kw = {
            'fnum': self.fnum,
            'pnum': (nRows, nCols, px),
            'nokpts': True,
            'show_name': True,
            'show_gname': False,
            'show_aidstr': True,
            'show_exemplar': False,
            'show_num_gt': False,
            'show_gname': False,
            'enable_chip_title_prefix': False,
            'title_suffix': kwargs.get('title_suffix', ""),
            # 'text_color': kwargs.get('color'),
        }

        viz_chip.show_chip(ibs, aid, **viz_chip_kw)
        ax = df2.gca()
        if kwargs.get('make_buttons', True):
            divider = df2.ensure_divider(ax)
            butkw = {
                'divider': divider,
                'size': '13%'
            }

        if aid in self.comp_aids:
            callback = partial(self.select, aid)
            self.append_button('Select This Image', callback=callback, **butkw)
            #Hack to toggle colors
            if aid in self.checkbox_states:
                #If we are selecting it, then make it green, otherwise change it back to grey
                button = self.checkbox_states[aid]  # NOQA
                if self.checkbox_states[aid]:
                    df2.draw_border(ax, color=(0, 1, 0), lw=4)
                else:
                    df2.draw_border(ax, color=(.7, .7, .7), lw=4)
            else:
                self.checkbox_states[aid] = False
            self.append_button('Examine', callback=partial(self.examine, aid), **butkw)

    def select(self, aid, event=None):
        print(' selected aid %r as best choice' % aid)
        state = self.checkbox_states[aid]
        self.checkbox_states[aid] = not state

        self.update_callback()
        self.backend_callback()
        self.show_page()

    def examine(self, aid, event=None):
        print(' examining aid %r against the query result' % aid)
        im.ishow_matches(self.ibs, self.qres, aid,
                         figtitle='Examine a specific image against the query')

        self.update_callback()
        self.backend_callback()
        self.show_page()

    def show_hud(self):
        """ Creates heads up display """
        # Button positioners
        hl_slot, hr_slot = df2.make_bbox_positioners(y=.02, w=.16,
                                                     h=3 * utool.PHI_B ** 4,
                                                     xpad=.02, startx=0, stopx=1)

        self.append_button('None of these', callback=partial(self.select, None), rect=hl_slot(0))
        self.append_button('Confirm Selection', callback=partial(self.confirm), rect=hl_slot(1))
        figtitle_fmt = '''
        Query Decision Interface
        '''
        figtitle = figtitle_fmt.format(**self.__dict__)  # sexy: using obj dict as fmtkw
        df2.set_figtitle(figtitle)

    def confirm(self, event=None):
        """

        CommandLine:
            python -m ibeis.viz.interact.interact_query_decision --test-confirm

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.viz.interact.interact_query_decision import *  # NOQA
            >>> import utool as ut
            >>> # build test data
            >>> import ibeis
            >>> ibs = ibeis.opendb('testdb1')
            >>> self = ibs
            >>> self.ibs = ibs
            >>> selected_aids = ut.get_list_column(ibs.get_name_aids(ibs.get_valid_nids()), 0)
            >>> comfirm_res = 'jeff'
            >>> # execute function
            >>> result = self.confirm(event)
            >>> # verify results
            >>> print(result)
        """
        print('[interact_query_decision] Confirming selected animals.')

        selected_aids = [aid for aid in self.comp_aids
                         if aid is not None and self.checkbox_states[aid]]
        if len(selected_aids) > 1:
            print('[interact_query_decision] Confirming merge')
            msg = (
                'You have selected more than one animal as a match to the query'
                'animal. This will merge ALL aforementioned animals into the'
                'selected name. Please ensure that this is your intention, and'
                'select desired name for the merge.')
            selected_names = self.ibs.get_annot_names(selected_aids)
            options = selected_names
            comfirm_res = guitool.user_option(None, msg=msg,
                                              title='Confirmation',
                                              options=options)
            if comfirm_res is None:
                self.update_callback()
                self.backend_callback()
                self.show_page()
                return
            else:
                merge_name  = comfirm_res
                is_merge_name = [name == merge_name for name in selected_names]
                sorted_aids = ut.sortedby(selected_aids, is_merge_name)[::-1]
        else:
            print('[interact_query_decision] Confirming single match')
            sorted_aids = selected_aids

        print('[interact_query_decision] Calling update callbacks')
        self.update_callback()
        self.backend_callback()
        print('[interact_query_decision] Calling decision callback')
        print('[interact_query_decision] self.decision_callback = %r' % (self.decision_callback,))
        self.decision_callback(sorted_aids)
        print('[interact_query_decision] sent callback')
        #self.show_page()
        self.close()
        print('[interact_query_decision] Finished confirm')


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.interact.interact_query_decision
        python -m ibeis.viz.interact.interact_query_decision --allexamples
        python -m ibeis.viz.interact.interact_query_decision --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
