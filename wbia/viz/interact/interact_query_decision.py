# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import utool as ut
from wbia.plottool import interact_helpers as ih
import wbia.plottool as pt
from functools import partial
from wbia.viz import viz_chip
from wbia.viz import viz_matches
from wbia.plottool.abstract_interaction import AbstractInteraction

ut.noinject(__name__, '[interact_query_decision]')


# ==========================
# query interaction
# ==========================

NUM_TOP = 3


class QueryVerificationInteraction(AbstractInteraction):
    """
    CommandLine:
        python -m wbia.viz.interact.interact_query_decision --test-QueryVerificationInteraction --show
        python -m wbia --imgsetid 2 --inc-query --yes

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.interact.interact_query_decision import *  # NOQA
        >>> import wbia
        >>> cm, qreq_ = wbia.testdata_cm()
        >>> comp_aids = cm.get_top_aids(NUM_TOP)
        >>> suggest_aids = comp_aids[0:1]
        >>> qvi = QueryVerificationInteraction(
        >>>     qreq_, cm, comp_aids, suggest_aids, progress_current=42, progress_total=1337)
        >>> ut.show_if_requested()
    """

    def __init__(
        self,
        qreq_,
        cm,
        comp_aids,
        suggest_aids,
        progress_current=None,
        progress_total=None,
        update_callback=None,
        backend_callback=None,
        name_decision_callback=None,
        **kwargs,
    ):
        print('[matchver] __init__')
        super(QueryVerificationInteraction, self).__init__(**kwargs)
        print('[matchver] comp_aids=%r' % (comp_aids,))
        print('[matchver] suggest_aids=%r' % (suggest_aids,))
        self.ibs = qreq_.ibs
        self.qreq_ = qreq_
        self.cm = cm
        self.query_aid = self.cm.qaid
        self.ibs.assert_valid_aids(comp_aids, verbose=True)
        self.ibs.assert_valid_aids(suggest_aids, verbose=True)
        self.ibs.assert_valid_aids((self.query_aid,), verbose=True)
        assert len(comp_aids) <= NUM_TOP
        self.comp_aids = comp_aids
        self.suggest_aids = suggest_aids
        self.suggest_aids = None  # HACK TO TURN OFF SUGGESTIONS
        self.progress_current = progress_current
        self.progress_total = progress_total

        def _nonefn():
            return None

        def _nonefn2(*args):
            return None

        if update_callback is None:
            update_callback = _nonefn
        if backend_callback is None:
            backend_callback = _nonefn
        if name_decision_callback is None:
            name_decision_callback = _nonefn2
        self.update_callback = (
            update_callback  # if something like qt needs a manual refresh on change
        )
        self.backend_callback = backend_callback
        self.name_decision_callback = name_decision_callback
        self.aid_checkbox_states = {}
        self.other_checkbox_states = {'none': True, 'junk': False}
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

        # HACK: make sure that comp_aids is of length NUM_TOP
        if len(self.comp_aids) != NUM_TOP:
            self.comp_aids += [None for i in range(NUM_TOP - len(self.comp_aids))]

        # column for each comparasion + the none button
        # row for the query, row for the comparasions
        self.nCols = len(self.comp_aids)
        self.nRows = 2

    def prepare_page(self):
        figkw = {
            'fnum': self.fnum,
            'doclf': True,
            'docla': True,
        }
        self.fig = pt.figure(**figkw)
        ih.disconnect_callback(self.fig, 'button_press_event')
        ih.connect_callback(self.fig, 'button_press_event', self.figure_clicked)
        # ih.connect_callback(self.fig, 'button_press_event', self.figure_clicked)

    def show_page(self, bring_to_front=False):
        """ Plots all subaxes on a page """
        print('[querydec] show_page()')
        self.prepare_page()
        # Variables we will work with to paint a pretty picture
        # ibs = self.ibs
        nRows = self.nRows
        nCols = self.nCols

        # Plot the Comparisions
        for count, c_aid in enumerate(self.comp_aids):
            if c_aid is not None:
                px = nCols + count + 1
                title_suffix = ''
                if self.suggest_aids is not None and c_aid in self.suggest_aids:
                    title_suffix = 'SUGGESTED BY IBEIS'
                self.plot_chip(c_aid, nRows, nCols, px, title_suffix=title_suffix)
            else:
                pt.imshow_null(
                    fnum=self.fnum,
                    pnum=(nRows, nCols, nCols + count + 1),
                    title='NO RESULT',
                )

        # Plot the Query Chip last
        with ut.EmbedOnException():
            query_title = 'Identify This Animal'
            self.plot_chip(self.query_aid, nRows, 1, 1, title_suffix=query_title)

        self.show_hud()
        pt.adjust_subplots(
            top=0.88, hspace=0.12, left=0.1, right=0.9, bottom=0.1, wspace=0.3
        )
        self.draw()
        self.show()
        if bring_to_front:
            self.bring_to_front()

    def plot_chip(self, aid, nRows, nCols, px, **kwargs):
        """ Plots an individual chip in a subaxis """
        ibs = self.ibs
        enable_chip_title_prefix = ut.is_developer()
        # enable_chip_title_prefix = False
        if aid in self.comp_aids:
            score = self.cm.get_annot_scores([aid])[0]
            rawscore = self.cm.get_annot_scores([aid])[0]
            title_suf = kwargs.get('title_suffix', '')
            if score != rawscore:
                if score is None:
                    title_suf += '\n score=____'
                else:
                    title_suf += '\n score=%0.2f' % score
            title_suf += '\n rawscore=%0.2f' % rawscore
        else:
            title_suf = kwargs.get('title_suffix', '')
            if enable_chip_title_prefix:
                title_suf = '\n' + title_suf

        # nid = ibs.get_annot_name_rowids(aid)
        viz_chip_kw = {
            'fnum': self.fnum,
            'pnum': (nRows, nCols, px),
            'nokpts': True,
            'show_gname': False,
            'show_exemplar': False,
            'show_num_gt': False,
            'show_gname': False,
            'title_suffix': title_suf,
            # 'text_color': kwargs.get('color'),
            ###
            # 'show_name': False,
            # 'show_aidstr': False,
            'enable_chip_title_prefix': enable_chip_title_prefix,
            'show_name': True,
            'show_aidstr': True,
            'show_viewcode': True,
            'show_quality_text': True,
        }

        viz_chip.show_chip(ibs, aid, **viz_chip_kw)
        ax = pt.gca()
        if kwargs.get('make_buttons', True):
            divider = pt.ensure_divider(ax)
            butkw = {'divider': divider, 'size': '13%'}

        self.aid2_ax = {}
        self.aid2_border = {}

        if aid in self.comp_aids:
            callback = partial(self.select, aid)
            self.append_button('Select This Animal', callback=callback, **butkw)
            # Hack to toggle colors
            if aid in self.aid_checkbox_states:
                # If we are selecting it, then make it green, otherwise change it back to grey
                if self.aid_checkbox_states[aid]:
                    border = pt.draw_border(ax, color=(0, 1, 0), lw=4)
                else:
                    border = pt.draw_border(ax, color=(0.7, 0.7, 0.7), lw=4)
                self.aid2_border[aid] = border
            else:
                self.aid_checkbox_states[aid] = False
            self.append_button('Examine', callback=partial(self.examine, aid), **butkw)

    def examine(self, aid, event=None):
        print(' examining aid %r against the query result' % aid)
        figtitle = 'Examine a specific image against the query'

        # fnum = 510
        fnum = pt.next_fnum()
        fig = pt.figure(fnum=fnum, pnum=(1, 1, 1), doclf=True, docla=True)
        # can cause freezes should be False
        INTERACT_EXAMINE = False
        if INTERACT_EXAMINE:
            # from wbia.viz.interact import interact_matches
            # fig = interact_matches.ishow_matches(self.ibs, self.cm, aid, figtitle=figtitle, fnum=fnum)
            fig = self.cm.ishow_matches(self.ibs, aid, figtitle=figtitle, fnum=fnum)
            print('Finished interact')
            # this is only relevant to matplotlib.__version__ < 1.4.2
            # raise Exception(
            #    'BLACK MAGIC: error intentionally included as a workaround that seems'
            #    'to fix a gui hang on certain computers.')
        else:
            viz_matches.show_matches(self.ibs, self.cm, aid, figtitle=figtitle)
            fig.show()

    def select(self, aid, event=None):
        print(' selected aid %r as best choice' % aid)
        state = self.aid_checkbox_states[aid]
        self.aid_checkbox_states[aid] = not state
        for key in self.other_checkbox_states:
            self.other_checkbox_states[key] = False
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def select_none(self, event=None):
        for aid in self.comp_aids:
            self.aid_checkbox_states[aid] = False
        self.other_checkbox_states['none'] = True
        self.other_checkbox_states['junk'] = False
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def select_junk(self, event=None):
        for aid in self.comp_aids:
            self.aid_checkbox_states[aid] = False
        self.other_checkbox_states['none'] = False
        self.other_checkbox_states['junk'] = True
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def quit(self, event=None):
        self.close()

    def show_hud(self):
        """ Creates heads up display """
        # Button positioners
        hl_slot, hr_slot = pt.make_bbox_positioners(
            y=0.02, w=0.16, h=3 * ut.PHI_B ** 4, xpad=0.05, startx=0, stopx=1
        )

        select_none_text = 'None of these'
        if self.suggest_aids is not None and len(self.suggest_aids) == 0:
            select_none_text += '\n(SUGGESTED BY IBEIS)'
        none_tup = self.append_button(
            select_none_text, callback=partial(self.select_none), rect=hl_slot(0)
        )
        # Draw boarder around the None of these button
        none_button_axis = none_tup[1]
        if self.other_checkbox_states['none']:
            pt.draw_border(none_button_axis, color=(0, 1, 0), lw=4, adjust=False)
        else:
            pt.draw_border(none_button_axis, color=(0.7, 0.7, 0.7), lw=4, adjust=False)

        select_junk_text = 'Junk Query Image'
        junk_tup = self.append_button(
            select_junk_text, callback=partial(self.select_junk), rect=hl_slot(1)
        )
        # Draw boarder around the None of these button
        junk_button_axis = junk_tup[1]
        if self.other_checkbox_states['junk']:
            pt.draw_border(junk_button_axis, color=(0, 1, 0), lw=4, adjust=False)
        else:
            pt.draw_border(junk_button_axis, color=(0.7, 0.7, 0.7), lw=4, adjust=False)

        # Add other HUD buttons
        self.append_button('Quit', callback=partial(self.quit), rect=hr_slot(0))
        self.append_button(
            'Confirm Selection', callback=partial(self.confirm), rect=hr_slot(1)
        )

        if self.progress_current is not None and self.progress_total is not None:
            self.progress_string = (
                str(self.progress_current) + '/' + str(self.progress_total)
            )
        else:
            self.progress_string = ''
        figtitle_fmt = """
        Animal Identification {progress_string}
        """
        figtitle = figtitle_fmt.format(**self.__dict__)  # sexy: using obj dict as fmtkw
        pt.set_figtitle(figtitle)

    def confirm(self, event=None):
        """

        CommandLine:
            python -m wbia.viz.interact.interact_query_decision --test-confirm

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.viz.interact.interact_query_decision import *  # NOQA
            >>> import utool as ut
            >>> # build test data
            >>> import wbia
            >>> ibs = wbia.opendb('testdb1')
            >>> self = ibs
            >>> self.ibs = ibs
            >>> selected_aids = ut.get_list_column(ibs.get_name_aids(ibs.get_valid_nids()), 0)
            >>> comfirm_res = 'jeff'
            >>> # execute function
            >>> #result = self.confirm(event)
            >>> # verify results
            >>> #print(result)
        """
        import wbia.guitool as gt

        print('[interact_query_decision] Confirming selected animals.')

        selected_aids = [
            aid
            for aid in self.comp_aids
            if aid is not None and self.aid_checkbox_states[aid]
        ]
        if len(selected_aids) == 0:
            print('[interact_query_decision] Confirming no match.')
            chosen_aids = []
            if self.other_checkbox_states['none']:
                chosen_aids = 'newname'
            elif self.other_checkbox_states['junk']:
                chosen_aids = 'junk'
            else:
                msg = 'INTERACT_QUERY_DECISION IMPOSSIBLE STATE'
                raise AssertionError(msg)
        elif len(selected_aids) == 1:
            print('[interact_query_decision] Confirming single match')
            chosen_aids = selected_aids
        else:
            print('[interact_query_decision] Confirming merge')
            msg = ut.textblock(
                """
                You have selected more than one animal as a match to the query
                animal.  By doing this you are telling IBEIS that these are ALL
                the SAME ANIMAL.  \n\n\nIf this is not what you want, click
                Cancel.  If it is what you want, choose one of the names below
                as the name to keep.
                """
            )
            selected_names = self.ibs.get_annot_names(selected_aids)
            options = selected_names
            parent = None
            title = 'Confirm Merge'
            merge_name = gt.user_option(parent, msg=msg, title=title, options=options)
            if merge_name is None:
                print('[interact_query_decision] cancelled merge')
                self.update_callback()
                self.backend_callback()
                self.show_page()
                return
            else:
                print('[interact_query_decision] confirmed merge')
                is_merge_name = [merge_name == name_ for name_ in selected_names]
                chosen_aids = ut.sortedby(selected_aids, is_merge_name)[::-1]

        print('[interact_query_decision] Calling update callbacks')
        self.update_callback()
        self.backend_callback()
        print('[interact_query_decision] Calling decision callback')
        print(
            '[interact_query_decision] self.name_decision_callback = %r'
            % (self.name_decision_callback,)
        )
        if isinstance(chosen_aids, six.string_types):
            # hack for string non-match commands
            chosen_names = chosen_aids
        else:
            chosen_names = self.ibs.get_annot_names(chosen_aids)
        self.name_decision_callback(chosen_names)
        print(
            '[interact_query_decision] sent name_decision_callback(chosen_names=%r)'
            % (chosen_names,)
        )

    def figure_clicked(self, event=None):
        from wbia.viz import viz_helpers as vh
        import wbia.guitool as gt

        ax = event.inaxes
        if ih.clicked_inside_axis(event):
            viztype = vh.get_ibsdat(ax, 'viztype')
            if viztype == 'chip':
                aid = vh.get_ibsdat(ax, 'aid')
                print('... aid=%r' % aid)
                if event.button == 3:  # right-click
                    from wbia.viz.interact import interact_chip

                    height = self.fig.canvas.geometry().height()
                    qpoint = gt.newQPoint(event.x, height - event.y)
                    if self.qreq_ is None:
                        config2_ = None
                    else:
                        if aid in self.qreq_.qaids:
                            config2_ = self.qreq_.query_config2_
                        else:
                            config2_ = self.qreq_.data_config2_
                    callback_list = interact_chip.build_annot_context_options(
                        self.ibs, aid, refresh_func=self.show_page, config2_=config2_
                    )
                    gt.popup_menu(self.fig.canvas, qpoint, callback_list)
                    # interact_chip.show_annot_context_menu(
                    #    self.ibs, aid, self.fig.canvas, qpoint, refresh_func=self.show_page)
                    # self.show_page()
                    # ibs.print_annotation_table()
                print(ut.repr2(event.__dict__))


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.interact.interact_query_decision
        python -m wbia.viz.interact.interact_query_decision --allexamples
        python -m wbia.viz.interact.interact_query_decision --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
