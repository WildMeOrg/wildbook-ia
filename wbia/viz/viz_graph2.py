# -*- coding: utf-8 -*-
"""
CommandLine:
    wbia make_qt_graph_interface --show
    wbia make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import vtool as vt
import numpy as np
from wbia import dtool
import networkx as nx
import itertools as it
import wbia.guitool as gt
import wbia.plottool as pt
import wbia.constants as const
from wbia.plottool import abstract_interaction
from wbia.guitool.__PYQT__ import QtCore
from wbia.guitool.__PYQT__.QtCore import Qt
from wbia.guitool import mpl_widget
from wbia.guitool import PrefWidget2
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA

(print, rrr, profile) = ut.inject2(__name__)


GRAPH_REVIEW_CFG_DEFAULTS = {
    'ranks_top': 3,
    'ranks_bot': 2,
    'hack_min_review': False,
    'filter_reviewed': True,
    'filter_photobombs': False,
    'filter_true_matches': True,
    'filter_false_matches': False,
    'filter_nonmatch_between_ccs': True,
    'filter_dup_namepairs': True,
    'show_match_thumb': True,
}


class AnnotPairDialog(gt.GuitoolWidget):
    r"""
    wbia AnnotPairDialog --show
    python -m wbia.algo.graph.mixin_loops qt_review_loop --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_graph2 import *  # NOQA
        >>> import wbia.guitool as gt
        >>> gt.ensure_qapp()
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> win = AnnotPairDialog(ibs=ibs, edge=(1, 2),
        >>>                       info_text='text describing this match')
        >>> gt.qtapp_loop(qwin=win, freq=10)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_graph2 import *  # NOQA
        >>> import wbia
        >>> import wbia.guitool as gt
        >>> gt.ensure_qapp()
        >>> infr = wbia.AnnotInference('PZ_Master1', 'all')
        >>> infr.reset_feedback('staging')
        >>> edges = [(86, 16273), (86, 5245), (92, 16273), (559, 16240),
        >>>         (559, 2111),]
        >>> edge = edges[0]
        >>> self = AnnotPairDialog(edge=edges, infr=infr, standalone=False)
        >>> self.seek(0)
        >>> self.show()


    """

    # skipped = QtCore.pyqtSignal()
    # request = QtCore.pyqtSignal(tuple)
    def initialize(
        self,
        edge=None,
        infr=None,
        ibs=None,
        info_text=None,
        cfgdict=None,
        standalone=True,
    ):
        """
        # Args:
        #     standalone (bool):

        """
        from wbia.gui import inspect_gui

        self.infr = infr

        self.history = ut.oset()

        if infr is not None:
            ibs = infr.ibs

        self.standalone = standalone

        self.annot_state1 = AnnotStateDialog(ibs=ibs)
        self.annot_state2 = AnnotStateDialog(ibs=ibs)

        self.annot_review = EdgeReviewDialog(
            # conf_editor='combo',
            with_confirm=False
        )

        self.tuner = inspect_gui.make_vsone_tuner(
            ibs, autoupdate=True, cfgdict=cfgdict, info_text=info_text
        )

        splitter = self.addNewSplitter(ori='horiz')
        splitter.addWidget(self.tuner)

        rbox = splitter.addNewWidget(ori='vert')

        rbox.setStyleSheet(
            """
            .QFrame {
                border-width: 2px;
                border-color: black;
                border-style: outset;
            }
            """
        )

        gt.set_qt_object_names(vars(self))

        self.setMinimumHeight(1)
        self.setMinimumWidth(1)

        if False:
            self.spoiler1 = gt.Spoiler(self, title='Aid1')
            self.spoiler2 = gt.Spoiler(self, title='Aid2')
            rbox.addWidget(self.spoiler1)
            rbox.addWidget(self.spoiler2)

            self.spoiler1.setContentLayout(self.annot_state1)
            self.spoiler2.setContentLayout(self.annot_state2)
        else:
            frame1 = rbox.addNewFrame()
            frame2 = rbox.addNewFrame()

            frame1.addWidget(self.annot_state1)
            frame2.addWidget(self.annot_state2)

        frame3 = rbox.addNewFrame()
        frame3.addWidget(self.annot_review)
        rbox.addNewSpacer(hPolicy='Expanding', vPolicy='Expanding')
        # gt.print_widget_heirarchy(
        #     self, annot_state=['sizePolicy', 'minimumHeight'], skip=True)

        butbar = rbox.addNewHWidget()
        self.accept_button = butbar.addNewButton('Accept', pressed=self.accept)
        if not standalone:
            self.skip_but = butbar.addNewButton('Skip', pressed=self.skip)

        np_bar = rbox.addNewHWidget()
        self.count = None
        self._total = None
        self.was_confirmed = False

        self.index_edit = None

        edges = None
        if edge is not None:
            if ut.isiterable(edge) and len(edge) > 0 and ut.isiterable(edge[0]):
                edges = edge
                edge = None

        if edges is not None:
            self.edges = edges
            for e in edges:
                self.history.add(e)
            self.count = 0
            self._total = len(edges)
            self.prev_but = np_bar.addNewButton('Prev', pressed=lambda: self.step_by(-1))
            self.index_edit = np_bar.addNewLineEdit(
                str(self.count + 1), editingFinishedSlot=self.edit_jump
            )
            self.next_but = np_bar.addNewButton('Next', pressed=lambda: self.step_by(1))
        elif not self.standalone:
            self.count = 0
            self.prev_but = np_bar.addNewButton('Prev', pressed=lambda: self.step_by(-1))
            self.index_edit = np_bar.addNewLineEdit(
                str(self.count + 1), editingFinishedSlot=self.edit_jump
            )
            self.next_but = np_bar.addNewButton('Next', pressed=lambda: self.step_by(1))
            self.next_but.setEnabled(False)

        if self.index_edit:
            self.index_edit.setText('{} / {}'.format(self.count + 1, self.total))

        self.last_external = True

        print('edge = %r' % (edge,))
        print('self.total = %r' % (self.total,))
        print('self.standalone = %r' % (self.standalone,))
        if edge is not None:
            self.set_edge(edge, info_text)
        elif self.total > 0 and not self.standalone:
            self.seek(0)
        else:
            if self.infr._gen is None:
                self.infr.start_id_review()
            self.continue_review()

    @property
    def total(self):
        if not self.standalone:
            return len(self.history)
        else:
            return self._total

    def keyPressEvent(self, event):
        if event.key() == gt.__PYQT__.QtCore.Qt.Key_Return:
            self.accept()
        else:
            return self.annot_review.keyPressEvent(event)

    def feedback_dict(self):
        feedback = self.annot_review.feedback_dict()
        feedback['annot1_state'] = self.annot_state1.current_annot_state()
        feedback['annot2_state'] = self.annot_state2.current_annot_state()
        return feedback

    def skip(self):
        edge = self.annot_review.edge
        if self.infr:
            self.infr.skip(edge)
            self.continue_review()

    def accept(self):
        print('[viz] accept')
        self.was_confirmed = True
        feedback = self.feedback_dict()
        if self.standalone:
            print('feedback = %s' % (ut.repr4(feedback),))
            if self.infr is not None:
                self.infr.accept(feedback)
            else:
                print('Edge feedback not recoreded')
            self.goto_next()
        else:
            need_next = (self.count + 1) == self.total
            print('self.total = {!r}'.format(self.total))
            print('self.count = {!r}'.format(self.count))
            print('need_next = {!r}'.format(need_next))
            print('self.last_external = {!r}'.format(self.last_external))
            if self.last_external:
                # always request next even if external
                # alg sent you back to rereview.
                need_next = True
            if self.infr is not None:
                self.infr.accept(feedback)
                if need_next:
                    self.continue_review()
            if not need_next:
                self.goto_next()

    def continue_review(self):
        print('[viz] continue review')
        user_request = self.infr.continue_review()
        print('user_request = {!r}'.format(user_request))
        if user_request is None:
            self.on_finished()
        else:
            self.request_review(user_request, external=True)

    def goto_next(self):
        if self.count is not None:
            # Move to the next item
            self.step_by(1)

    def set_edge(self, edge, info_text=None, external=True):
        print('set edge = %r' % (edge,))
        self.last_external = external
        self.history.add(edge)
        assert edge in self.history
        if not self.standalone:
            index = self.history.index(edge)
            self.count = index
            self.count = max(self.count, 0)
            self.count = min(self.count, self.total - 1)

            self.skip_but.setEnabled(self.count == self.total - 1)
            self.next_but.setEnabled(self.count != self.total - 1)
            self.prev_but.setEnabled(self.count != 0)

            self.index_edit.setText('{} / {}'.format(self.count + 1, self.total))

        self.was_confirmed = False
        edge_data = None if self.infr is None else self.infr.get_nonvisual_edge_data(edge)
        self.tuner.set_edge(edge, info_text)
        self.annot_state1.set_aid(edge[0])
        self.annot_state2.set_aid(edge[1])
        self.annot_review.set_edge(edge, edge_data)
        self.annot_review.setFocus(True)

    def edit_jump(self):
        index = int(self.index_edit.text().split('/')[0]) - 1
        index = max(0, index)
        index = min(self.total - 1, index)
        self.seek(index)

    def step_by(self, amount=1):
        self.seek(self.count + amount)

    def seek(self, index):
        assert isinstance(index, int)
        print('seek index = {}'.format(index))
        self.count = index
        self.count = max(self.count, 0)
        self.count = min(self.count, self.total - 1)
        if self.standalone:
            self.index_edit.setText('{} / {}'.format(self.count + 1, self.total))
            edge = self.edges[self.count]
            self.request_review([edge, None, {'standalone': True}], external=False)
        else:
            edge = self.history[index]
            print('request edge = %r' % (edge,))
            user_request = self.infr.emit_manual_review(edge)
            self.request_review(user_request, external=False)

    def request_review(self, user_request, external=True):
        edge, priority, edge_data = user_request[0]
        print('Got request for edge={}'.format(edge))
        info_text = 'edge=%r' % (edge,)
        info_text += '\npriority=%r' % (priority,)
        info_text += '\n' + ut.repr4(edge_data, si=True)
        self.set_edge(edge, info_text, external=external)
        self.show()

    def on_finished(self):
        if self.isVisible():
            gt.user_info(self, 'Review Complete')


class AnnotStateDialog(gt.GuitoolWidget):
    """
    python -m wbia.viz.viz_graph2 AnnotStateDialog --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_graph2 import *  # NOQA
        >>> import wbia.guitool as gt
        >>> gt.ensure_qapp()
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> aid = 1
        >>> self = AnnotStateDialog(ibs=ibs, aid=aid)
        >>> gt.qtapp_loop(qwin=self, freq=10)
    """

    def __init__(self, parent=None, **kwargs):
        # kwargs.pop('ori', kwargs.pop('ori', 'vert'))
        kwargs['ori'] = 'grid'
        # kwargs['margin'] = 1
        super(AnnotStateDialog, self).__init__(parent=parent, **kwargs)

    def initialize(self, ibs, aid=None):
        self.ibs = ibs

        const = ibs.const
        valid_quals = list(const.QUALITY_TEXT_TO_INT.keys())
        valid_views = list(const.VIEW.CODE_TO_NICE.keys())

        valid_quals = ut.partial_order(valid_quals, ['good', 'poor', 'ok'])
        valid_views = ut.partial_order(valid_views, ['left', 'right', 'front'])

        row_counter = it.count(0)

        row = next(row_counter)
        self.addNewLabel('Aid:', row=row, column=0, align='left')
        self.aid_label = self.addNewLabel(align='left', row=0, column=1)
        self.addNewLabel('--', row=row, column=2, align='center')
        self.time_label = self.addNewLabel(align='right', row=0, column=3)

        row = next(row_counter)
        self.addNewLabel('Quality:', row=row, column=0, align='left')
        self.qual_combo = self.addNewComboBox(
            options=valid_quals, editor_mode='hybrid', row=row, column=1, columnSpan=4,
        )
        gt.adjustSizePolicy(self.qual_combo, hStretch=5)

        row = next(row_counter)
        self.addNewLabel('Viewpoint:', row=row, column=0, align='left')
        self.view_combo = self.addNewComboBox(
            options=valid_views, editor_mode='hybrid', row=row, column=1, columnSpan=4,
        )
        gt.adjustSizePolicy(self.view_combo, hStretch=5)

        row = next(row_counter)
        self.addNewLabel('Multiple:', row=row, column=0, align='left')
        self.ismulti_cb = self.addNewCheckBox(
            row=row, column=1, columnSpan=4, direction='RightToLeft'
        )

        row = next(row_counter)
        self.addNewLabel('Tags:', row=row, column=0, align='left')
        # tag_box.addNewSpacer(hPolicy='Expanding')
        self.tag_edit = self.addNewTagEdit(row=row, column=1, columnSpan=4)
        self.tag_edit.setSizePolicy(
            gt.newSizePolicy(hSizePolicy='Expanding', vSizePolicy='Fixed', hStretch=100)
        )

        # self.addNewSpacer(hPolicy='Preferred', vPolicy='Preferred')

        self.set_all_margins(1)
        self.setMinimumHeight(1)
        self.setMinimumWidth(1)

        # gt.set_qt_object_names(locals())
        # gt.set_qt_object_names(vars(self))

        if aid is not None:
            self.set_aid(aid)

    # def _new_form_hbox(self, text, ori='horiz'):
    #     # form_box = self.addNewWidget(ori=ori, margin=1)
    #     label = form_box.addNewLabel(text, align='left')
    #     # gt.adjustSizePolicy(label, hStretch=1)
    #     # label.setObjectName('form_box_label_' + text)
    #     # form_box.addNewSpacer(hPolicy='Preferred', vPolicy='Preferred')
    #     return form_box

    def set_aid(self, aid):
        # read wbia state
        # TODO: allow read from infr graph node attributes
        # Set qt state
        import datetime

        annot_state = self.wbia_read(aid)

        unixtime = annot_state['image_unixtimes_asfloat']
        if unixtime is not None and not np.isnan(unixtime):
            date = datetime.datetime.fromtimestamp(unixtime)
            timestr = date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestr = 'NaN'

        self.time_label.setText(timestr)
        self.orig_state = annot_state
        self.aid_label.setText(repr(annot_state['aid']))
        self.qual_combo.setCurrentValue(annot_state['quality_texts'])
        self.view_combo.setCurrentValue(annot_state['viewpoint_code'])
        self.ismulti_cb.setChecked(bool(annot_state['multiple']))
        self.tag_edit.setTags(annot_state['case_tags'])

    def wbia_read(self, aid):
        ibs = self.ibs
        annot = ibs.annots([aid])[0]
        annot_dict = annot._make_lazy_dict()
        annot_state = ut.dict_subset(
            annot_dict,
            [
                'aid',
                'quality_texts',
                'viewpoint_code',
                'case_tags',
                'multiple',
                'image_unixtimes_asfloat',
            ],
        )
        if annot_state['quality_texts'] is None:
            annot_state['quality_texts'] = self.ibs.const.QUAL_UNKNOWN
        if annot_state['viewpoint_code'] is None:
            annot_state['viewpoint_code'] = const.VIEW.CODE.UNKNOWN
        return annot_state

    def current_annot_state(self):
        return {
            'aid': int(self.aid_label.text()),
            'quality_texts': self.qual_combo.currentValue(),
            'viewpoint_code': self.view_combo.currentValue(),
            'case_tags': self.tag_edit.tags(),
            'multiple': self.ismulti_cb.isChecked(),
        }


class EdgeReviewDialog(gt.GuitoolWidget):
    r"""

    python -m wbia.viz.viz_graph2 EdgeReviewDialog --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_graph2 import *  # NOQA
        >>> import wbia.guitool as gt
        >>> gt.ensure_qapp()
        >>> self = EdgeReviewDialog(edge=(1, 2))
        >>> gt.qtapp_loop(qwin=self, freq=10)
        >>> print(self.feedback_dict())
    """

    # def _new_form_hbox(self, text):
    #     form_box = self.addNewWidget(ori='horiz', margin=1)
    #     label = form_box.addNewLabel(text, align='left')
    #     label.setObjectName('form_box_label_' + text)
    #     gt.adjustSizePolicy(form_box, vStretch=1, vPolicy='Preferred')
    #     return form_box
    def __init__(self, parent=None, **kwargs):
        kwargs['ori'] = 'grid'
        super(EdgeReviewDialog, self).__init__(parent=parent, **kwargs)

    def initialize(
        self,
        edge=None,
        edge_data=None,
        conf_editor='hybrid',
        with_confirm=True,
        user_id=None,
    ):
        # from wbia.guitool.__PYQT__ import QtWidgets
        import wbia

        if user_id is None:
            user_id = ut.get_user_name() + '@' + ut.get_computer_name() + ':qt'

        EVIDENCE_DECISION = wbia.const.EVIDENCE_DECISION
        # TODO: meta decision
        CONFIDENCE = wbia.const.CONFIDENCE

        match_state_codes = list(EVIDENCE_DECISION.CODE_TO_INT.keys())
        user_conf_codes = list(CONFIDENCE.CODE_TO_INT.keys())

        match_state_codes = ut.partial_order(match_state_codes, [POSTV, NEGTV])
        user_conf_codes = ut.partial_order(
            user_conf_codes,
            # ['absolutely_sure', 'not_sure']
            ['pretty_sure', 'not_sure'],
        )
        match_state_options = [
            (EVIDENCE_DECISION.CODE_TO_NICE[code], code) for code in match_state_codes
        ]
        user_conf_options = [
            (CONFIDENCE.CODE_TO_NICE[code], code) for code in user_conf_codes
        ]

        self.set_all_margins(1)

        row_counter = it.count(0)

        row = next(row_counter)
        edge_row = self.addNewLabel('Edge:', row=row, column=0, align='left')
        self.edge_label = self.addNewLabel(align='right', row=row, column=1)

        row = next(row_counter)
        state_row = self.addNewLabel('Decision:', row=row, column=0, align='left')
        self.match_state_combo = self.addNewComboBox(
            options=match_state_options, editor_mode='hybrid', row=row, column=1
        )
        # self.match_state_combo.setStyleSheet(
        #     '''
        #     QRadioButton {
        #     text: "Bottom"
        #     }
        #     '''
        # )

        # tags_row = self.addNewWidget(ori='horiz', margin=1)
        row = next(row_counter)
        tags_row = self.addNewLabel('Other:', row=row, column=0, align='left')
        tags_row = self.addNewHWidget(row=row, column=1)
        # tags_row.addNewSpacer(hPolicy='Expanding', vPolicy='Fixed')
        self.tag_checkboxes = {}
        for tagname in ['photobomb', 'scenerymatch']:
            checkbox = tags_row.addNewCheckBox(tagname)
            checkbox.setObjectName('tag_cb_' + tagname)
            self.tag_checkboxes[tagname] = checkbox

        row = next(row_counter)
        pairtag_row = self.addNewLabel('Tags:', row=row, column=0, align='left')
        self.pairtag_edit = self.addNewTagEdit()

        row = next(row_counter)
        conf_row = self.addNewLabel('Confidence:', row=row, column=0, align='left')
        self.conf_combo = self.addNewComboBox(
            options=user_conf_options, editor_mode=conf_editor, row=row, column=1
        )

        row = next(row_counter)
        user_row = self.addNewLabel('User-ID:', row=row, column=0, align='left')
        self.user_edit = self.addNewLineEdit(user_id, row=row, column=1)

        # self.addNewSpacer(hPolicy='Expanding', vPolicy='Expanding')

        if with_confirm:
            row = next(row_counter)
            self.button_row = self.addNewWidget(
                ori='horiz', margin=1, row=row, column=0, columnSpan=2
            )
            self.button_row._guitool_layout.setAlignment(Qt.AlignBottom)
            self.confirm_button = self.button_row.addNewButton(
                'Confirm', pressed=self.confirm
            )
            self.cancel_button = self.button_row.addNewButton(
                'Cancel', pressed=self.cancel
            )
            # gt.adjustSizePolicy(self.button_row, vStretch=1)
        self.was_confirmed = False

        # gt.adjustSizePolicy(edge_row, vStretch=1)
        # gt.adjustSizePolicy(state_row, vStretch=1)
        # gt.adjustSizePolicy(tags_row, vStretch=1)
        # gt.adjustSizePolicy(conf_row, vStretch=1)
        # gt.adjustSizePolicy(user_row, vStretch=1)

        self.set_edge(edge, edge_data)

        self.set_all_margins(1)
        self.setMinimumHeight(1)
        self.setMinimumWidth(1)

        gt.set_qt_object_names(vars(self))
        gt.set_qt_object_names(locals())

    def keyPressEvent(self, event):
        print('Got event', event.key())
        handled = False
        if event.key() == gt.__PYQT__.QtCore.Qt.Key_F:
            self.match_state_combo.setCurrentValue(NEGTV)
            handled = True
        elif event.key() == gt.__PYQT__.QtCore.Qt.Key_T:
            self.match_state_combo.setCurrentValue(POSTV)
            handled = True
        elif event.key() == gt.__PYQT__.QtCore.Qt.Key_N:
            self.match_state_combo.setCurrentValue(INCMP)
            handled = True
        elif event.key() == gt.__PYQT__.QtCore.Qt.Key_P:
            self.match_state_combo.setCurrentValue(NEGTV)
            self.tag_checkboxes['photobomb'].setChecked(True)
            handled = True
        if not handled:
            super(EdgeReviewDialog, self).keyPressEvent(event)

    def read_edge_state(self, edge, edge_data):
        edge_state = {
            'edge': edge,
            'evidence_decision': UNREV,
            'meta_decision': 'null',
            'user_id': 'user:' + self.user_edit.text(),
            'tags': [],
            'timestamp_c1': ut.get_timestamp('int', isutc=True),
            # 'confidence': 'unspecified'
            'confidence': const.CONFIDENCE.CODE.NOT_SURE,
        }
        print('edge_data = %s' % (ut.repr4(edge_data),))
        if edge_data is not None:
            # Read edge state from edge_data
            edge_state['tags'] = edge_data.get('tags', [])
            edge_state['evidence_decision'] = edge_data.get('evidence_decision', UNREV)
            # Use previous confidence if you are the same user
            if edge_state['user_id'] == edge_data.get('user_id', None):
                edge_state['confidence'] = edge_data.get(
                    'confidence', edge_state['confidence']
                )
        return edge_state

    def set_edge(self, edge, edge_data=None):
        print('Set Edge State: edge=%r' % (edge,))
        edge_state = self.read_edge_state(edge, edge_data)

        if edge is not None and len(edge) != 2:
            raise ValueError('Edge must be 2 ints')
        print('edge_state = %s' % (ut.repr4(edge_state),))
        # set qt state
        self.edge = edge_state['edge']
        self.edge_label.setText(repr(edge_state['edge']))
        self.match_state_combo.setCurrentValue(edge_state['evidence_decision'])
        tags = edge_state['tags']
        tags = [] if tags is None else tags[:]
        remaining_tags = tags[:]
        for tagname, checkbox in self.tag_checkboxes.items():
            if tagname in tags:
                checkbox.setChecked(True)
                remaining_tags.remove(tagname)
            else:
                checkbox.setChecked(False)
        self.pairtag_edit.setTags(remaining_tags)
        self.conf_combo.setCurrentValue(edge_state['confidence'])
        self.timestamp_c1 = edge_state['timestamp_c1']

    def cancel(self):
        self.was_confirmed = False
        self.close()

    def confirm(self):
        self.was_confirmed = True
        self.close()

    def feedback_dict(self):
        import wbia

        decision_nice = self.match_state_combo.currentText()
        conf_nice = self.conf_combo.currentText()
        decision_code = wbia.const.EVIDENCE_DECISION.NICE_TO_CODE[decision_nice]
        tags = [key for key, check in self.tag_checkboxes.items() if check.checkState()]
        tags += self.pairtag_edit.tags()
        tags = [t for t in tags if len(t) != 0]
        confidence = wbia.const.CONFIDENCE.NICE_TO_CODE[conf_nice]
        user_id = self.user_edit.text()
        feedback = {
            'edge': self.edge,
            # 'aid1': self.edge[0],
            # 'aid2': self.edge[1],
            'evidence_decision': decision_code,
            'timestamp_c1': self.timestamp_c1,
            'timestamp_c2': ut.get_timestamp('int', isutc=True),
            'tags': tags,
            'confidence': confidence,
            'user_id': 'user:' + user_id,
        }
        return feedback


class DevGraphWidget(gt.GuitoolWidget):
    signal_graph_update = QtCore.pyqtSignal()

    def emit_graph_update(graph_widget):
        graph_widget.on_graph_update()

    def init_signals_and_slots(self):
        # https://doc.qt.io/archives/qtjambi-4.5.2_01/com/trolltech/qt/core/Qt.ConnectionType.html
        # connection_type = QtCore.Qt.AutoConnection
        # connection_type = QtCore.Qt.BlockingQueuedConnection
        connection_type = QtCore.Qt.DirectConnection
        # connection_type = QtCore.Qt.QueuedConnection
        self.signal_graph_update.connect(self.on_graph_update, type=connection_type)

    def initialize(graph_widget, use_image, self_parent):
        graph_widget.self_parent = self_parent
        graph_widget.plotinfo = None

        graph_widget.mpl_needs_update = True
        graph_widget.cb = None

        graph_widget.selected_aids = []
        graph_widget.splitter = graph_widget.addNewSplitter(ori=Qt.Horizontal)
        graph_widget.ctrls_ = graph_widget.splitter.addNewWidget(
            ori=Qt.Vertical, verticalStretch=1, margin=1, spacing=1
        )
        graph_widget.ctrls = graph_widget.ctrls_.addNewSplitter(ori='vert')

        graph_widget.mpl_wgt = mpl_widget.MatplotlibWidget(
            parent=graph_widget, horizontalStretch=1, pan_and_zoom=True
        )
        graph_widget.splitter.addWidget(graph_widget.mpl_wgt)

        ctrls = graph_widget.ctrls
        bbar1 = ctrls.addNewWidget(ori='vert', margin=1, spacing=1)
        bbar2 = ctrls.addNewWidget(ori='vert', margin=1, spacing=1)

        graph_widget.mark_state_funcs = self_parent.make_mark_state_funcs(
            graph_widget.selected_graph_pairs
        )
        for key, func in graph_widget.mark_state_funcs:
            bbar1.addNewButton(key.replace(' &', ': ').replace('&', ''), pressed=func)

        bbar1.addNewButton('Deselect', pressed=graph_widget.deselect)
        bbar1.addNewButton('Show Annots', pressed=graph_widget.show_selected)

        small_graph = len(self_parent.infr.aids) < 20

        import wbia

        GraphVizConfig = wbia.AnnotInference.make_viz_config(use_image, small_graph)

        def on_graphviz_config_changed(key=None):
            if key == 'pin_positions':
                graph_widget.set_pin_state(graph_widget.graphviz_config[key])
            else:
                graph_widget.emit_graph_update()

        graph_widget.graphviz_config = GraphVizConfig()
        graph_widget.graphviz_config_widget = PrefWidget2.EditConfigWidget(
            config=graph_widget.graphviz_config,
            with_buttons=False,
            changed=on_graphviz_config_changed,
        )
        # remove headers
        graph_widget.graphviz_config_widget.tree_view.header().hide()
        bbar2.addWidget(graph_widget.graphviz_config_widget)

        # Connect signals and slots
        graph_widget.mpl_wgt.click_inside_signal.connect(graph_widget.on_click_inside)
        graph_widget.mpl_wgt.key_press_signal.connect(graph_widget.on_key_press)
        graph_widget.mpl_wgt.pick_event_signal.connect(graph_widget.on_pick)
        graph_widget.splitter.setSizes([30, 70])
        graph_widget.init_signals_and_slots()

    @property
    def infr(graph_widget):
        return graph_widget.self_parent.infr

    def on_graph_update(graph_widget):
        # ut.cprint('[graph] on_graph_update', 'green')
        if graph_widget.mpl_wgt is None or graph_widget.mpl_wgt.visibleRegion().isEmpty():
            # Flag that graph should draw next time it is visible
            graph_widget.mpl_needs_update = True
        else:
            # Draw the graph because it is visible
            graph_widget.draw_graph()

    def draw_graph(graph_widget):
        # Is it possible to make things more responsive with threads?
        # http://stackoverflow.com/questions/20324804/how-to-use-qthread-correctly-in-pyqt-with-movetothread
        # self.my_thread = QtCore.QThread()
        # self.my_thread.start()
        # self.graph_widget.moveToThread(self.my_thread)
        print('[graph] draw_graph', 'green')
        graph_widget.mpl_needs_update = False
        graph_widget.mpl_wgt.ax.cla()

        visibility_kw = graph_widget.graphviz_config.asdict()

        visibility_kw.pop('pin_positions')
        in_image = visibility_kw.pop('in_image')
        use_image = visibility_kw.get('show_image')
        if use_image:
            graph_widget.infr.update_node_image_config(in_image=in_image)
        graph_widget.infr.update_visual_attrs(**visibility_kw)
        try:
            graph_widget.plotinfo = pt.show_nx(
                graph_widget.infr.graph,
                layout='custom',
                as_directed=False,
                ax=graph_widget.mpl_wgt.ax,
                use_image=use_image,
                verbose=0,
            )
        except IOError:
            graph_widget.infr.initialize_visual_node_attrs()
            graph_widget.plotinfo = pt.show_nx(
                graph_widget.infr.graph,
                layout='custom',
                as_directed=False,
                ax=graph_widget.mpl_wgt.ax,
                use_image=use_image,
                verbose=0,
            )
        graph_widget.mpl_wgt.ax.set_aspect('equal')

        for aid in graph_widget.selected_aids:
            graph_widget.highlight_aid(aid, True)

        graph_widget.mpl_wgt.canvas.draw()
        graph_widget.mpl_wgt.fig.subplots_adjust(
            left=0.02, top=0.98, bottom=0.02, right=0.85
        )

    def set_pin_state(graph_widget, flag):
        if flag:
            nx.set_node_attributes(graph_widget.infr.graph, name='pin', values='true')
        else:
            ut.nx_delete_node_attr(graph_widget.infr.graph, 'pin')

    def selected_graph_pairs(graph_widget):
        return it.combinations(graph_widget.selected_aids, 2)

    def show_selected(graph_widget):
        print('[graph_widget] show_selected')
        # old = getattr(graph_widget, '_figscope', None)
        # if old is not None:
        #     old.close()
        if len(graph_widget.selected_aids) == 2:
            edge = tuple(sorted(graph_widget.selected_aids))
            self = AnnotPairDialog(infr=graph_widget.infr, edge=edge)
            self.show()
            self.activateWindow()
            self.raise_()
            graph_widget._figscope = self
        else:
            # fnum = pt.ensure_fnum(10)
            # print('fnum = %r' % (fnum,))
            fig = graph_widget.infr.draw_aids(graph_widget.selected_aids, fnum=10)
            # fig = pt.figure(fnum=fnum)
            # viz_chip.show_many_chips(
            #     graph_widget.infr.ibs,
            #     graph_widget.selected_aids,
            #     fnum=fnum
            # )
            # fig.canvas.update()
            fig.show()
            fig.canvas.draw()
            graph_widget._figscope = fig

    def highlight_aid(graph_widget, aid, color=None):
        # TODO: move to mpl widget
        if graph_widget.plotinfo is None:
            return
        node = aid
        frame = graph_widget.plotinfo['patch_frame_dict'][node]
        node_dict = ut.nx_node_dict(graph_widget.infr.graph)

        framewidth = node_dict[node]['framewidth']
        if color is True:
            color = pt.ORANGE
        if color is None or color is False:
            color = pt.DARK_BLUE
            color = node_dict[node]['color']
            color = pt.ensure_nonhex_color(color)
            frame.set_linewidth(framewidth)
        else:
            frame.set_linewidth(framewidth * 2)
        frame.set_facecolor(color)
        frame.set_edgecolor(color)

    def deselect(graph_widget):
        print('[graph_widget] deselect')
        # print('graph_widget.selected_aids = %r' % (graph_widget.selected_aids,))
        graph_widget.toggle_selected_aid(graph_widget.selected_aids[:])

    def toggle_selected_aid(graph_widget, aids):
        print('[graph_widget] toggle_selected_aid')
        for aid in ut.ensure_iterable(aids):
            # TODO: move to mpl widget
            if aid in graph_widget.selected_aids:
                graph_widget.selected_aids.remove(aid)
                # graph_widget.highlight_aid(aid, pt.WHITE)
                graph_widget.highlight_aid(aid, color=None)
            else:
                graph_widget.selected_aids.append(aid)
                graph_widget.highlight_aid(aid, True)
        print('graph_widget.selected_aids = %r' % (graph_widget.selected_aids,))
        if graph_widget.mpl_wgt is not None:
            graph_widget.mpl_wgt.fig.canvas.draw()

    def show_popup_menu(graph_widget, options, event):
        """
        context menu
        """
        height = graph_widget.mpl_wgt.fig.canvas.geometry().height()
        qpoint = gt.newQPoint(event.x, height - event.y)
        qwin = graph_widget.mpl_wgt.fig.canvas
        gt.popup_menu(qwin, qpoint, options)

    def on_key_press(graph_widget, event):
        # called by matplotlib events
        key = event.key.upper()
        option_dict = gt.make_option_dict(graph_widget.mark_state_funcs, shortcuts=True)
        assert 'D' not in option_dict
        option_dict['D'] = graph_widget.deselect
        if key in option_dict:
            option_dict[key]()

    def on_pick(self, event):
        artist = event.artist
        plotdat = pt.get_plotdat_dict(artist)
        infr = self.self_parent.infr
        if plotdat:
            if 'node' in plotdat:
                if False:
                    all_node_data = ut.sort_dict(plotdat['node_data'].copy())
                    visual_node_data = ut.dict_subset(
                        all_node_data, infr.visual_node_attrs, None
                    )
                    node_data = ut.delete_dict_keys(all_node_data, infr.visual_node_attrs)
                    print('visual_node_data: ' + ut.repr2(visual_node_data, nl=1))
                    print('node_data: ' + ut.repr2(node_data, nl=1))
                    print('node: ' + ut.repr2(plotdat['node']))
            elif 'edge' in plotdat:
                all_edge_data = ut.sort_dict(plotdat['edge_data'].copy())
                visual_edge_data = ut.dict_subset(
                    all_edge_data, infr.visual_edge_attrs, None
                )
                edge_data = ut.delete_dict_keys(all_edge_data, infr.visual_edge_attrs)
                print('visual_edge_data: ' + ut.repr2(visual_edge_data, nl=1))
                print('edge_data: ' + ut.repr2(edge_data, nl=1))
                print('edge: ' + ut.repr2(plotdat['edge']))
            else:
                print('unknown artist ' + ut.repr2(plotdat))
                print('artist = %r' % (artist,))
                print('event = %r' % (event,))

    def on_click_inside(graph_widget, event, ax):
        pos = graph_widget.plotinfo['node']['pos']
        nodes = list(pos.keys())
        pos_list = ut.dict_take(pos, nodes)

        # TODO: FIXME
        # x = 10
        # y = 10
        x, y = event.xdata, event.ydata
        point = np.array([x, y])
        pos_list = np.array(pos_list)
        index, dist = vt.closest_point(point, pos_list, distfunc=vt.L2)
        node = nodes[index]
        aid = node
        context_shown = False

        if event.button == 3 and not context_shown:
            if len(graph_widget.selected_aids) != 2:
                print('This funciton only work if exactly 2 are selected')
            else:
                context_shown = True
                aid1, aid2 = graph_widget.selected_aids
                aid_pairs = [(aid1, aid2)]
                options = graph_widget.self_parent.get_edge_options(aid_pairs)
                graph_widget.show_popup_menu(options, event)

        bbox = vt.bbox_from_center_wh(
            graph_widget.plotinfo['node']['pos'][node],
            graph_widget.plotinfo['node']['size'][node],
        )
        annot_selected = vt.point_inside_bbox(point, bbox)

        if annot_selected:
            ibs = graph_widget.infr.ibs
            print(ut.repr2(ibs.get_annot_info(aid, default=True, name=True, gname=True)))
            if event.button == 1:
                graph_widget.toggle_selected_aid(aid)

            if event.button == 3 and not context_shown:
                # right click
                from wbia.viz.interact import interact_chip

                context_shown = True
                # refresh_func = functools.partial(viz.show_name, ibs, nid,
                # fnum=fnum, sel_aids=sel_aids)
                refresh_func = None
                config2_ = None
                options = interact_chip.build_annot_context_options(
                    graph_widget.infr.ibs,
                    aid,
                    refresh_func=refresh_func,
                    with_interact_name=False,
                    config2_=config2_,
                )
                graph_widget.show_popup_menu(options, event)

    def eventFilter(graph_widget, source, event):
        if event.type() == QtCore.QEvent.Show:
            if graph_widget.mpl_needs_update:
                graph_widget.emit_graph_update()
        return super(DevGraphWidget, graph_widget).eventFilter(source, event)


class AnnotGraphWidget(gt.GuitoolWidget):
    signal_state_update = QtCore.pyqtSignal(bool, bool)

    def initialize(
        self, infr=None, use_image=False, init_mode='rereview', review_cfg=None
    ):
        print('[viz_graph] initialize')

        self.init_mode = init_mode
        print('self.init_mode = %r' % (self.init_mode,))

        if review_cfg is None:
            mode = 'filtered' if self.init_mode == 'split' else 'unfiltered'
            self.preset_config(mode)

        self.infr = infr
        self.initialize_menus()

        self.graph_tab_widget = self.addNewTabWidget(verticalStretch=1)

        self.statbar1 = self.addNewWidget(
            ori='horiz', verticalStretch=1, margin=1, spacing=1
        )
        self.statbar2 = self.addNewWidget(
            ori='horiz', verticalStretch=1, margin=1, spacing=1
        )

        self.prog_bar = self.addNewProgressBar(visible=False)

        self.initialize_api_tabs()

        self.statbar1.addNewButton(
            'Match and Score', min_width=1, pressed=self.match_and_score_edges
        )
        self.statbar1.addNewButton(
            'ScoreVsOne', min_width=1, pressed=self.score_edges_vsone
        )
        self.statbar1.addNewButton('Edit Filters', min_width=1, pressed=self.edit_filters)
        self.statbar1.addNewButton('Repopulate', min_width=1, pressed=self.repopulate)

        self.statbar2.addNewButton(
            'Reset DBState', min_width=1, pressed=self.reset_review
        )
        self.statbar2.addNewButton(
            'Reset Rereview', min_width=1, pressed=self.reset_rereview
        )

        self.num_names_lbl = self.statbar2.addNewLabel('NUM_NAMES_LBL')
        self.state_lbl = self.statbar2.addNewLabel('STATE_LBL')

        self.statbar2.addNewButton('Accept', pressed=self.accept)

        # _show_graph = self.init_mode in ['split', 'rereview', 'review']
        _show_graph = True
        if _show_graph:
            # TODO: separate graph view into its own class
            self.graph_tab = self.graph_tab_widget.addNewTab('Graph')
            # TODO: make this its own proper widget
            self.graph_widget = DevGraphWidget(
                parent=self, self_parent=self, use_image=use_image
            )
            self.graph_tab.addWidget(self.graph_widget)
            # self.graph_widget.connect_kepress_to_slot
        else:
            self.graph_widget = None
            self.graph_tab = None
        self.init_signals_and_slots()

    def init_signals_and_slots(self):
        # https://doc.qt.io/archives/qtjambi-4.5.2_01/com/trolltech/qt/core/Qt.ConnectionType.html
        # connection_type = QtCore.Qt.AutoConnection
        # connection_type = QtCore.Qt.BlockingQueuedConnection
        connection_type = QtCore.Qt.DirectConnection
        # connection_type = QtCore.Qt.QueuedConnection
        self.signal_state_update.connect(self.update_state, type=connection_type)

    def initialize_api_tabs(self):
        self.api_tabs = {}
        self.api_widgets = {}

        def _add_item_widget_tab(key, view_class='table'):
            title = key.title().replace('_', ' ')
            self.api_tabs[key] = self.graph_tab_widget.addNewTab(title)
            self.api_widgets[key] = gt.APIItemWidget(view_class=view_class)
            self.api_tabs[key].addWidget(self.api_widgets[key])

        _add_item_widget_tab('edges')
        _add_item_widget_tab('nodes')
        _add_item_widget_tab('name_nodes', view_class='tree')
        _add_item_widget_tab('name_edges', view_class='tree')

        node_view = self.api_widgets['nodes'].view
        node_view.contextMenuClicked.connect(self.node_context)
        node_view.connect_single_key_to_slot(gt.ALT_KEY, self.on_alt_pressed)

        name_node_view = self.api_widgets['name_nodes'].view
        name_node_view.contextMenuClicked.connect(self.node_context)
        name_node_view.connect_single_key_to_slot(gt.ALT_KEY, self.on_alt_pressed)

        edge_view = self.api_widgets['edges'].view
        edge_view.doubleClicked.connect(self.edge_doubleclick)
        edge_view.contextMenuClicked.connect(self.edge_context)
        edge_view.connect_keypress_to_slot(self.edge_keypress)
        edge_view.connect_single_key_to_slot(gt.ALT_KEY, self.on_alt_pressed)

        name_edge_view = self.api_widgets['name_edges'].view
        name_edge_view.doubleClicked.connect(self.edge_doubleclick)
        name_edge_view.contextMenuClicked.connect(self.edge_context)
        name_edge_view.connect_keypress_to_slot(self.edge_keypress)
        name_edge_view.connect_single_key_to_slot(gt.ALT_KEY, self.on_alt_pressed)

    def populate_edge_model(self):
        print('[viz_graph] populate_edge_model')
        # if self.init_mode is None:
        #     self.review_cfg['show_match_thumb'] = False
        key = 'edges'
        tab = self.api_tabs[key]
        widget = self.api_widgets[key]
        api = make_edge_api(self.infr, review_cfg=self.review_cfg)
        title = key.title().replace('_', ' ')
        headers = api.make_headers(tblnice=title, tblname=key)
        widget.change_headers(headers)
        widget.resize_headers(api)
        widget.view.verticalHeader().setVisible(True)
        tab.setTabText('%s (%r)' % (title, widget.model.num_rows_total))
        widget.view.verticalHeader().setDefaultSectionSize(221)

    def populate_node_model(self):
        print('[viz_graph] populate_node_model')
        api = make_node_api(self.infr)
        key = 'nodes'
        tab = self.api_tabs[key]
        widget = self.api_widgets[key]
        title = key.title().replace('_', ' ')
        headers = api.make_headers(tblnice=title, tblname=key)
        widget.change_headers(headers)
        widget.view.verticalHeader().setVisible(True)
        try:
            widget.view.verticalHeader().setMovable(True)
        except AttributeError:
            widget.view.verticalHeader().setSectionsMovable(True)
        tab.setTabText('%s (%r)' % (title, widget.model.num_rows_total))

    def populate_name_node_model(self):
        api = make_name_node_api(self.infr, review_cfg=self.review_cfg)
        key = 'name_nodes'
        tab = self.api_tabs[key]
        widget = self.api_widgets[key]
        title = key.title().replace('_', ' ')
        headers = api.make_headers(tblnice=title, tblname=key)
        widget.change_headers(headers)
        tab.setTabText('%s (%r)' % (title, widget.model.num_rows_total))

    def populate_name_edge_model(self):
        api = make_name_edge_api(self.infr, review_cfg=self.review_cfg)
        key = 'name_edges'
        tab = self.api_tabs[key]
        widget = self.api_widgets[key]
        title = key.title().replace('_', ' ')
        headers = api.make_headers(tblnice=title, tblname=key)
        widget.change_headers(headers)
        tab.setTabText('%s (%r)' % (title, widget.model.num_rows_total))

    def initialize_menus(self):
        self.menubar = gt.newMenubar(self)
        self.menus = {}

        key = 'Dev'
        menu = self.menus[key] = self.menubar.newMenu(key)
        menu.newAction(triggered=self.print_info)
        menu.newAction(triggered=self.embed, shortcut='ctrl+shift+I')
        menu.newAction(triggered=self.expand_image_and_names)
        menu.newAction(triggered=self.emit_state_update)
        menu.newAction(triggered=self.print_staging_table)
        menu.newAction(triggered=self.print_annotmatch_table)
        menu.newAction(triggered=self.print_deltas)

        key = 'Actions'
        menu = self.menus[key] = self.menubar.newMenu(key)
        menu.newAction(triggered=self.commit_to_staging)

        key = 'Debug'
        menu = self.menus[key] = self.menubar.newMenu(key)
        menu.newAction(triggered=self.use_wbia_names)
        menu.newAction(triggered=self.reset_staging)
        menu.newAction(triggered=self.reset_annotmatch)
        menu.newAction(triggered=self.ensure_full)
        menu.newAction(triggered=self.ensure_cliques)
        menu.newAction(triggered=self.vsone_subset)
        menu.newAction(
            text='relabel_using_reviews',
            triggered=lambda: self.infr.relabel_using_reviews(),
        )
        menu.newAction(
            text='apply_nondynamic_update',
            triggered=lambda: self.infr.apply_nondynamic_update(),
        )

    def preset_config(self, mode='filtered'):
        print('[graph] preset_config mode=%r' % (mode,))
        if mode == 'filtered':
            self.review_cfg = GRAPH_REVIEW_CFG_DEFAULTS.copy()
        elif mode == 'unfiltered':
            self.review_cfg = GRAPH_REVIEW_CFG_DEFAULTS.copy()
            for key in self.review_cfg.keys():
                if key.startswith('filter_'):
                    self.review_cfg[key] = False

    def showEvent(self, event):
        super(AnnotGraphWidget, self).showEvent(event)
        ut.cprint('[viz_graph] showEvent', 'green')
        # Fire initialize event after we show the GUI
        QtCore.QTimer.singleShot(50, self.init_inference)

    def init_inference(self):
        print('[viz_graph] init_inference mode=%r' % (self.init_mode))
        if self.init_mode is None:
            pass
        elif self.init_mode == 'split':
            self.preset_config('filtered')
            self.reset_split()
        elif self.init_mode == 'rereview':
            self.preset_config('unfiltered')
            self.reset_rereview()
        elif self.init_mode == 'review':
            self.reset_review()
        else:
            raise ValueError('Unknown init_mode=%r' % (self.init_mode,))
        self.repopulate()

        if ut.get_argflag('--graphtab'):
            index = self.graph_tab_widget.indexOf(self.graph_tab)
            self.graph_tab_widget.setCurrentIndex(index)
            # self.graph_tab_widget.setCurrentIndex(2)

        match = ut.get_argval('--match', type_=list, default=None)
        import ubelt as ub

        if match:
            pairs = list(ub.chunks(match, 2))
            self.mark_pair_state(pairs, POSTV)
        nomatch = ut.get_argval('--nomatch', type_=list, default=None)
        if nomatch:
            pairs = list(ub.chunks(nomatch, 2))
            self.mark_pair_state(pairs, NEGTV)

    def repopulate(self):
        # self.update_state(structure_changed=True)
        self.emit_state_update(structure_changed=True)

    def emit_state_update(self, structure_changed=False, disable_global_update=False):
        self.signal_state_update.emit(structure_changed, disable_global_update)

    def update_state(self, structure_changed=False, disable_global_update=False):
        print('[viz_graph] update_state mode=%s' % (self.init_mode,))
        # if self.init_mode in ['split', 'rereview']:
        if not disable_global_update:
            if self.init_mode == 'split':
                self.infr.apply_feedback_edges()
                self.infr.relabel_using_reviews()
            elif self.init_mode == 'rereview':
                self.infr.apply_feedback_edges()
                # self.infr.apply_match_scores()
                self.infr.relabel_using_reviews()
            elif self.init_mode == 'review':
                self.infr.apply_match_edges()
                self.infr.ensure_mst()
                self.infr.apply_feedback_edges()
                # self.infr.apply_match_scores()
                self.infr.relabel_using_reviews()
                self.infr.apply_nondynamic_update()

        # Set gui status indicators
        status = self.infr.connected_component_status()
        truth_colors = self.infr._get_truth_colors()
        if status['num_inconsistent']:
            self.state_lbl.setText(
                'Inconsistent Names: %d' % (status['num_inconsistent'],)
            )
            self.state_lbl.setColor('black', truth_colors[NEGTV][0:3] * 255)
        else:
            self.state_lbl.setText('Consistent')
            self.state_lbl.setColor('black', truth_colors[POSTV][0:3] * 255)

        self.num_names_lbl.setText('Names: max=%r' % (status['num_names_max']))

        # print('[viz_graph] on_update_state mode=%s' % (self.init_mode,))
        if structure_changed:
            # TODO: only make this API if the tab is clicked
            self.populate_node_model()
            self.populate_edge_model()
            self.populate_name_node_model()
            self.populate_name_edge_model()
        if self.graph_widget is not None:
            self.graph_widget.emit_graph_update()

        for widget in self.api_widgets.values():
            model = widget.view.model()
            # FIXME: should only clear cache in viewport
            model.clear_cache()
            model.layoutChanged.emit()

    def ensure_cliques(self):
        self.infr.ensure_cliques()
        self.infr.relabel_using_reviews()
        self.infr.apply_nondynamic_update()
        self.repopulate()

    def ensure_full(self):
        self.infr.ensure_full()
        self.infr.relabel_using_reviews()
        self.infr.apply_nondynamic_update()
        self.repopulate()

    def match_and_score_edges(self):
        with gt.GuiProgContext('Scoring Edges', self.prog_bar) as ctx:
            # TODO: add in a cfgdict here with settable params
            self.infr.exec_matching(prog_hook=ctx.prog_hook)
            self.infr.apply_match_edges(self.review_cfg)
            # self.infr.apply_match_scores()
        self.repopulate()

    def score_edges_vsone(self):
        # DEPRICATE
        # TODO: replace with new interface
        with gt.GuiProgContext('Scoring Edges', self.prog_bar) as ctx:
            edges = list(self.infr.edges())
            self.infr.exec_vsone_subset(edges, prog_hook=ctx.prog_hook)
            # self.infr.exec_vsone(prog_hook=ctx.prog_hook)
            # self.infr.apply_match_scores()
        self.repopulate()

    def reset_review(self):
        print('[viz_graph] reset_review')
        infr = self.infr
        with gt.GuiProgContext('Reset Review', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            with ut.Timer('reset_feedback'):
                infr.reset_feedback('staging', apply=True)
            with ut.Timer('reinit_name_labels'):
                infr.reset_labels_to_wbia()
            if self.graph_widget is not None:
                self.graph_widget.set_pin_state(True)
            with ut.Timer('ensure_mst'):
                infr.ensure_mst()
            with ut.Timer('apply_match_edges'):
                infr.apply_match_edges()
            # with ut.Timer('apply_match_scores'):
            #     infr.apply_match_scores()
            # with ut.Timer('relabel_using_reviews'):
            #     self.infr.relabel_using_reviews()
            # with ut.Timer('apply_nondynamic_update'):
            #     self.infr.apply_nondynamic_update()
            self.repopulate()
            ctx.set_progress(3, 3)

    def reset_annotmatch(self):
        print('[viz_graph] reset_rereview')
        infr = self.infr
        with gt.GuiProgContext('Reset Review', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr.reset_feedback('annotmatch', apply=True)
            infr.relabel_using_reviews()
            ctx.set_progress(msg='repopulate')
            self.repopulate()
            ctx.set_progress(8, 8)

    def reset_staging(self):
        print('[viz_graph] reset_rereview')
        infr = self.infr
        with gt.GuiProgContext('Reset Review', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr.reset_feedback('staging', apply=True)
            infr.relabel_using_reviews()
            ctx.set_progress(msg='repopulate')
            self.repopulate()
            ctx.set_progress(8, 8)

    def reset_rereview(self):
        """
        Goal:
            All names are removed.
            Reset edges so only reviewed edges are shown.
            You can change the state of those edges.
            They are not filtered.
        """
        print('[viz_graph] reset_rereview')
        infr = self.infr
        self.init_mode = 'rereview'
        with gt.GuiProgContext('Reset Review', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr.initialize_graph()
            if self.graph_widget is not None:
                self.graph_widget.set_pin_state(True)
            ctx.set_progress(msg='reset name labels')
            infr.reset_name_labels()
            ctx.set_progress(msg='reset feedback')
            infr.reset_feedback('staging', apply=True)
            ctx.set_progress(msg='remove name labels')
            infr.clear_name_labels()
            # ctx.set_progress(msg='apply match scores')
            # infr.apply_match_scores()
            ctx.set_progress(msg='repopulate')
            self.repopulate()
            ctx.set_progress(8, 8)

    def reset_split(self):
        infr = self.infr
        self.init_mode = 'split'
        with gt.GuiProgContext('Initializing', self.prog_bar) as ctx:
            ctx.set_progress(0, 3)
            infr.initialize_graph()
            if self.graph_widget is not None:
                self.graph_widget.set_pin_state(True)
            ctx.set_progress(1, 3)
            infr.remove_feedback()
            ctx.set_progress(2, 3)
            infr.clear_name_labels()
            ctx.set_progress(3, 3)

    def edit_filters(self):
        # TODO: split up review configs / show thumbs etc...
        config = dtool.Config.from_dict(self.review_cfg)
        dlg = gt.ConfigConfirmWidget.as_dialog(
            self,
            title='Edit Filters',
            msg='Edit Filters',
            with_spoiler=False,
            config=config,
        )
        dlg.resize(700, 500)
        dlg.exec_()
        print('config = %r' % (config,))
        updated_config = dlg.widget.config  # NOQA
        print('updated_config = %r' % (updated_config,))
        self.review_cfg = updated_config.asdict()
        self.repopulate()

    def edge_doubleclick(self, qtindex):
        """
        qtindex = qtindex = self.api_widgets['edges'].view.get_row_and_qtindex_from_id(1)[0]
        """
        print('[viz_graph] DoubleClicked: ' + str(gt.qtype.qindexinfo(qtindex)))
        model = qtindex.model()
        aid1 = model.get_header_data('aid1', qtindex)
        aid2 = model.get_header_data('aid2', qtindex)
        cm, aid1, aid2 = self.infr.lookup_cm(aid1, aid2)
        if cm is not None:
            cm.ishow_single_annotmatch(self.infr.qreq_, aid2, mode=0)
        else:
            # Hack
            self.graph_widget.deselect()
            self.graph_widget.toggle_selected_aid([aid1, aid2])
            # self.graph_widget.selected_aids = [aid1, aid2]
            self.graph_widget.show_selected()

    def mark_pair_state(self, pairs, state):
        valid_states = {POSTV, NEGTV, INCMP, UNREV}
        statetags = state.split('+')
        state = statetags[0]
        tags = statetags[1].split(';') if len(statetags) > 1 else []
        assert state in valid_states
        for aid1, aid2 in pairs:
            user_id = ut.get_user_name() + '@' + ut.get_computer_name() + ':qt-mark'
            self.infr.add_feedback((aid1, aid2), state, tags=tags, user_id=user_id)
        self.emit_state_update(disable_global_update=True)

    def make_mark_state_funcs(self, selection_func):
        def _mark_selected_pair_state(state):
            self.mark_pair_state(selection_func(), state)

        options = [
            ('Mark &True', ut.partial(_mark_selected_pair_state, POSTV)),
            ('Mark &False', ut.partial(_mark_selected_pair_state, NEGTV)),
            ('Mark &Not-Comparable', ut.partial(_mark_selected_pair_state, INCMP)),
            (
                'Mark &Photobomb',
                ut.partial(_mark_selected_pair_state, 'nomatch+photobomb'),
            ),
            (
                'Mark &SceneryMatch',
                ut.partial(_mark_selected_pair_state, 'nomatch+scenerymatch'),
            ),
            ('&Unreview', ut.partial(_mark_selected_pair_state, UNREV)),
            # unreview will only remove internal feedback, anything commited will not change
        ]
        return options

    def name_selection(self, view):
        selected_qtindex_list_ = view.selectedRows()
        selected_qtindex_names = []
        for qtindex in selected_qtindex_list_:
            model = qtindex.model()
            if model.name in {'name_edges', 'name_nodes'}:
                level = qtindex.internalPointer().level
                if level == 0:
                    selected_qtindex_names.append(qtindex)
        name_labels = []
        for qtindex in selected_qtindex_names:
            model = qtindex.model()
            name_label = model.get_header_data('name_label', qtindex)
            name_labels.append(name_label)
        return name_labels

    def edge_selection(self, view):
        selected_qtindex_list_ = view.selectedRows()
        selected_qtindex_edges = []
        for qtindex in selected_qtindex_list_:
            model = qtindex.model()
            if model.name == 'name_edges':
                level = qtindex.internalPointer().level
                if level == 1:
                    selected_qtindex_edges.append(qtindex)
            elif model.name == 'edges':
                selected_qtindex_edges.append(qtindex)
        aid_pairs = []
        for qtindex in selected_qtindex_edges:
            model = qtindex.model()
            aid1 = model.get_header_data('aid1', qtindex)
            aid2 = model.get_header_data('aid2', qtindex)
            aid_pairs.append((aid1, aid2))
        return aid_pairs

    def node_selection(self, view):
        selected_qtindex_list = view.selectedRows()
        aids = [
            qtindex.model().get_header_data('aid', qtindex)
            for qtindex in selected_qtindex_list
        ]
        return aids

    def get_node_options(self, aids):
        """
        Context-menu options for annotation nodes
        """
        from wbia.viz.interact import interact_chip

        options = []
        if len(aids) == 1:
            options += interact_chip.build_annot_context_options(
                self.infr.ibs,
                aids[0],
                refresh_func=None,
                with_interact_name=False,
                config2_=None,
            )
        if len(aids) > 1:
            aid_pairs = list(it.combinations(aids, 2))
            options += self.get_edge_options(aid_pairs)
        return options

    def custom_review(self, aid_pairs):
        assert len(aid_pairs) == 1
        aid1, aid2 = aid_pairs[0]
        # import utool
        # utool.embed()
        edge_data = self.infr.get_nonvisual_edge_data((aid1, aid2))
        dlg = EdgeReviewDialog.as_dialog(self, edge=(aid1, aid2), edge_data=edge_data)
        dlg.resize(400, 300)
        dlg.exec_()
        if dlg.widget.was_confirmed:
            feedback = dlg.widget.feedback_dict()
            self.infr.add_feedback(**feedback)

    def get_edge_options(self, aid_pairs):
        """
        Context-menu options for annotation edges
        """
        options = []
        if len(aid_pairs) > 0:
            options += self.make_mark_state_funcs(lambda: aid_pairs)

        if len(aid_pairs) == 1:
            options += [
                ('&Custom Review', lambda: self.custom_review(aid_pairs)),
            ]
            from wbia.gui import inspect_gui

            ibs = self.infr.ibs
            aid1, aid2 = aid_pairs[0]
            qreq_ = self.infr.qreq_
            pair_tag_options = inspect_gui.make_aidpair_tag_context_options(
                ibs, aid1, aid2
            )
            chip_context_options = inspect_gui.make_annotpair_context_options(
                ibs, aid1, aid2, qreq_=qreq_
            )
            external_options = []
            external_options += [('Match Ta&gs', pair_tag_options)]
            external_options += chip_context_options
            options += [
                ('External Options', external_options),
            ]

            nids = ut.unique(ibs.get_annot_nids([aid1, aid2]))
            options += [
                (
                    'New Split Case Interaction',
                    ut.partial(make_qt_graph_interface, ibs, nids=nids),
                ),
            ]

            options += [
                (
                    'Tune Vsone(vt)',
                    inspect_gui.make_vsone_context_options(ibs, aid1, aid2, qreq_)[0][1],
                )
            ]

        # if len(selected_qtindex_names) > 0:
        #     import utool
        #     utool.embed()
        #     options += [
        #         # TODO
        #         # ('Restrict Graph To Names', lambda: self.restrict_graph_to_names),
        #         ('Vsone subset', lambda: self.vsone_subset()),
        #     ]
        # else:
        if len(aid_pairs) > 0:
            options += [
                ('Vsone subset', lambda: self.vsone_subset(aid_pairs)),
            ]

        return options

    def vsone_subset(self, edges=None):
        print('[graph] vsone_subset')
        if edges is None:
            edges = self.infr.graph.edges()
        self.infr.exec_vsone_subset(edges)

    def restrict_graph_to_names(self):
        pass

    @gt.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def edge_context(self, qtindex, qpoint):
        print('context')
        # print('option_dict = %s' % (ut.repr3(option_dict, nl=2),))
        model = qtindex.model()
        view = model.view
        aid_pairs = self.edge_selection(view)
        options = self.get_edge_options(aid_pairs)
        gt.popup_menu(self, qpoint, options)

    @gt.slot_(QtCore.QModelIndex, QtCore.QPoint)
    def node_context(self, qtindex, qpoint):
        view = qtindex.model().view
        aids = self.node_selection(view)
        options = self.get_node_options(aids)
        gt.popup_menu(self, qpoint, options)

    def on_alt_pressed(self, view, event):
        selected_qtindex_list = view.selectedRows()
        if len(selected_qtindex_list) > 0:
            # popup context menu on alt
            qtindex = selected_qtindex_list[-1]
            qrect = view.visualRect(qtindex)
            pos = qrect.center()
            self.edge_context(qtindex, pos)

    def edge_keypress(self, view, event):
        """
        view = self.api_widgets['edges'].view
        """
        event_key = event.key()
        aid_pairs = self.edge_selection(view)
        options = self.get_edge_options(aid_pairs)
        option_dict = gt.make_option_dict(options, shortcuts=True)
        handled = False
        for key, func in option_dict.items():
            if event_key == getattr(QtCore.Qt, 'Key_' + key.upper()):
                func()
                handled = True
                break
        if not handled:
            print('Key  not handled %r' % (event_key,))
            return
        return handled

    def print_staging_table(self):
        db = self.infr.ibs.staging
        print(db.get_table_csv('reviews'))

    def print_annotmatch_table(self):
        db = self.infr.ibs.db
        print(db.get_table_csv('annotmatch'))

    def commit_to_staging(self):
        print('[graph] commit to staging')
        self.infr.write_wbia_staging_feedback()

    def print_deltas(self):
        pairs = [
            ('external', 'internal'),
            ('annotmatch', 'all'),
            ('staging', 'all'),
            ('annotmatch', 'staging'),
        ]
        for old, new in pairs:
            print('old = %r' % (old,))
            print('new = %r' % (new,))
            print(self.infr.match_state_delta(old, new))

    def use_wbia_names(self):
        # Hack
        infr = self.infr
        num_names, num_inconsistent = infr.relabel_using_reviews()
        aid_to_newname = infr.get_wbia_name_delta()
        nx.set_node_attributes(
            infr.graph, name='name_label', values=aid_to_newname['new_name'].to_dict()
        )

    def hack_keep_old_tags(self):
        # Creates new reviews that rectify old tags in the annotmatch table
        # into the staging table.
        assert len(self.infr.internal_feedback) == 0
        infr = self.infr
        edge_delta_df = infr.match_state_delta(old='annotmatch', new='external')
        # Find those with tag changes
        tag_flags = (edge_delta_df['old_tags'] != edge_delta_df['new_tags']) & (
            ~(edge_delta_df['am_rowid'].isnull())
        )
        tag_delta_df = edge_delta_df[tag_flags]
        aid_pairs = tag_delta_df.index
        decision = tag_delta_df['new_decision']
        tags = tag_delta_df['old_tags'] + tag_delta_df['new_tags']
        tags = tags.map(ut.unique)

        for (aid1, aid2), state, tags in zip(aid_pairs, decision, tags):
            infr.add_feedback((aid1, aid2), state, tags=tags)

        infr.apply_feedback_edges()
        infr.apply_nondynamic_update()

    def accept(self):
        """
        First determine deltas of what has changed between internal state +
        staging and the the annotations node names as well as the annotmatch
        edges. Then commit to staging, annotmatch, and the annotation table.
        """
        print('[viz_graph] accept')
        infr = self.infr

        # Ensure internal state is up to date
        infr.relabel_using_reviews(rectify=True)

        edge_delta_df = infr.match_state_delta(old='annotmatch', new='all')
        name_delta_df = infr.get_wbia_name_delta()
        name_stats_df = infr.name_group_stats()
        name_delta_stats_df = infr.wbia_name_group_delta_info()

        info = infr.wbia_delta_info(edge_delta_df, name_delta_df)
        import utool

        with utool.embed_on_exception_context:
            assert (
                info['num_edges_added'] == info['num_edges_added_to_am']
            ), 'should never happen when staging is moving into annotmatch'

        msg = ut.codeblock(
            """
            Are you sure this is correct?
            strictly split {num_old_split} into {num_new_split} names.
            strictly merged {num_old_merge} into {num_new_merge} names.
            hybrid split/merged {num_old_hybrid} into {num_new_hybrid} names.
            #annot_names_changed={num_annots_with_names_changed}
            #edges_added={num_edges_added_to_am}
            #edges_modified={num_edges_modified}
            #inconsistent:consistent={num_inconsistent}:{num_consistent}
            #redundant:nonredundant={num_pos_redun}:{num_non_pos_redun}
            """
        ).format(**info)
        # tags_modified={num_changed_tags}
        # tags+decisions_modified={num_changed_decision_and_tags}
        # decision_modified={num_changed_decision}

        lines = []
        print_ = lines.append
        print_('=================')
        print_(msg)
        pdkw = dict(float_format='%.2f')
        print_(
            '---DATAFRAME\nname_delta_stats_df =\n'
            + name_delta_stats_df.to_string(**pdkw)
        )
        print_('---DATAFRAME\nname_stats_df =\n' + name_stats_df.to_string(**pdkw))

        pdkw = dict(max_rows=len(name_delta_df) + 1)
        print_('internal_feedback = ' + ut.repr2(infr.internal_feedback, nl=1))
        print_('---DATAFRAME\nname_delta_df =\n' + name_delta_df.to_string(**pdkw))
        pdkw = dict(max_rows=len(edge_delta_df) + 1)
        print_('---DATAFRAME\nedge_delta_df =\n' + edge_delta_df.to_string(**pdkw))

        print('\n'.join(lines))

        if not gt.are_you_sure(msg=msg):
            raise Exception('Cancel')
        print('\n'.join(lines))

        # Write to staging then write to annotmatch and names
        self.infr.write_wbia_staging_feedback()
        self.infr.write_wbia_annotmatch_feedback(edge_delta_df)
        self.infr.write_wbia_name_assignment(name_delta_df)
        gt.user_info(self, 'Name Change Complete')

    def sizeHint(self):
        return QtCore.QSize(1100, 500)

    def closeEvent(self, event):
        abstract_interaction.unregister_interaction(self)
        super(AnnotGraphWidget, self).closeEvent(event)

    def debug_annot_review_state(self):
        import pandas as pd

        pd.options.display.max_columns = 40
        pd.options.display.width = 160
        ibs = self.infr.ibs
        aid = 30

        df = pd.DataFrame.from_dict(self.infr.graph.edge[aid], orient='index')
        df['aid1'] = aid
        df['aid2'] = df.index.values
        df.set_index(['aid1', 'aid2'], inplace=True)
        print('Graph Edges')
        print(df)

        db = ibs.staging
        rowids = ut.flatten(ibs.get_review_rowids_from_single([aid]))
        tablename = 'reviews'
        exclude_columns = 'review_user_confidence review_user_identity'.split(' ')
        df = db.get_table_as_pandas(tablename, rowids, exclude_columns=exclude_columns)
        df = df.rename(columns={'annot_1_rowid': 'aid1', 'annot_2_rowid': 'aid2'})
        df.set_index(['aid1', 'aid2'], inplace=True)
        print('Staging')
        print(df)

        db = ibs.db
        rowids = ut.flatten(ibs.get_annotmatch_rowids_from_aid([aid]))
        tablename = 'annotmatch'
        exclude_columns = 'annotmatch_confidence annotmatch_posixtime_modified annotmatch_reviewer'.split(
            ' '
        )
        df = db.get_table_as_pandas(tablename, rowids, exclude_columns=exclude_columns)
        df = df.rename(columns={'annot_rowid1': 'aid1', 'annot_rowid2': 'aid2'})
        df.set_index(['aid1', 'aid2'], inplace=True)
        print('Annot Match')
        print(df)

    def print_info(self):
        print('[graph] print_info')
        print('external_feedback = ' + ut.repr2(self.infr.external_feedback, nl=1))
        print('internal_feedback = ' + ut.repr2(self.infr.internal_feedback, nl=1))
        infr = self.infr
        print('infr = %r' % (infr,))
        if infr is not None and infr.graph is not None:
            print(ut.repr3(ut.graph_info(infr.simplify_graph())))

    def embed(self):
        infr = self.infr  # NOQA
        ibs = infr.ibs  # NOQA
        graph = infr.graph  # NOQA
        import utool

        utool.embed()

    def expand_image_and_names(self):
        # call get_name_image_closure()
        ibs = self.infr.ibs
        aids = self.infr.aids
        annots = ibs.annots(aids)
        aids = annots.get_name_image_closure()
        nids = ibs.get_annot_nids(aids)
        import wbia

        new_infr = wbia.AnnotInference(ibs, aids, nids, verbose=self.infr.verbose)
        new_infr.initialize_graph()
        self.infr = new_infr
        self.init_inference()


class EdgeAPIHelper(object):
    def __init__(self, infr):
        self.infr = infr
        self.graph = infr.graph
        self.ibs = infr.ibs

    def make_partial_edge_headers(self):
        """
        These are partial api headers meant to augment edge headers
        """
        custom_edge_props = [
            # TODO: allow user to specify things like hardness / failed / passed or
            # whatever
            'priority',
            'maybe_error',
            'failed',
            'hardness',
            'normscore',
        ]

        edge_col_name_list = [
            # 'index',
            'thumb1',
            'thumb2',
            'match_thumb',
            'inference',
            'review',
            'score',
            'rank',
            'tags',
            'timedelta',
            'kmdist',
            'speed',
        ]
        edge_col_name_list.extend(custom_edge_props)
        edge_col_name_list += [
            'cc_size1',
            'cc_size2',
            'aid1',
            'aid2',
            # 'data',
        ]

        col_getter_dict = {
            'data': self.get_edge_data,
            'timedelta': self.get_edge_timedelta,
            'speed': self.get_edge_speed,
            'kmdist': self.get_edge_kmdist,
            'inference': self.get_inference_text,
            'review': self.get_review_text,
            'score': self.edge_attr_getter('score'),
            'rank': self.edge_attr_getter('rank', -1),
            'tags': self.get_pair_tags,
            'cc_size1': lambda edge: self.get_num_other(edge[0]),
            'cc_size2': lambda edge: self.get_num_other(edge[1]),
            'thumb1': self.ibs.get_annot_chip_thumbtup,
            'thumb2': self.ibs.get_annot_chip_thumbtup,
            'match_thumb': self.get_match_thumbtup,
        }
        for name in custom_edge_props:
            col_getter_dict[name] = self.edge_attr_getter(name)

        col_ider_dict = {name: ('aid1', 'aid2') for name in col_getter_dict.keys()}
        # col_ider_dict = ({
        col_ider_dict.update({'thumb1': 'aid1', 'thumb2': 'aid2'})

        col_types_dict = {
            'rank': int,
            'score': float,
            'timedelta': float,
            'speed': float,
            'thumb1': 'PIXMAP',
            'thumb2': 'PIXMAP',
            'match_thumb': 'PIXMAP',
            'cc_size1': int,
            'cc_size2': int,
        }

        col_display_role_func_dict = {
            'timedelta': ut.partial(ut.get_posix_timedelta_str, year=True, approx=2),
            'speed': lambda speed: '%.2f km/h' % (speed,),
            'kmdist': lambda speed: '%.2f km' % (speed,),
        }

        col_bgrole_dict = {
            'inference': self.get_inference_bgrole,
            'review': self.get_review_bgrole,
        }

        col_width_dict = {
            'index': 42,
            'aid1': 50,
            'aid2': 50,
            'cc_size1': 80,
            'cc_size2': 80,
            'score': 65,
            'rank': 42,
            # 'timedelta': 65,
        }

        partial_headers = {
            'edge_col_name_list': edge_col_name_list,
            'col_getter_dict': col_getter_dict,
            'col_ider_dict': col_ider_dict,
            'col_ider_dict': col_ider_dict,
            'col_types_dict': col_types_dict,
            'col_display_role_func_dict': col_display_role_func_dict,
            'col_bgrole_dict': col_bgrole_dict,
            'col_width_dict': col_width_dict,
        }
        return partial_headers

    def get_edge_timedelta(self, edge):
        self.edge_assert(edge)
        return self.ibs.get_unflat_annots_timedelta_list([edge])[0][0]

    def get_edge_speed(self, edge):
        self.edge_assert(edge)
        return self.ibs.get_unflat_annots_speeds_list2([edge])[0][0]

    def get_edge_kmdist(self, edge):
        self.edge_assert(edge)
        return self.ibs.get_unflat_annots_kmdists_list([edge])[0][0]

    def get_edge_data(self, edge):
        aid1, aid2 = edge
        attrs = self.graph.get_edge_data(aid1, aid2).copy()
        remove_attrs = self.infr.visual_edge_attrs + [
            'rank',
            'evidence_decision',
            'score',
        ]
        try:
            remove_attrs.remove('style')
        except ValueError:
            pass
        ut.delete_dict_keys(attrs, remove_attrs)
        attrs = {k: v for k, v in attrs.items() if v is not None}
        return ut.repr2(attrs, precision=2, explicit=True, nobr=True)

    def edge_attr_getter(self, attr, default=None):
        def get_edge_attr(edge):
            data = self.graph.get_edge_data(*edge)
            return data.get(attr, default)

        return get_edge_attr

    def get_pair_tags(self, edge):
        aid1, aid2 = edge
        self.edge_assert(edge)
        # ibs = self.infr.ibs
        # # FIXME: use graph properties instead
        # am_rowids = ibs.get_annotmatch_rowid_from_undirected_superkey(
        #     [aid1], [aid2])
        # tag_text = ibs.get_annotmatch_tag_text(am_rowids)[0]
        tags = self.infr.graph.get_edge_data(*edge).get('tags', [])
        if tags is None:
            tag_text = ''
        else:
            tag_text = ';'.join(tags)
        return str(tag_text)

    def edge_assert(self, edge):
        aid1, aid2 = edge
        assert not ut.isiterable(aid1), 'aid1=%r, aid2=%r' % (aid1, aid2)
        assert not ut.isiterable(aid2), 'aid1=%r, aid2=%r' % (aid1, aid2)

    def _get_inference_info(self, edge):
        aid1, aid2 = edge
        node_dict = ut.nx_node_dict(self.graph)
        nid1 = node_dict[aid1]['name_label']
        nid2 = node_dict[aid2]['name_label']
        # nid1, nid2 = self.infr.node_labels(aid1, aid2)
        data = self.infr.graph.get_edge_data(*edge)
        inferred_state = data.get('inferred_state', None)
        maybe_error = data.get('maybe_error', False)

        if inferred_state is None:
            state = 'unknown'
        elif inferred_state.startswith('inconsistent'):
            state = inferred_state
        else:
            if inferred_state == INCMP:
                state = INCMP
            else:
                inferred_truth = {'same': True, 'diff': False}[inferred_state]
                name_truth = nid1 == nid2
                if name_truth != inferred_truth:
                    state = 'disagree'
                else:
                    state = 'same' if nid1 == nid2 else 'diff'

        text_parts = []

        if maybe_error:
            text_parts.append('(ERROR?)')

        if state == 'inconsistent_external':
            text_parts.append('inconsistent')
        elif state == 'inconsistent_internal':
            text_parts.append('inconsistent')
        else:
            text_parts.append(state)

        if nid1 == nid2:
            text_parts.append(' nid=%r' % (nid1,))
        else:
            text_parts.append(' nids=%r,%r' % (nid1, nid2))

        if state == 'inconsistent_external':
            text_parts.append('(external)')
        elif state == 'inconsistent_internal':
            text_parts.append('(internal)')

        text = ' '.join(text_parts)
        info = (state, text, maybe_error)
        return info

    def get_inference_text(self, edge):
        info = self._get_inference_info(edge)
        state, text, maybe_error = info
        return text

    def get_review_text(self, edge):
        graph = self.infr.graph
        text = graph.get_edge_data(*edge).get('evidence_decision', UNREV)
        return text

    def get_inference_bgrole(self, edge):
        """ Background role for status column """
        state, text, maybe_error = self._get_inference_info(edge)
        if state == 'disagree':
            color = pt.WHITE
        elif state.startswith('inconsistent'):
            color = pt.ORANGE
            if state == 'inconsistent_external':
                lighten_amount = 0.55
                color = pt.lighten_rgb(color, lighten_amount)
            elif not maybe_error:
                lighten_amount = 0.35
                color = pt.lighten_rgb(color, lighten_amount)
        else:
            lighten_amount = 0.35
            truth_colors = self.infr._get_truth_colors()
            if state == 'unknown':
                lighten_amount = 0.7
                color = truth_colors[UNREV]
            else:
                color = truth_colors[POSTV] if state == 'same' else truth_colors[NEGTV]
            # self.graph.get_edge_data(*edge).get('evidence_decision', UNREV)]
            if lighten_amount is not None:
                color = pt.lighten_rgb(color, lighten_amount)
        color = pt.to_base255(color)
        return color

    def get_review_bgrole(self, edge):
        """ Background role for status column """
        data = self.graph.get_edge_data(*edge)
        state = data.get('evidence_decision', UNREV)
        truth_colors = self.infr._get_truth_colors()
        if state == UNREV:
            inference_state, text, maybe_error = self._get_inference_info(edge)
            if inference_state == 'same':
                color = truth_colors[POSTV]
            elif inference_state == 'diff':
                color = truth_colors[NEGTV]
            else:
                color = truth_colors[state]
        else:
            color = truth_colors[state]
        lighten_amount = 0.35
        if state == UNREV:
            lighten_amount = 0.7
        if lighten_amount is not None:
            color = pt.lighten_rgb(color, lighten_amount)
        color = pt.to_base255(color)
        return color

    def get_match_thumbtup(self, edge, thumbsize=None):
        # sibs, qaid2_cm, qaids, daids, index, qreq_=None,
        #                   thumbsize=(128, 128), match_thumbtup_cache={}):
        aid1, aid2 = edge
        try:
            cm, aid1, aid2 = self.infr.lookup_cm(aid1, aid2)
        except KeyError:
            return None
        from wbia.gui import id_review_api

        if cm is None:
            # HACK: check if a PairwiseMatch exists
            match = self.infr.vsone_matches.get((aid1, aid2))
            if match is not None:
                fpath, func, func2 = id_review_api.make_ensure_match_img_nosql_func(
                    self.infr.ibs, match, None
                )
                thumbdat = {
                    'fpath': fpath,
                    'thread_func': func,
                    'main_func': func2,
                }
                return thumbdat
            else:
                return None
        # assert cm.qaid == aid1, 'aids do not aggree'
        # Hacky new way of drawing
        fpath, func, func2 = id_review_api.make_ensure_match_img_nosql_func(
            self.infr.qreq_, cm, aid2
        )
        thumbdat = {
            'fpath': fpath,
            'thread_func': func,
            'main_func': func2,
        }
        return thumbdat

    def get_num_other(self, aid):
        node = aid
        node_to_name_label = nx.get_node_attributes(self.graph, 'name_label')
        name_label = node_to_name_label[node]
        labels = list(node_to_name_label.values())
        return labels.count(name_label)


def make_name_edge_api(infr, review_cfg={}):
    node_to_name = infr.get_node_attrs('name_label')
    name_to_nodes = ut.group_items(node_to_name.keys(), node_to_name.values())

    name_to_edges = {
        name: list(infr.graph.subgraph(nodes).edges())
        for name, nodes in name_to_nodes.items()
    }

    names = list(name_to_edges.keys())
    flat_edges, grouped_edge_idxs = ut.invertible_flatten1(name_to_edges.values())
    grouped_edges = ut.apply_grouping(flat_edges, grouped_edge_idxs)
    n_annots_list = [len(set(ut.flatten(edges))) for edges in grouped_edges]

    nid_col_name_list = ['name_label', 'n_annots', 'n_edges']

    self = EdgeAPIHelper(infr)
    partial_headers = self.make_partial_edge_headers()
    edge_col_name_list = partial_headers['edge_col_name_list']

    col_name_list = nid_col_name_list + edge_col_name_list
    col_level_dict = {}
    for col in nid_col_name_list:
        col_level_dict[col] = 0
    for col in edge_col_name_list:
        col_level_dict[col] = 1
    iders = [
        list(range(len(names))),
        grouped_edge_idxs,
    ]
    col_getter_dict = {
        'name_label': names,
        'n_edges': list(map(len, grouped_edge_idxs)),
        'n_annots': n_annots_list,
        'aid1': ut.take_column(flat_edges, 0),
        'aid2': ut.take_column(flat_edges, 1),
    }

    col_getter_dict.update(partial_headers['col_getter_dict'])

    col_display_role_func_dict = partial_headers['col_display_role_func_dict']
    col_bgrole_dict = partial_headers['col_bgrole_dict']

    col_ider_dict = partial_headers['col_ider_dict']

    def name_bg_role(idx):
        name = names[idx]
        edges = name_to_edges[name]
        color = pt.WHITE
        for edge in edges:
            data = self.infr.graph.get_edge_data(*edge)
            inferred_state = data.get('inferred_state', None)
            if inferred_state is not None and inferred_state.startswith('inconsistent'):
                color = pt.ORANGE
                break
        color = pt.to_base255(color)
        return color

    col_bgrole_dict['name_label'] = name_bg_role
    col_bgrole_dict['n_edges'] = name_bg_role
    col_bgrole_dict['n_annots'] = name_bg_role

    name_api = gt.CustomAPI(
        col_name_list,
        col_ider_dict=col_ider_dict,
        col_types_dict=partial_headers['col_types_dict'],
        col_getter_dict=col_getter_dict,
        col_bgrole_dict=col_bgrole_dict,
        col_display_role_func_dict=col_display_role_func_dict,
        col_width_dict=partial_headers['col_width_dict'],
        get_thumb_size=lambda: 221,
        col_level_dict=col_level_dict,
        iders=iders,
        sortby='n_edges',
        sort_reverse=True
        # sortby='aid1',
    )
    return name_api


def make_edge_api(infr, review_cfg={}):
    """
    TODO:
        mark an edge that would cause an inconsistency
        to be marked as a deeper red.
        These are edges we currently believe to be false
        By default edges should not be red. they should be a light unknown yellow.
        Dark unknown yellow is for noncomparable annotations

    """
    if review_cfg['hack_min_review']:
        assert False
    else:
        edges_and_data = list(infr.edges(data=True))
        if review_cfg['filter_photobombs']:
            edges_and_data = [
                (edge, d)
                for edge, d in edges_and_data
                if 'photobomb' not in d.get('tags')
            ]
        if review_cfg['filter_reviewed']:
            edges_and_data = [
                (edge, d)
                for edge, d in edges_and_data
                if d.get('evidence_decision', UNREV) != UNREV
            ]
        if review_cfg['filter_true_matches']:
            edges_and_data = [
                (edge, d)
                for edge, d in edges_and_data
                if not infr.pos_redun_edge_flag(edge)
            ]
        if review_cfg['filter_false_matches']:
            edges_and_data = [
                (edge, d)
                for edge, d in edges_and_data
                if not infr.neg_redun_edge_flag(*edge)
            ]
        aids1, aids2 = ut.listT(ut.take_column(edges_and_data, 0))
        # aids1, aids2 = infr.get_filtered_edges(review_cfg)

    self = EdgeAPIHelper(infr)
    partial_headers = self.make_partial_edge_headers()
    # from six import next
    # data = next(infr.graph.edges(data=True))[-1]

    # if not DEVELOPER_MODE:
    #    col_name_list.remove('data')
    col_name_list = partial_headers['edge_col_name_list']

    if not review_cfg['show_match_thumb']:
        # FIXME: do one-vs-one scoring instead
        col_name_list.remove('match_thumb')

    col_getter_dict = {
        'index': np.arange(len(aids1)),
        'aid1': aids1,
        'aid2': aids2,
    }
    col_getter_dict.update(partial_headers['col_getter_dict'])

    edge_api = gt.CustomAPI(
        col_name_list,
        col_ider_dict=partial_headers['col_ider_dict'],
        col_types_dict=partial_headers['col_types_dict'],
        col_getter_dict=col_getter_dict,
        col_bgrole_dict=partial_headers['col_bgrole_dict'],
        col_display_role_func_dict=partial_headers['col_display_role_func_dict'],
        col_width_dict=partial_headers['col_width_dict'],
        get_thumb_size=lambda: 221,
        sortby='score',
        # sortby='aid1',
        sort_reverse=True,
    )
    # api = edge_api
    return edge_api


def make_node_api(infr):
    aids = sorted(list(infr.graph.nodes()))
    col_name_list = ['aid', 'thumb', 'name_label']

    def get_node_data(aid):
        node_dict = ut.nx_node_dict(infr.graph)
        data = node_dict[aid].copy()
        ut.delete_dict_keys(data, infr.visual_node_attrs)
        return ut.repr2(data, precision=2)

    node_dict = ut.nx_node_dict(infr.graph)
    col_getter_dict = {
        'aid': np.array(aids),
        'data': get_node_data,
        'thumb': infr.ibs.get_annot_chip_thumbtup,
        'name_label': lambda node: node_dict[node].get('name_label', None),
    }
    col_ider_dict = {
        'thumb': 'aid',
        'data': 'aid',
        'name_label': 'aid',
    }
    col_types_dict = {
        'thumb': 'PIXMAP',
    }
    node_api = gt.CustomAPI(
        col_name_list,
        col_ider_dict=col_ider_dict,
        col_types_dict=col_types_dict,
        col_getter_dict=col_getter_dict,
        sortby='aid',
        sort_reverse=False,
    )
    return node_api


def make_name_node_api(infr, review_cfg={}):
    # TODO: only make this API if the tab is clicked
    node_to_name = infr.get_node_attrs('name_label')
    name_to_nodes = ut.group_items(node_to_name.keys(), node_to_name.values())

    self = EdgeAPIHelper(infr)

    names = list(name_to_nodes.keys())
    flat_aids, grouped_aid_idxs = ut.invertible_flatten1(name_to_nodes.values())

    col_name_list = ['name_label', 'n_annots', 'aid', 'thumb', 'name_label2']
    col_level_dict = {
        'name_label': 0,
        'n_annots': 0,
        'aid': 1,
        'thumb': 1,
        'name_label2': 1,
    }
    iders = [
        list(range(len(names))),
        grouped_aid_idxs,
    ]

    node_dict = ut.nx_node_dict(infr.graph)
    col_getter_dict = {
        'thumb': infr.ibs.get_annot_chip_thumbtup,
        'name_label': names,
        'n_annots': list(map(len, grouped_aid_idxs)),
        'aid': flat_aids,
        'name_label2': lambda node: node_dict[node].get('name_label', None),
    }
    col_ider_dict = {
        'thumb': 'aid',
        'name_label2': 'aid',
    }
    col_types_dict = {
        'thumb': 'PIXMAP',
    }

    col_bgrole_dict = {
        'inference': self.get_inference_bgrole,
        'review': self.get_review_bgrole,
    }

    name_api = gt.CustomAPI(
        col_name_list,
        iders=iders,
        # col_types_dict=col_types_dict,
        col_ider_dict=col_ider_dict,
        col_getter_dict=col_getter_dict,
        col_types_dict=col_types_dict,
        col_bgrole_dict=col_bgrole_dict,
        # col_display_role_func_dict=col_display_role_func_dict,
        # col_width_dict=col_width_dict,
        get_thumb_size=lambda: 221,
        col_level_dict=col_level_dict,
        sortby='n_annots',
        sort_reverse=True,
    )
    return name_api


def make_qt_graph_review(qreq_, cm_list):
    r"""
    CommandLine:
        wbia make_qt_graph_review --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_graph2 import *  # NOQA
        >>> import wbia.guitool as gt
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> qreq_ = wbia.testdata_qreq_(defaultdb=defaultdb)
        >>> cm_list = qreq_.execute()
        >>> gt.ensure_qtapp()
        >>> win = make_qt_graph_review(qreq_, cm_list)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=win, freq=10)
    """
    gt.ensure_qtapp()
    import wbia

    infr = wbia.AnnotInference.from_qreq_(qreq_, cm_list)

    gt.ensure_qtapp()
    print('infr = %r' % (infr,))
    win = AnnotGraphWidget(infr=infr, use_image=False, init_mode='review')
    abstract_interaction.register_interaction(win)
    win.show()
    return win


def make_qt_graph_interface(
    ibs, aids=None, nids=None, gids=None, init_mode='review', graph_tab=False
):
    r"""
    CommandLine:
        wbia make_qt_graph_interface --dbdir ~/lev/media/hdd/work/WWF_Lynx/ --show --nids=281 --graphtab
        wbia make_qt_graph_interface --dbdir ~/lev/media/hdd/work/WWF_Lynx/ --show --gids=2289 --graphtab

        wbia make_qt_graph_interface --dbdir ~/lev/media/hdd/work/WWF_Lynx/ --show --graph-tab --aids=2587,2398
        wbia make_qt_graph_interface --show
        wbia make_qt_graph_interface --show --db PZ_PB_RF_TRAIN

        wbia make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9 --graph-tab
        wbia make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9
        wbia make_qt_graph_interface --show

        wbia make_qt_graph_interface --show --db RotanTurtles --aids=610,716

        wbia make_qt_graph_interface --db LEWA_splits --nids=1 --show --sample

        wbia make_qt_graph_interface --db PZ_MTEST --nids=1 --show --init-mode=rereview

        wbia make_qt_graph_interface --dbdir=~/lev/media/danger/GGR/GGR-IBEIS --nids=2300 --show
        wbia make_qt_graph_interface --dbdir=~/lev/media/danger/GGR/GGR-IBEIS --nids=4617 --show

        # unmount
        fusermount -u ~/lev
        # mount
        sshfs -o idmap=user lev:/ ~/lev
        wbia make_qt_graph_interface --dbdir=/home/joncrall/lev/media/hdd/work/EWT_Cheetahs --show -a default:view=right;frontright;backright

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.viz_graph2 import *  # NOQA
        >>> import wbia.guitool as gt
        >>> import wbia
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
        >>> aids = ut.get_argval('--aids', type_=list, default=None)
        >>> nids = ut.get_argval('--nids', type_=list, default=None)
        >>> gids = ut.get_argval('--gids', type_=list, default=None)
        >>> init_mode = ut.get_argval('--init_mode', default='review')
        >>> graph_tab = ut.get_argflag('--graph-tab')
        >>> gt.ensure_qtapp()
        >>> win = make_qt_graph_interface(ibs, aids, nids, gids, init_mode, graph_tab)
        >>> ut.quit_if_noshow()
        >>> gt.qtapp_loop(qwin=win, freq=10)
    """
    print('[qt_graph] make_qt_graph_interface init()')
    print('[qt_graph] nids = %s' % (ut.trunc_repr(nids),))
    print('[qt_graph] aids = %s' % (ut.trunc_repr(aids),))
    print('[qt_graph] gids = %s' % (ut.trunc_repr(gids),))
    if gids is not None:
        nids = ut.unique(ut.flatten(ibs.get_image_nids(gids)))
    if nids is not None and aids is None:
        aids = ut.flatten(ibs.get_name_aids(nids))
    if aids is None:
        # import wbia
        # aids = wbia.testdata_aids(ibs=ibs)
        # ['right', 'frontright', 'backright']
        aids = ibs.get_valid_aids()
        # aids = ibs.filter_annots_general(max_pername=2, min_pername=2)
        # aids = ibs.filter_annots_general(view=['right', 'frontright', 'backright'])
        # [0:20]
    if ut.get_argflag('--sample'):
        rng = np.random.RandomState(42)
        aids = rng.choice(aids, 30, replace=False)

    print('make_qt_graph_interface aids = %r' % (aids,))
    nids = ibs.get_annot_name_rowids(aids)
    import wbia

    # infr = wbia.AnnotInference(ibs, aids, nids, verbose=ut.VERBOSE)
    infr = wbia.AnnotInference(ibs, aids, nids, verbose=5)
    infr.initialize_graph()

    gt.ensure_qtapp()
    print('infr = %r' % (infr,))
    win = AnnotGraphWidget(infr=infr, use_image=False, init_mode=init_mode)
    abstract_interaction.register_interaction(win)
    win.show()

    if graph_tab:
        index = win.graph_tab_widget.indexOf(win.graph_tab)
        win.graph_tab_widget.setCurrentIndex(index)
        print(
            'win.graph_widget.use_image_cb.setChecked = %r'
            % (win.graph_widget.use_image_cb.setChecked,)
        )
        win.graph_widget.use_image_cb.setChecked(True)

    # match = ut.get_argval('--match', type_=list, default=None)
    # print('match = %r' % (match,))
    # if match:
    #     win.mark_pair_state([match], POSTV)

    if False:
        win.expand_image_and_names()

    return win


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.viz.viz_graph2
        python -m wbia.viz.viz_graph2 --allexamples
        wbia make_qt_graph_interface --show --aids=1,2,3,4,5,6,7,8,9 --graph

        python -m wbia.viz.viz_graph2 make_qt_graph_interface --show --aids=1,2,3,4,5,6,7 --graph --match=1,4 --nomatch=3,1,5,7
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
