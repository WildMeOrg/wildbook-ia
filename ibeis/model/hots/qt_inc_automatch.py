"""

utprof.py -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:3 --num-init 5000 --stateful-query
utprof.py -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:0
python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:0

TODO:
    * spatially constrained matching
    * benchmarks for reconstruction
      vs addition.
        - accuracy
           compares percent of correct
           error = (qfx2_idx_reindex != qfx2_idx_append).sum()
        - time
           compares time of
              - addition
              - reindex
              - multindex search
              - regular search
           error = (qfx2_idx_reindex != qfx2_idx_append).sum()

#import numpy as np
#import sys
#from ibeis.model.hots import automated_oracle as ao
#from ibeis.model.hots import automated_helpers as ah
#from ibeis.model.hots import special_query   *
"""
from __future__ import absolute_import, division, print_function
import six
from six.moves import builtins  # NOQA
import utool as ut
from guitool.__PYQT__ import QtCore
import guitool
from ibeis.model.hots import automated_matcher as automatch
from ibeis.model.hots import automated_helpers as ah
# Can't inject print into this module  otherwise bad things happen
# with qt signals and slots
ut.noinject(__name__, '[qtinc]')
#print, print_, printDBG, rrr, profile = ut.inject(__name__, '[qtinc]')


INC_LOOP_BASE = guitool.__PYQT__.QtCore.QObject


def test_inc_query(ibs_gt, num_initial=0):
    """
    entry point for interactive query tests
    test_interactive_incremental_queries

    Args:
        ibs       (list) : IBEISController object
        qaid_list (list) : list of annotation-ids to query

    CommandLine:
        python dev.py -t inc --db PZ_MTEST --qaid 1:30:3 --cmd
        python dev.py --db PZ_MTEST --allgt --cmd
        python dev.py --db PZ_MTEST --allgt -t inc
        python dev.py --db PZ_MTEST --allgt -t inc


    CommandLine:
        python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:0  --interact-after 444440 --noqcache
        python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:1  --interact-after 444440 --noqcache

        python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:0
        python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:1
        python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:2
        python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:3

        utprof.py -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:3 --ninit 5000
        utprof.py -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:0

        # Writes out test script
        python -c "import utool as ut; ut.write_modscript_alias('Tinc.sh', 'ibeis.model.hots.qt_inc_automatch')"

        sh Tinc.sh --test-test_inc_query:0
        sh Tinc.sh --test-test_inc_query:1
        sh Tinc.sh --test-test_inc_query:2
        sh Tinc.sh --test-test_inc_query:3

        utprof.py -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:3 --num-init 5000

        sh Tinc.sh --test-test_inc_query:0 --ninit 10
        sh Tinc.sh --test-test_inc_query:0 --ninit 10 --verbose-debug --verbose-helpful

        python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:0 --ia 10

        # Runs into a merge case
        python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:0 --ia 30

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('testdb1')
        >>> test_inc_query(ibs_gt)

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('PZ_MTEST')
        >>> test_inc_query(ibs_gt)

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('GZ_ALL')
        >>> test_inc_query(ibs_gt)

    Example3:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs_gt = ibeis.opendb('PZ_Master0')
        >>> test_inc_query(ibs_gt)

    """
    from ibeis import main_module
    main_module._preload()
    guitool.ensure_qtapp()
    num_initial
    self = IncQueryHarness()
    num_initial = ut.get_argval(('--num-initial', '--num-init', '--ninit'), int, 0)
    interactive_after = ut.get_argval(('--interactive-after', '--ia'), type_=int, default=None)
    # Add information to an empty database from a groundtruth database
    ibs, aid_list1, aid1_to_aid2 = ah.setup_incremental_test(ibs_gt)
    if True or interactive_after is not None:
        back = main_module._init_gui()
        back.connect_ibeis_control(ibs)
    else:
        back = None
    interactive = self.test_incremental_query(
        ibs_gt, ibs, aid_list1, aid1_to_aid2,
        interactive_after=interactive_after, num_initial=num_initial, back=back)
    if back is None and interactive_after is None and interactive:
        # need to startup backend
        back = main_module._init_gui()
        back.connect_ibeis_control(ibs)
    if interactive or interactive_after is not None:
        guitool.qtapp_loop()


def incremental_test_qt(ibs, num_initial=0):
    """
    CommandLine:
        python -m ibeis.model.hots.qt_inc_automatch --test-incremental_test_qt

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.qt_inc_automatch import *  # NOQA
        >>> main_locals = ibeis.main(db='testdb1')
        >>> ibs = main_locals['ibs']
        >>> back = main_locals['back']
        >>> #num_initial = 0
        >>> num_initial = 0
        >>> incremental_test_qt(ibs, num_initial)
        >>> pt.present()
        >>> execstr = ibeis.main_loop(main_locals)
        >>> print(execstr)
    """
    import ibeis.constants as const
    species = const.Species.ZEB_PLAIN
    qaid_list = ibs.get_valid_aids(species=species)
    #daid_list = ibs.get_valid_aids()
    exec_interactive_incremental_queries(ibs, qaid_list)


def exec_interactive_incremental_queries(ibs, qaid_list, back=None):
    self = IncQueryHarness()
    self = self.begin_incremental_query(ibs, qaid_list, back=back)


class IncQueryHarness(INC_LOOP_BASE):
    """
    Provides incremental query with a way to work around hitting the recusion
    limit. FIXME: currently it doesnt do this.

    TODO: maybe abstract this into a interuptable loop harness
    """
    next_query_signal = guitool.signal_()
    name_decision_signal = guitool.signal_(list)
    exemplar_decision_signal = guitool.signal_(bool)

    def __init__(self):
        INC_LOOP_BASE.__init__(self)
        self.inc_query_gen = None
        self.ibs = None
        # connect signals to slots
        self.next_query_signal.connect(self.next_query_slot)
        self.name_decision_signal.connect(self.name_decision_slot)
        self.exemplar_decision_signal.connect(self.exemplar_decision_slot)
        # Weird we need to put emits inside this closure scope otherwise
        # we get a segfault. Thanks PyQt
        def next_query_callback():
            self.next_query_signal.emit()

        def name_decision_callback(chosen_names):
            self.name_decision_signal.emit(chosen_names)

        def exemplar_decision_callback(exemplar_decision):
            self.exemplar_decision_signal.emit(exemplar_decision)

        self.incinfo = {
            'next_query_callback': next_query_callback,
            'name_decision_callback': name_decision_callback,
            'exemplar_decision_callback': exemplar_decision_callback,
            'metatup': None,
            'use_oracle': False,
            'interactive': True,
            'count': 0,
            'fnum': 512,
            #'next_query_callback': self.next_query_signal.emit,
            #'name_decision_callback': self.name_decision_signal.emit,
            #'try_decision_callback': self.try_decision_signal.emit
        }

    def setup_back_callbacks(self, back, incinfo):
        import functools
        from ibeis.gui.guiheaders import NAMES_TREE  # ADD AS NEEDED
        back.front.set_table_tab(NAMES_TREE)
        update_callback = functools.partial(back.front.update_tables, tblnames=[NAMES_TREE])
        incinfo['update_callback'] = update_callback
        incinfo['finish_callback'] = functools.partial(back.user_info, msg='Finished Query')

    def test_incremental_query(self, ibs_gt, ibs, aid_list1, aid1_to_aid2,
                               num_initial=0, interactive_after=None, back=None):
        """
        Adds and queries new annotations one at a time with oracle guidance
        """
        incinfo = self.incinfo
        if back is not None:
            self.setup_back_callbacks(back, incinfo)
        self.ibs = ibs
        incinfo['nTotal'] = len(aid_list1)
        #incinfo['nTotal'] = len(aid_list1)
        # Create test query generator
        next_query_callback = self.incinfo['next_query_callback']  # NOQA
        del self.incinfo['next_query_callback']
        self.inc_query_gen = automatch.test_generate_incremental_queries(
            ibs_gt, ibs, aid_list1, aid1_to_aid2, num_initial, incinfo)
        # When in interactive mode it seems like the stack never gets out of hand
        # but if the oracle is allowed to make decisions and emit signals like
        # the user then we get into a maximum recursion limit.
        incinfo['interactive'] = False
        incinfo['use_oracle'] = True
        hack_run_name_decision = False
        with ut.Timer('test_incremental_query'):
            for item  in self.inc_query_gen:
                (ibs, qres, qreq_, incinfo) = item
                # update incinfo
                self.qres      = qres
                self.qreq_     = qreq_
                self.incinfo = incinfo
                incinfo['count'] += 1
                if incinfo.get('PLEASE_STOP', False):
                    print('PLEASE_STOP triggered')
                    interactive_after = incinfo['count'] - 1
                    incinfo['next_query_callback'] = next_query_callback
                    incinfo['use_oracle'] = False
                    incinfo['interactive'] = True
                    hack_run_name_decision = True
                    break
                automatch.run_until_name_decision_signal(ibs, qres, qreq_, incinfo=incinfo)
                #ut.embed()
                if interactive_after is not None and incinfo['count'] > interactive_after:
                    # stop the automated queries and start interaction
                    print('Interactive after triggered')
                    incinfo['interactive'] = True
                    break

        # BEGIN INTERACTIVE PART
        # (this does nothing if inc_query_gen is exhausted)
        # need to fix the incinfo dictionary
        incinfo['next_query_callback'] = next_query_callback
        incinfo['use_oracle'] = False
        #incinfo['metatup'] = None
        if hack_run_name_decision:
            # need to rn this so PLEASE_STOP can trigger user interaction
            automatch.run_until_name_decision_signal(ibs, qres, qreq_, incinfo=incinfo)
        else:
            incinfo['next_query_callback']()
        return incinfo['interactive']

    def begin_incremental_query(self, ibs, qaid_list, back=None):
        """
        runs incremental query in live mode
        """
        self.ibs = ibs
        incinfo = self.incinfo
        if back is not None:
            self.setup_back_callbacks(back, incinfo)
        incinfo['nTotal'] = len(qaid_list)
        # Create live query generator
        self.inc_query_gen = automatch.generate_incremental_queries(
            ibs, qaid_list, incinfo=incinfo)
        # Call the first query
        incinfo['next_query_callback']()

    #@guitool.slot_()
    @QtCore.pyqtSlot()
    def next_query_slot(self):
        """
        callback used when all interactions are completed.
        Generates the next incremental query and then tries
        the automatic interactions
        """
        try:
            # Generate the query result
            item = six.next(self.inc_query_gen)
            (ibs, qres, qreq_, incinfo) = item
            # update incinfo
            self.qres      = qres
            self.qreq_     = qreq_
            self.incinfo = incinfo
            incinfo['count'] += 1
            automatch.run_until_name_decision_signal(ibs, qres, qreq_, incinfo=incinfo)
            #import plottool as pt
            #pt.present()

        except StopIteration:
            #TODO: close this figure
            # incinfo['fnum']
            print('NO MORE QUERIES. CLOSE DOWN WINDOWS AND DISPLAY DONE MESSAGE')
            import plottool as pt
            fig1 = pt.figure(fnum=511)
            fig2 = pt.figure(fnum=512)
            pt.close_figure(fig1)
            pt.close_figure(fig2)
            if 'finish_callback' in self.incinfo:
                self.incinfo['update_callback']()
            if 'finish_callback' in self.incinfo:
                self.incinfo['finish_callback']()
            pass

    #@guitool.slot_(list)
    @QtCore.pyqtSlot(list)
    def name_decision_slot(self, chosen_names):
        """
        the name decision signal was emited
        """
        #print('[QT] name_decision_slot')
        ibs = self.ibs
        qres        = self.qres
        qreq_       = self.qreq_
        incinfo     = self.incinfo
        automatch.exec_name_decision_and_continue(
            chosen_names, ibs, qres, qreq_, incinfo=incinfo)

    #@guitool.slot_(bool)
    @QtCore.pyqtSlot(bool)
    def exemplar_decision_slot(self, exemplar_decision):
        incinfo = self.incinfo
        ibs = self.ibs
        qres        = self.qres
        qreq_       = self.qreq_
        incinfo     = self.incinfo
        automatch.exec_exemplar_decision_and_continue(
            exemplar_decision, ibs, qres, qreq_, incinfo=incinfo)
        #automatch.exec_name_exemplar_and_continue(name, choicetup, ibs, qres,
        #                                          qreq_, incinfo=incinfo)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.qt_inc_automatch
        python -m ibeis.model.hots.qt_inc_automatch --allexamples
        python -m ibeis.model.hots.qt_inc_automatch --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
