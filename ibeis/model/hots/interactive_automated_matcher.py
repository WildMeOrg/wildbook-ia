from __future__ import absolute_import, division, print_function
import six
from six.moves import builtins  # NOQA
import utool as ut
#import numpy as np
#import sys
#from ibeis.model.hots import automated_oracle as ao
#from ibeis.model.hots import automated_helpers as ah
#from ibeis.model.hots import special_query
import guitool
from ibeis.model.hots import automated_matcher as automatch
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[qtinc]')


INC_LOOP_BASE = guitool.__PYQT__.QtCore.QObject


class IncQueryHarness(INC_LOOP_BASE):
    next_query_signal = guitool.signal_()
    name_decision_signal = guitool.signal_(list)

    def __init__(self):
        INC_LOOP_BASE.__init__(self)
        self.inc_query_gen = None
        self.ibs = None
        self.dry = False
        self.interactive = True
        # connect signals to slots
        self.next_query_signal.connect(self.next_query_slot)
        self.name_decision_signal.connect(self.name_decision_slot)

    def request_nonblocking_inc_query(self, ibs, qaid_list, daid_list):
        self.ibs = ibs

        def emit_name_decision(sorted_aids):
            """
            Weird we need to put emits inside this closure scope otherwise
            fe get a segfault. Thanks PyQt
            """
            #print(sorted_aids)
            #print(';)')
            self.name_decision_signal.emit(sorted_aids)

        def emit_next_query():
            """
            Weird we need to put emits inside this closure scope otherwise
            fe get a segfault. Thanks PyQt
            """
            self.next_query_signal.emit()

        callbacks = {
            'next_query_callback': emit_next_query,
            'name_decision_callback': emit_name_decision,
            #'next_query_callback': self.next_query_signal.emit,
            #'name_decision_callback': self.name_decision_signal.emit,
            #'try_decision_callback': self.try_decision_signal.emit
        }
        self.inc_query_gen = automatch.generate_incremental_queries(
            ibs, qaid_list, daid_list, callbacks=callbacks)
        # Call the first query
        callbacks['next_query_callback']()

    @guitool.slot_()
    def next_query_slot(self):
        """
        callback used when all interactions are completed.
        Generates the next incremental query and then tries
        the automatic interactions
        """
        try:
            dry = self.dry
            interactive = self.interactive
            item = six.next(self.inc_query_gen)
            (ibs, qres, qreq_, choicetup, metatup, callbacks, threshold) = item
            self.choicetup = choicetup
            self.qres      = qres
            self.qreq_     = qreq_
            self.metatup   = metatup
            self.callbacks = callbacks
            self.threshold = threshold
            automatch.try_automatic_decision(ibs, qres, qreq_, choicetup,
                                             threshold, interactive=interactive,
                                             metatup=metatup, dry=dry,
                                             callbacks=callbacks)
        except StopIteration:
            print('NO MORE QUERIES. CLOSE DOWN WINDOWS AND DISPLAY DONE MESSAGE')
            pass

    @guitool.slot_(list)
    def name_decision_slot(self, sorted_aids):
        print('[QT] name_decision_slot')
        try:
            ibs = self.ibs
            choicetup   = self.choicetup
            qres        = self.qres
            qreq_       = self.qreq_
            metatup     = self.metatup
            callbacks   = self.callbacks
            threshold   = self.threshold
            interactive = self.interactive
            dry         = self.dry
            if sorted_aids is None or len(sorted_aids) == 0:
                name = None
            else:
                name = ibs.get_annot_names(sorted_aids[0])
            automatch.make_name_decision(name, choicetup, ibs, qres, qreq_,
                                         threshold, interactive=interactive,
                                         metatup=metatup, dry=dry,
                                         callbacks=callbacks)
        except StopIteration:
            print('NO MORE QUERIES. CLOSE DOWN WINDOWS AND DISPLAY DONE MESSAGE')
            pass


def incremental_test_qt(ibs, num_initial=0):
    """
    CommandLine:
        python -m ibeis.model.hots.interactive_automated_matcher --test-incremental_test_qt

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.all_imports import *  # NOQA
        >>> from ibeis.model.hots.interactive_automated_matcher import *  # NOQA
        >>> main_locals = ibeis.main(db='testdb1')
        >>> ibs = main_locals['ibs']
        >>> #num_initial = 0
        >>> num_initial = 0
        >>> incremental_test_qt(ibs, num_initial)
        >>> execstr = ibeis.main_loop(main_locals)
        >>> print(execstr)
    """
    qaid_list = ibs.get_valid_aids()
    daid_list = ibs.get_valid_aids()
    self = IncQueryHarness()
    self = self.request_nonblocking_inc_query(ibs, qaid_list, daid_list)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.interactive_automated_matcher
        python -m ibeis.model.hots.interactive_automated_matcher --allexamples
        python -m ibeis.model.hots.interactive_automated_matcher --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
