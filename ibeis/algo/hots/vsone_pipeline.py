# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
# import numpy as np
# import vtool as vt
# from vtool import matching
import utool as ut
# from six.moves import zip, range
print, rrr, profile = ut.inject2(__name__)


def prepare_annot_pairs(ibs, qaids, daids, qconfig2_, dconfig2_):
    # Prepare lazy attributes for annotations
    qannot_cfg = ibs.depc.stacked_config(None, 'featweight', qconfig2_)
    dannot_cfg = ibs.depc.stacked_config(None, 'featweight', dconfig2_)

    unique_qaids = set(qaids)
    unique_daids = set(daids)

    # Determine a unique set of annots per config
    configured_aids = ut.ddict(set)
    configured_aids[qannot_cfg].update(unique_qaids)
    configured_aids[dannot_cfg].update(unique_daids)

    # Make efficient annot-object representation
    configured_obj_annots = {}
    for config, aids in configured_aids.items():
        annots = ibs.annots(sorted(list(aids)), config=config)
        configured_obj_annots[config] = annots.view()

    # These annot views behave like annot objects
    # but they use the same internal cache
    annots1 = configured_obj_annots[qannot_cfg].view(qaids)
    annots2 = configured_obj_annots[dannot_cfg].view(daids)
    return annots1, annots2


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.algo.hots.vsone_pipeline
        python -m ibeis.algo.hots.vsone_pipeline --allexamples
        python -m ibeis.algo.hots.vsone_pipeline --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
