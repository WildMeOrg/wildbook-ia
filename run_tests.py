#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool as ut


def run_tests():
    # Build module list and run tests
    import ibeis.ibsfuncs
    import ibeis.control.DBCACHE_SCHEMA
    import ibeis.control.DB_SCHEMA
    import ibeis.control.IBEISControl
    import ibeis.control._autogen_ibeiscontrol_funcs
    import ibeis.dbio.export_subset
    import ibeis.dev.experiment_harness
    import ibeis.dev.results_all
    import ibeis.gui.inspect_gui
    import ibeis.model.Config
    import ibeis.model.detect.randomforest
    import ibeis.model.hots.hots_query_result
    import ibeis.model.hots.match_chips4
    import ibeis.model.hots.nn_weights
    import ibeis.model.hots.pipeline
    import ibeis.model.hots.query_request
    import ibeis.model.hots.score_normalization
    import ibeis.model.hots.voting_rules2
    import ibeis.model.preproc.preproc_annot
    import ibeis.model.preproc.preproc_chip
    import ibeis.model.preproc.preproc_detectimg
    import ibeis.model.preproc.preproc_encounter
    import ibeis.model.preproc.preproc_feat
    import ibeis.model.preproc.preproc_featweight
    import ibeis.model.preproc.preproc_image
    import ibeis.model.preproc.preproc_probchip
    import ibeis.model.preproc.preproc_residual
    import ibeis.viz.viz_sver

    module_list = [
        ibeis.ibsfuncs,
        ibeis.control.DBCACHE_SCHEMA,
        ibeis.control.DB_SCHEMA,
        ibeis.control.IBEISControl,
        ibeis.control._autogen_ibeiscontrol_funcs,
        ibeis.dbio.export_subset,
        ibeis.dev.experiment_harness,
        ibeis.dev.results_all,
        ibeis.gui.inspect_gui,
        ibeis.model.Config,
        ibeis.model.detect.randomforest,
        ibeis.model.hots.hots_query_result,
        ibeis.model.hots.match_chips4,
        ibeis.model.hots.nn_weights,
        ibeis.model.hots.pipeline,
        ibeis.model.hots.query_request,
        ibeis.model.hots.score_normalization,
        ibeis.model.hots.voting_rules2,
        ibeis.model.preproc.preproc_annot,
        ibeis.model.preproc.preproc_chip,
        ibeis.model.preproc.preproc_detectimg,
        ibeis.model.preproc.preproc_encounter,
        ibeis.model.preproc.preproc_feat,
        ibeis.model.preproc.preproc_featweight,
        ibeis.model.preproc.preproc_image,
        ibeis.model.preproc.preproc_probchip,
        ibeis.model.preproc.preproc_residual,
        ibeis.viz.viz_sver,
    ]
    ut.doctest_module_list(module_list)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    run_tests()
