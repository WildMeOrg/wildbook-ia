# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

initialize = ('# Initialization', ut.codeblock(
    r'''
    # STARTBLOCK
    {autogen_str}
    # Matplotlib stuff
    import matplotlib as mpl
    %matplotlib inline
    %load_ext autoreload
    %autoreload

    # Set global utool flags
    import utool as ut
    ut.util_io.__PRINT_WRITES__ = False
    ut.util_io.__PRINT_READS__ = False
    ut.util_parallel.__FORCE_SERIAL__ = True
    ut.util_cache.VERBOSE_CACHE = False
    ut.NOT_QUIET = False

    import plottool as pt
    fix_figsize = ut.partial(pt.set_figsize, w=30, h=10, dpi=256)
    pt.custom_figure.TITLE_SIZE = 20
    pt.custom_figure.LABEL_SIZE = 20
    pt.custom_figure.FIGTITLE_SIZE = 20

    draw_case_kw = dict(show_in_notebook=True, annot_modes=[0, 1])

    # Setup database specific parameter configurations
    db = '{dbname}'

    # Pick one of the following annotation configurations
    # to choose the query and database annotations
    a = [
        {annotconfig_list_body}
    ]
    #'ctrl:pername=None,view=left,view_ext=1,exclude_reference=False'


    # Set to override any special configs
    qaid_override = None
    daid_override = None

    # Uncomment one or more of the following pipeline configurations to choose
    # how the algorithm will run.  If multiple configurations are chosen, they
    # will be compared in the histograms, but only the first configuration will
    # be used for inspecting results.
    t = [
        {pipeline_list_body}
    ]

    # Load database for this test run
    import ibeis
    ibs = ibeis.opendb(db=db)

    # Make notebook cells wider
    from IPython.core.display import HTML
    HTML("<style>body .container {{ width:99% !important; }}</style>")
    # ENDBLOCK
    '''))


fluke_select = ('# Humpback Select',  ut.codeblock(
    r'''
    # STARTBLOCK
    # Tag annotations which have been given manual notch points
    from ibeis_flukematch.plugin import *  # NOQA
    ibs = ibeis.opendb(defaultdb='humpbacks')
    all_aids = ibs.get_valid_aids()
    isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag')
    aid_list = ut.compress(all_aids, isvalid)
    # Tag the appropriate annots
    ibs.append_annot_case_tags(aid_list, ['hasnotch'] * len(aid_list))
    #depc = ibs.depc
    #qaid_override = aid_list[0:5]
    #daid_override = aid_list[0:7]
    #print(qaid_override)
    #print(daid_override)
    # ENDBLOCK
    '''))

annot_config_info =  ('# Annotation Config Info', ut.codeblock(
    r'''
    # STARTBLOCK
    acfg_list, expanded_aids_list = ibeis.expt.experiment_helpers.get_annotcfg_list(
        ibs, acfg_name_list=a, qaid_override=qaid_override, daid_override=daid_override, verbose=0)
    ibeis.expt.annotation_configs.print_acfg_list(acfg_list, expanded_aids_list, ibs, per_qual=True)
    # ENDBLOCK
    ''')
)


pipe_config_info =  ('# Pipeline Config Info', ut.codeblock(
    r'''
    # STARTBLOCK
    cfgdict_list, pipecfg_list = ibeis.expt.experiment_helpers.get_pipecfg_list(
        test_cfg_name_list=t, ibs=ibs)
    ibeis.expt.experiment_helpers.print_pipe_configs(cfgdict_list, pipecfg_list)
    # ENDBLOCK
    ''')
)


dbsize_expt = ('# Database Size Experiment ', ut.codeblock(
    r'''
    # STARTBLOCK
    if True:
        test_result = ibeis.run_experiment(
            e='rank_surface',
            db=db,
            a=['varysize_td'],
            t=['candk'])
        #test_result.print_unique_annot_config_stats()
        #test_result.print_acfg_info()
        test_result.draw_func()

    if True:
        # This test requires a little bit of relaxation to get enough data
        test_result = ibeis.run_experiment(
            e='rank_surface',
            db=db,
            a=['varysize_tdqual:qmin_pername=3,dpername=[1,2]'],
            t=['candk'])
        #test_result.print_unique_annot_config_stats()
        #test_result.print_acfg_info()
        test_result.draw_func()
    # ENDBLOCK
    ''')
)


timedelta_distribution = ('# Result Timedelta Distribution', ut.codeblock(
    r'''
    # STARTBLOCK
    test_result = ibeis.run_experiment(
        e='timedelta_hist',
        db=db,
        a=a[0:1],
        t=t[0:1],
        qaid_override=qaid_override, daid_override=daid_override,
        truepos=True)
    test_result.draw_func()
    fix_figsize()
    # ENDBLOCK
    ''')
)


#latex_stats = ibeis.other.dbinfo.latex_dbstats([ibs], table_position='[h]') + '\n%--'
##print(latex_stats)
#pdf_fpath = ut.compile_latex_text(latex_stats, dpath=None, verbose=False, quiet=True, pad_stdout=False)
#pdf_fpath = ut.tail(pdf_fpath, n=2)
#print(pdf_fpath)
#from IPython.display import HTML
#HTML('<iframe src="%s" width=700 height=350></iframe>' % pdf_fpath)
#_ = ibeis.other.dbinfo.get_dbinfo(ibs)
timestamp_distribution = ('# Timestamp Distribution', ut.codeblock(
    r'''
    # STARTBLOCK
    # Get images of those used in the tests
    ibs, qaids, daids = ibeis.testdata_expanded_aids(a=a[0], ibs=ibs)
    aids = ut.unique_keep_order(ut.flatten([qaids, daids]))
    gids = ut.unique_keep_order(ibs.get_annot_gids(aids))
    # Or just get time delta of all images
    #gids = ibs.get_valid_gids()

    ibeis.other.dbinfo.show_image_time_distributions(ibs, gids)
    #ibeis.other.dbinfo.show_image_time_distributions(ibs, gids)
    # ENDBLOCK
    '''))

example_annotations = ('# Example Annotations / Detections', ut.codeblock(
    r'''
    # STARTBLOCK
    # Get a sample of images
    #gids = ibs.get_valid_gids()
    ibs, qaids, daids = ibeis.testdata_expanded_aids(a=a[0], ibs=ibs)
    aids = ut.unique_keep_order(ut.flatten([qaids, daids]))
    gids = ut.unique_keep_order(ibs.get_annot_gids(aids))
    # Or just get time delta of all images
    #gids = ibs.get_valid_gids()

    aids = ibs.get_image_aids(gids)

    nAids_list = list(map(len, aids))
    gids_sorted = ut.sortedby(gids, nAids_list)[::-1]
    samplex = list(range(5))
    print(samplex)
    gids_sample = ut.take(gids_sorted, samplex)

    import ibeis.viz
    for gid in ut.ProgressIter(gids_sample, lbl='drawing image'):
        ibeis.viz.show_image(ibs, gid)
    # ENDBLOCK
'''))


example_names = ('# Example Name Graph',  ut.codeblock(
    r'''
    # STARTBLOCK
    from ibeis.viz import viz_graph
    ibs, qaids, daids = ibeis.testdata_expanded_aids(a=a[0], ibs=ibs)
    aids = ut.unique_keep_order(ut.flatten([qaids, daids]))
    # Sample some annotations
    aids = ibs.sample_annots_general(aids, filter_kw=dict(sample_size=20, min_pername=2), verbose=False)
    # Visualize name graph
    namegraph = viz_graph.make_name_graph_interaction(ibs, aids=aids, zoom=.4)
    fix_figsize()
    # ENDBLOCK
    ''')
)


#######
# CONFIG COMPARISONS
#######


per_annotation_accuracy = ('# Query Accuracy (% correct annotations)', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='rank_cdf',
        db=db, a=a, t=t, qaid_override=qaid_override, daid_override=daid_override)
    #testres.print_unique_annot_config_stats()
    _ = testres.draw_func()
    fix_figsize()
    # ENDBLOCK
    '''
))

per_name_accuracy = ('# Query Accuracy (% correct names)', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='rank_cdf',
        db=db, a=a, t=t, do_per_annot=False, qaid_override=qaid_override, daid_override=daid_override)
    #testres.print_unique_annot_config_stats()
    _ = testres.draw_func()
    fix_figsize()
    # ENDBLOCK
    '''
))


config_overlap = ('# Configuration Overlap', ut.codeblock(
    r'''
    # STARTBLOCK
    # How well do different configurations compliment each other?
    testres.print_config_overlap()
    # ENDBLOCK
    '''
))


feat_score_sep = ('# Feature Correspondence Score Separation', ut.codeblock(
    r'''
    # STARTBLOCK
    test_result = ibeis.run_experiment(
        e='TestResult.draw_feat_scoresep',
        db=db,
        a=a,
        t=t,
        #disttype=['L2_sift']
    )
    #test_result.draw_feat_scoresep(f='', disttype=['L2_sift'])
    test_result.draw_feat_scoresep(f='', disttype=None)
    fix_figsize()
    # ENDBLOCK
    '''))


success_annot_scoresep = ('# Scores of Success Cases', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='draw_annot_scoresep',
        db=db, a=a[0:1], t=t[0:1],
        f=[':fail=False,min_gf_timedelta=None'],
    )
    _ = testres.draw_func()
    fix_figsize()
    # ENDBLOCK
    '''))

all_annot_scoresep = ('# All Score Distribution', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='scores',
        db=db, a=a[0:1], t=t[0:1],
        qaid_override=qaid_override, daid_override=daid_override,
        f=[':fail=None,min_gf_timedelta=None']
    )
    _ = testres.draw_func()
    fix_figsize()
    testres.draw_taghist()()
    fix_figsize()
    # ENDBLOCK
    '''))


view_intereseting_tags = ('# Interesting Tags', ut.codeblock(
    r'''
    # STARTBLOCK
    test_result = ibeis.run_experiment(
        e='draw_cases',
        db=db,
        a=a,
        t=t,
        f=[':index=0:5,with_tag=interesting'],
        **draw_case_kw)
    _ = test_result.draw_func()
    # ENDBLOCK
    '''))

easy_success_cases = ('# Cases: Top Success Cases', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='draw_cases',
        db=db, a=a[0:1], t=t[0:1],
        f=[':fail=False,index=0:3,sortdsc=gtscore,max_pername=1'],
        # REM f=[':fail=False,index=0:3,sortdsc=gtscore,without_gf_tag=Photobomb,max_pername=1'],
        # REM f=[':fail=False,sortdsc=gtscore,without_gf_tag=Photobomb,max_pername=1'],
        figsize=(30, 8),
        **draw_case_kw)

    _ = testres.draw_func()
    # ENDBLOCK
    '''))

hard_success_cases = ('# Cases: Challenging Success Cases', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='draw_cases',
        db=db, a=a[0:1], t=t[0:1],
        f=[':fail=False,index=0:3,sortasc=gtscore,max_pername=1'],
        # REM f=[':fail=False,index=0:3,sortdsc=gtscore,without_gf_tag=Photobomb,max_pername=1'],
        # REM f=[':fail=False,sortdsc=gtscore,without_gf_tag=Photobomb,max_pername=1'],
        figsize=(30, 8),
        **draw_case_kw)

    _ = testres.draw_func()
    # ENDBLOCK
    '''))


# ================
# Individual Cases
# ================


failure_type2_cases =  ('# Cases: Failure (false neg)', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='draw_cases',
        db=db, a=a[0:1], t=t[0:1],
        f=[':fail=True,index=0:3,sortdsc=gtscore,max_pername=1'],
        figsize=(30, 8),
        **draw_case_kw)
    _ = testres.draw_func()
    # ENDBLOCK
    '''))

failure_type1_cases = ('# Cases: Failure (false pos)', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
    e='draw_cases',
    db=db, a=a[0:1], t=t[0:1],
    f=[':fail=True,index=0:3,sortdsc=gfscore,max_pername=1'],
    figsize=(30, 8),
    **draw_case_kw)
    _ = testres.draw_func()
    # ENDBLOCK
    '''))


investigate_specific_case = ('# Cases: Custom Investigation', ut.codeblock(
    r'''
    # STARTBLOCK
    test_result = ibeis.run_experiment(
        e='draw_cases',
        db=db,
        a=a,
        #t=t,
        t=[t[0], t[0] + 'SV=False'],
        qaid_override=[2604],  # CHOOSE A SPECIFIC ANNOTATION
        figsize=(30, 8),
        **draw_case_kw)
    _ = test_result.draw_func()
    # ENDBLOCK
    '''))
