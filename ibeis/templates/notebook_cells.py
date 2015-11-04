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

    draw_case_kw = dict(show_in_notebook=True, annot_modes=[0, 1])

    # Setup database specific parameter configurations
    db = '{dbname}'

    # Pick one of the following annotation configurations
    # to choose the query and database annotations
    a = [
        'default:is_known=True',
        #'default:qsame_encounter=True,been_adjusted=True,excluderef=True'
        #'default:qsame_encounter=True,been_adjusted=True,excluderef=True,qsize=10,dsize=20',
        #'timectrl:',
        #'timectrl:qsize=10,dsize=20',
        #'timectrl:been_adjusted=True,dpername=3',
        #'unctrl:been_adjusted=True',
    ]

    # Uncomment one or more of the following pipeline configurations to choose
    # how the algorithm will run.  If multiple configurations are chosen, they
    # will be compared in the histograms, but only the first configuration will
    # be used for inspecting results.
    t = [
        'default',
        #'default:K=1',
        #'default:K=1,adapteq=True',
        #'default:K=1,AI=False',
        #'default:K=1,AI=False,QRH=True',
        #'default:K=1,RI=True,AI=False',
    ]

    # Load database for this test run
    import ibeis
    ibs = ibeis.opendb(db=db)



    # ENDBLOCK
    '''))

annot_config_info =  ('# Annotation Config Info', ut.codeblock(
    r'''
    # STARTBLOCK
    acfg_list, expanded_aids_list = ibeis.expt.experiment_helpers.get_annotcfg_list(
        ibs, acfg_name_list=a)
    ibeis.expt.annotation_configs.print_acfg_list(acfg_list, expanded_aids_list, ibs)
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
        truepos=True)
    test_result.draw_func()
    # ENDBLOCK
    ''')
)

timestamp_distribution = ('# Timestamp Distribution', ut.codeblock(
    r'''
    # STARTBLOCK
    #latex_stats = ibeis.other.dbinfo.latex_dbstats([ibs], table_position='[h]') + '\n%--'
    ##print(latex_stats)}
    #pdf_fpath = ut.compile_latex_text(latex_stats, dpath=None, verbose=False, quiet=True, pad_stdout=False)
    #pdf_fpath = ut.tail(pdf_fpath, n=2)
    #print(pdf_fpath)
    #from IPython.display import HTML
    #HTML('<iframe src="%s" width=700 height=350></iframe>' % pdf_fpath)

    #_ = ibeis.other.dbinfo.get_dbinfo(ibs)
    ibeis.other.dbinfo.show_image_time_distributions(ibs, ibs.get_valid_gids())
    # ENDBLOCK
    '''))

detection_summary = ('# Detection Summary', ut.codeblock(
    r'''
    # STARTBLOCK
    # Get a sample of images
    if False:
        gids = ibs.get_valid_gids()
    else:
        from ibeis.init.filter_annots import expand_single_acfg
        from ibeis.expt import experiment_helpers
        acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(
            ibs, [a[0]], use_cache=False)
        qaids, daids = expanded_aids_list[0]
        all_aids = ut.flatten([qaids, daids])
        gids = ut.unique_keep_order2(ibs.get_annot_gids(all_aids))

    aids = ibs.get_image_aids(gids)

    nAids_list = list(map(len, aids))
    gids_sorted = ut.sortedby(gids, nAids_list)[::-1]
    samplex = list(range(5))
    print(samplex)
    gids_sample = ut.list_take(gids_sorted, samplex)

    import ibeis.viz
    for gid in ut.ProgressIter(gids_sample, lbl='drawing image'):
        ibeis.viz.show_image(ibs, gid)
    # ENDBLOCK
'''))

per_annotation_accuracy = ('# Query Accuracy (% correct annotations)', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='rank_cdf',
        db=db, a=a, t=t)
    #testres.print_unique_annot_config_stats()
    _ = testres.draw_func()
    # ENDBLOCK
    '''
))

per_name_accuracy = ('# Query Accuracy (% correct names)', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='rank_cdf',
        db=db, a=a, t=t, do_per_annot=False)
    #testres.print_unique_annot_config_stats()
    _ = testres.draw_func()
    # ENDBLOCK
    '''
))

success_scores = ('# Scores of Success Cases', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='scores',
        db=db, a=a[0:1], t=t[0:1],
        f=[':fail=False,min_gf_timedelta=None'],
    )
    _ = testres.draw_func()
    # ENDBLOCK
    '''))

all_scores = ('# Score Distribution', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='scores',
        db=db, a=a[0:1], t=t[0:1],
        f=[':fail=None,min_gf_timedelta=None']
    )
    _ = testres.draw_func()
    test_result.draw_taghist()()
    # ENDBLOCK
    '''))

success_cases = ('# Success Cases', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='draw_cases',
        db=db, a=a[0:1], t=t[0:1],
        f=[':fail=False,index=0:3,sortdsc=gtscore,max_pername=1'],
        # REM f=[':fail=False,index=0:3,sortdsc=gtscore,without_gf_tag=Photobomb,max_pername=1'],
        # REM f=[':fail=False,sortdsc=gtscore,without_gf_tag=Photobomb,max_pername=1'],
        figsize=(15, 8),
        **draw_case_kw)

    _ = testres.draw_func()
    # ENDBLOCK
    '''))

failure_type2_cases =  ('# Failure Cases Cases (false neg)', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
        e='draw_cases',
        db=db, a=a[0:1], t=t[0:1],
        f=[':fail=True,index=0:3,sortdsc=gtscore,max_pername=1'],
        **draw_case_kw)
    _ = testres.draw_func()
    # ENDBLOCK
    '''))

failure_type1_cases = ('# Failure Cases Cases (false pos)', ut.codeblock(
    r'''
    # STARTBLOCK
    testres = ibeis.run_experiment(
    e='draw_cases',
    db=db, a=a[0:1], t=t[0:1],
    f=[':fail=True,index=0:3,sortdsc=gfscore,max_pername=1'],
    **draw_case_kw)
    _ = testres.draw_func()
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


investigate_specific_case = ('# Specific Case Investigation', ut.codeblock(
    r'''
    # STARTBLOCK
    test_result = ibeis.run_experiment(
        e='draw_cases',
        db=db,
        a=a,
        #t=t,
        t=[t[0], t[0] + 'SV=False'],
        qaid_override=[2604],  # CHOOSE A SPECIFIC ANNOTATION
        **draw_case_kw)
    _ = test_result.draw_func()
    # ENDBLOCK
    '''))
