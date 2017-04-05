import utool as ut


def incomp_inference_test():
    from ibeis.algo.hots import demo
    kwargs = dict(num_pccs=0)
    infr = demo.demodata_infr(infer=False, **kwargs)
    infr.verbose = 10
    # Make 2 consistent and 2 inconsistent CCs
    infr.add_feedback(( 1,  2), 'match')
    infr.add_feedback(( 2,  3), 'match')
    infr.add_feedback(( 3,  4), 'match')
    infr.add_feedback(( 4,  1), 'match')
    # -----
    infr.add_feedback((11, 12), 'match')
    infr.add_feedback((12, 13), 'match')
    infr.add_feedback((13, 14), 'match')
    infr.add_feedback((14, 11), 'match')
    infr.add_feedback((12, 14), 'nomatch')
    # -----
    infr.add_feedback((21, 22), 'match')
    infr.add_feedback((22, 23), 'match')
    infr.add_feedback((23, 21), 'nomatch')
    # -----
    infr.add_feedback((31, 32), 'match')
    infr.add_feedback((32, 33), 'match')
    infr.add_feedback((33, 31), 'match')
    infr.add_feedback(( 2, 32), 'nomatch')
    infr.add_feedback(( 3, 33), 'nomatch')
    infr.add_feedback((12, 21), 'nomatch')
    # -----
    # Incomparable within CCs
    print('==========================')
    infr.add_feedback(( 1, 3), 'incomp')
    infr.add_feedback(( 1, 4), 'incomp')
    infr.add_feedback(( 1, 2), 'incomp')
    infr.add_feedback((11, 13), 'incomp')
    infr.add_feedback((11, 14), 'incomp')
    infr.add_feedback((11, 12), 'incomp')
    infr.add_feedback(( 1, 31), 'incomp')
    infr.add_feedback(( 2, 32), 'incomp')
    infr.add_feedback((12, 21), 'incomp')
    infr.add_feedback((23, 21), 'incomp')
    infr.add_feedback((12, 14), 'incomp')
    print('Final state:')
    print(ut.repr4(sorted(infr.gen_edge_attrs('decision'))))


def pos_neg_test():
    from ibeis.algo.hots import demo
    kwargs = dict(num_pccs=0)
    infr = demo.demodata_infr(infer=False, **kwargs)
    infr.verbose = 10
    c = ut.identity
    # Make 3 inconsistent CCs
    infr.add_feedback(( 1,  2), 'match'), c('Consistent merge')
    infr.add_feedback(( 2,  3), 'match'), c('Consistent merge')
    infr.add_feedback(( 3,  4), 'match'), c('Consistent merge')
    infr.add_feedback(( 4,  1), 'match'), c('Consistent merge')
    infr.add_feedback(( 1,  3), 'nomatch')
    # -----
    infr.add_feedback((11, 12), 'match')
    infr.add_feedback((12, 13), 'match')
    infr.add_feedback((13, 11), 'nomatch')
    # -----
    infr.add_feedback((21, 22), 'match')
    infr.add_feedback((22, 23), 'match')
    infr.add_feedback((23, 21), 'nomatch')
    # -----
    # Fix inconsistency
    infr.add_feedback((23, 21), 'match')
    # Merge inconsistent CCS
    infr.add_feedback(( 1, 11), 'match')
    # Negative edge within an inconsistent CC
    infr.add_feedback(( 2, 13), 'nomatch')
    # Negative edge external to an inconsistent CC
    infr.add_feedback((12, 21), 'nomatch')
    # -----
    # Make inconsistency from positive
    infr.add_feedback((31, 32), 'match')
    infr.add_feedback((33, 34), 'match')
    infr.add_feedback((31, 33), 'nomatch')
    infr.add_feedback((32, 34), 'nomatch')
    infr.add_feedback((31, 34), 'match')
    # Fix everything
    infr.add_feedback(( 1,  3), 'match')
    infr.add_feedback(( 2,  4), 'match')
    infr.add_feedback((32, 34), 'match')
    infr.add_feedback((31, 33), 'match')
    infr.add_feedback((13, 11), 'match')
    infr.add_feedback((23, 21), 'match')
    infr.add_feedback(( 1, 11), 'nomatch')
    print('Final state:')
    print(ut.repr4(sorted(infr.gen_edge_attrs('decision'))))
