    grid_basis = [
        #ut.DimensionBasis('p', np.linspace(1, 100.0, 50)),
        ut.DimensionBasis(
            'p',
            # higher seems better but effect flattens out
            # current_best = 73
            # This seems to imply that anything with a distinctivness less than
            # .9 is not relevant
            #[73.0]
            [.5, 1.0, 2]
            #[1, 20, 73]
        ),
        ut.DimensionBasis(
            # the score seems to significantly drop off when k>2
            # but then has a spike at k=8
            # best is k=2
            'K',
            [2]
            #[2, 3, 4, 5, 7, 8, 9, 16],
        ),
        #ut.DimensionBasis('dcvs_clip_max', ),
        #ut.DimensionBasis('dcvs_clip_max', np.linspace(.01, .11, 100)),
        ut.DimensionBasis(
            'dcvs_clip_max',
            # THERE IS A VERY CLEAR SPIKE AT .09
            [.09],
            #[.09, 1.0],
            #np.linspace(.05, .15, 10),
        ),
        #ut.DimensionBasis(FiltKeys.FG + '_power', ),
        ut.DimensionBasis(
            FiltKeys.FG + '_power',
            # the forground power seems to be very influential in scoring
            # it seems higher is better but effect flattens out
            # the reason it seems to be better is because it zeros out weights
            [.1, 1.0, 2.0]
            #np.linspace(.01, 30.0, 10)
        ),
        ut.DimensionBasis(
            FiltKeys.HOMOGERR + '_power',
            # current_best = 2.5
            #[2.5]
            [.1, 1.0, 2.0]
            #np.linspace(.1, 10, 5)
            #np.linspace(.1, 10, 30)
        ),
    ]
