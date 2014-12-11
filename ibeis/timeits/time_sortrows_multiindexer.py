        """
        qfx2_xxx = qfx2_dist_
        %timeit np.vstack([row.take(sortx) for sortx, row in zip(qfx2_sortx, qfx2_xxx)])
        %timeit np.vstack((row.take(sortx) for sortx, row in zip(qfx2_sortx, qfx2_xxx)))
        %timeit np.vstack([row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_xxx)])
        %timeit np.vstack((row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_xxx)))

        import utool as ut
        #ut.rrrr()
        np.set_printoptions(threshold=10000000)
        setup = ut.unindent(
        '''
        import numpy as np
        from numpy import array, float32, int32
        qfx2_sortx = %s
        qfx2_xxx = %s
        ''' % (np.array_repr(qfx2_sortx), np.array_repr(qfx2_xxx)))
        stmt1 = 'np.vstack([row.take(sortx) for sortx, row in zip(qfx2_sortx, qfx2_xxx)])'
        stmt2 = 'np.vstack([row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_xxx)])'
        stmt3 = 'np.vstack((row[sortx] for sortx, row in zip(qfx2_sortx, qfx2_xxx)))'
        iterations = 100
        verbose = True
        stmt_list = [stmt1, stmt2, stmt3]
        (passed, time_list, result_list) = ut.timeit_compare(stmt_list, setup=setup, iterations=iterations, verbose=verbose)
        """

