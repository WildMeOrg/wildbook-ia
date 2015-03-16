from __future__ import absolute_import, division, print_function
from six.moves import reduce
import fractions
import ibeis
import numpy as np
import plottool as pt
import random
import scipy as sp
import utool as ut
from six.moves import builtins
import six
(print, print_, printDBG, rrr, profile) = ut.inject( __name__, '[optimze_k]', DEBUG=False)


def evaluate_training_data(ibs, varydict, nDaids_basis):
    # load a dataset
    #dbname = 'PZ_MTEST'
    #dbname = 'GZ_ALL'
    def get_set_groundfalse(ibs, qaids):
        # get groundfalse annots relative to the entire set
        valid_nids = ibs.get_valid_nids()
        qnids = ibs.get_annot_nids(qaids)
        nid_list = list(set(valid_nids) - set(qnids))
        aids_list = ibs.get_name_aids(nid_list)
        return ut.flatten(aids_list)

    # determanism
    np.random.seed(0)
    random.seed(0)
    cfgdict_list = ut.all_dict_combinations(varydict)

    nError_list  = []
    nDaids_list  = []
    cfgdict_list2 = []

    qaids_all = ibs.filter_junk_annotations(ibs.get_annot_rowid_sample(per_name=1, min_ngt=2, distinguish_unknowns=True))
    qaids = qaids_all[::2]
    print('nQaids = %r' % len(qaids))
    daids_gt_sample = ut.flatten(ibs.get_annot_groundtruth_sample(qaids, isexemplar=None))
    daids_gf_all = get_set_groundfalse(ibs, qaids)
    ut.assert_eq(len(daids_gt_sample), len(qaids), 'missing gt')
    for target_nDaids in ut.ProgressIter(nDaids_basis, lbl='testing dbsize'):
        print('---------------------------')
        # Sample one match from the groundtruth with padding
        daids_gf_sample = ut.random_sample(daids_gf_all, max(0, target_nDaids - len(daids_gt_sample)))
        daids = sorted(daids_gt_sample + daids_gf_sample)
        nDaids = len(daids)
        if target_nDaids != nDaids:
            continue

        with ut.Indenter('[nDaids=%r]' % (nDaids)):
            print('nDaids = %r' % nDaids)
            for cfgdict in ut.ProgressIter(cfgdict_list, lbl='testing cfgdict'):
                qreq_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict)
                qres_list = ibs.query_chips(qreq_=qreq_, verbose=ut.VERBOSE)
                gt_ranks_list = [qres.get_gt_ranks(ibs=ibs) for qres in qres_list]
                incorrect_list = [len(gt_ranks) == 0 or min(gt_ranks) != 0 for gt_ranks in gt_ranks_list]
                nErrors = sum(incorrect_list)
                nError_list.append(nErrors)
                nDaids_list.append(nDaids)
                cfgdict_list2.append(cfgdict.copy())

    nError_list = np.array(nError_list)
    nDaids_list = np.array(nDaids_list)
    K_list = np.array([cfgdict['K'] for cfgdict in cfgdict_list2])
    return nDaids_list, K_list, nError_list


def test_training_data(varydict, nDaids_basis):
    varydict['nDaids'] = nDaids_basis
    cfgdict_list = ut.all_dict_combinations(varydict)
    K_list = ut.get_list_column(cfgdict_list, 'K')
    nDaids_list = ut.get_list_column(cfgdict_list, 'nDaids')
    max_error = min(nDaids_basis)
    nError_perterb = np.random.rand(len(K_list))

    #def distance_point_polynomial(point, poly_coeff):
    #    """
    #    References:
    #        http://kitchingroup.cheme.cmu.edu/blog/2013/02/14/Find-the-minimum-distance-from-a-point-to-a-curve/
    #    """
    #    def f(x):
    #        return x ** 2
    #    def objective(X, *args):
    #        point = args[0]
    #        x, y = X
    #        px, py = point
    #        return np.sqrt((x - px) ** 2 + (y - py) ** 2)
    #    def c1(X, *args):
    #        x, y = X
    #        return f(x) - y
    #    X = sp.optimize.fmin_cobyla(objective, x0=[0.5, 0.5], args=(point,), cons=[c1], disp=False)
    #    return X
    #point_list = np.array([point for point in zip(nDaids_list, K_list)])
    #poly_coeff = [0.2,  0.5]  # K model_params
    #closest_point_list = np.array([distance_point_polynomial(point, poly_coeff) for point in point_list])
    #dist_list = np.sqrt(((point_list - closest_point_list) ** 2).sum(axis=1))
    #nError_list = max_error * dist_list / dist_list.max() + nError_perterb
    nError_list = (np.array(nDaids_list) * .00001)
    nError_list /= nError_list.max()
    nError_list *= (max_error - 2)
    nError_list += 1 + nError_perterb

    #K_list      = np.array([  1,   1,    1,   4,   4,    4,   7,   7,    7,   10,  10,   10,   13,  13,   13])
    #nDaids_list = np.array([100, 500, 1000, 100, 500, 1000, 100, 500, 1000,  100, 500, 1000,  100, 500, 1000])
    #nError_list = np.array([  5,  54,  130,  50,  50,   70,  14,  54,   40,   20,   9,   43,   90,  20,  130])
    return nDaids_list, K_list, nError_list


# Convert our non-uniform grid into a uniform grid using gcd
def compute_interpolation_grid(known_nd_data, pad_steps=0):
    """ use gcd to get the number of steps to take in each dimension """
    ug_steps = [reduce(fractions.gcd, np.unique(x_).tolist()) for x_ in known_nd_data.T]
    ug_min   = known_nd_data.min(axis=0)
    ug_max   = known_nd_data.max(axis=0)
    ug_basis = [
        np.arange(min_ - (step_ * pad_steps), max_ + (step_ * (pad_steps + 1)), step_)
        for min_, max_, step_ in zip(ug_min, ug_max, ug_steps)
    ]
    ug_shape = tuple([basis.size for basis in ug_basis][::-1])
    # ig = interpolated grid
    unknown_nd_data = np.vstack([_pts.flatten() for _pts in np.meshgrid(*ug_basis)]).T
    return unknown_nd_data, ug_shape


def interpolate_error(known_nd_data, known_targets, unknown_nd_data):
    """
    References:
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.griddata.html
    """
    method = 'cubic'  # {'linear', 'nearest', 'cubic'}
    interpolated_targets = sp.interpolate.griddata(known_nd_data, known_targets, unknown_nd_data, method=method)
    interpolated_targets[np.isnan(interpolated_targets)] = known_targets.max() * 2
    return interpolated_targets


def compute_K(nDaids, model_params, force_int=True):
    """
    Args:
        nDaids (int): number of database annotations to compute K for
        model_params (list): coefficients of the n-degree polynomial

    CommandLine:
        python -m ibeis.dev.optimize_k --test-compute_K --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.dev.optimize_k import *  # NOQA
        >>> import plottool as pt
        >>> nDaids_list = np.arange(0, 1000)
        >>> model_params = [.2, .5]
        >>> K_list = compute_K(nDaids_list, model_params)
        >>> pt.plot2(nDaids_list, K_list, x_label='num_names', y_label='K',
        ...          equal_aspect=False, marker='g-', pad=1, dark=True)
        >>> pt.show_if_requested()

    """
    K = np.polyval(model_params, nDaids)
    if force_int:
        K = np.round(K)
    return K


def minimize_compute_K_params(known_nd_data, known_target_points, given_data_dims):
    """
    References:
        http://docs.scipy.org/doc/scipy-0.14.0/reference/optimize.html
    """
    poly_degree = 1
    mode = 'brute'
    #mode = 'simplex'

    if poly_degree == 2:
        initial_model_params = [0, 0.2,  0.5]  # a guess
        ranges = (slice(0, 1, .1), slice(0, 1, .1), slice(0, 1, .1))
        #brute_force_basis = list(map(np.mgrid.__getitem__, ranges))
    elif poly_degree == 1:
        #initial_model_params = [ 0.00814424,  0.1855764 ]
        initial_model_params = [  6.73655087e-05,   9.25]
        initial_model_params = [  0,   10]
        #initial_model_params = [0.02,  0.5]
        fidelity = 10
        ranges = (slice(0, 1, .01 * fidelity), slice(0, 10, .1 * fidelity))
    else:
        assert poly_degree > 2
        initial_model_params = [0 for _ in range(poly_degree)]
        ranges = [slice(-2, 2, .1) for _ in range(poly_degree)]
        #raise AssertionError('Unknown poly_degree=%r' % (poly_degree,))

    infiter = builtins.iter(int, 1)
    # TODO: progress iter for unknown size
    if mode == 'brute':
        brute_force_basis = list(map(np.mgrid.__getitem__, ranges))
        nTotal = np.prod([_basis.size for _basis in brute_force_basis])
    else:
        nTotal = 1
    optprog = ut.ProgressIter(infiter, nTotal=nTotal, lbl='optimizing', freq=1)
    optprogiter = builtins.iter(optprog)

    def objective_func(model_params, *args):
        known_nd_data, known_target_points, unique_nDaids = args
        # Return the error over all of the daids
        K_list = np.array([compute_K(_nDaids, model_params, force_int=False) for _nDaids in unique_nDaids])
        six.next(optprogiter)
        if np.any(K_list <= 0):
            return np.inf
        unknown_nd_data = np.vstack([unique_nDaids, K_list]).T
        error_list = interpolate_error(known_nd_data, known_target_points, unknown_nd_data)
        total_error = error_list.sum()
        #print('-----------------')
        #print('model_params = %s' % (np.array_str(np.array(model_params)),))
        #print('K_list       = %s' % (np.array_str(np.array(K_list)),))
        #print('total_error = %r' % (total_error,))
        return total_error

    unique_nDaids = np.unique(known_nd_data.take(given_data_dims, axis=1))
    args = known_nd_data, known_target_points, unique_nDaids

    if mode == 'simplex':
        _out = sp.optimize.fmin(objective_func, initial_model_params, xtol=.01, args=args, disp=True, full_output=True)
        xopt, fopt, nIter, funcalls, warnflag = _out[:5]
        #, allvecs
        opt_model_params = xopt
        #opt_model_params = sp.optimize.basinhopping(objective_func, guess, args=args)
        #opt_model_params = sp.optimize.brute(objective_func, ranges, args=args, )
    elif mode == 'brute':
        x0, fval, grid, Jout = sp.optimize.brute(
            objective_func, ranges, args=args, full_output=True)
        opt_model_params = x0
    else:
        raise AssertionError('Unknown mode=%r' % (mode,))

    opt_K_list = [compute_K(_nDaids, opt_model_params) for _nDaids in unique_nDaids]
    print('opt_model_params = %r' % (opt_model_params,))
    print('opt_K_list = %r' % (opt_K_list,))
    return opt_model_params


def plot_search_surface(known_nd_data, known_target_points, given_data_dims, opt_model_params=None):
    pt.figure(2, doclf=True)

    # Interpolate uniform grid positions
    unknown_nd_data, ug_shape = compute_interpolation_grid(known_nd_data, 0 * 5)
    interpolated_error = interpolate_error(known_nd_data, known_target_points, unknown_nd_data)

    ax = pt.plot_surface3d(
        unknown_nd_data.T[0].reshape(ug_shape),
        unknown_nd_data.T[1].reshape(ug_shape),
        interpolated_error.reshape(ug_shape),
        xlabel='nDaids',
        ylabel='K',
        zlabel='error',
        #dark=False,
    )
    ax.scatter(known_nd_data.T[0], known_nd_data.T[1], known_target_points, s=100, c=pt.YELLOW)

    assert len(given_data_dims) == 1, 'can only plot 1 given data dim'
    xdim = given_data_dims[0]
    ydim = (xdim + 1) % (len(known_nd_data.T))
    known_nd_min = known_nd_data.min(axis=0)
    known_nd_max = known_nd_data.max(axis=0)
    xmin, xmax = known_nd_min[xdim], known_nd_max[xdim]
    ymin, ymax = known_nd_min[ydim], known_nd_max[ydim]
    zmin, zmax = known_target_points.min(), known_target_points.max()

    if opt_model_params is not None:
        # plot learned data if availabel
        #given_known_nd_data = known_nd_data.take(given_data_dims, axis=1)
        xdata = np.linspace(xmin, xmax)
        ydata = compute_K(xdata, opt_model_params)
        xydata = np.array((xdata, ydata)).T
        zdata = interpolate_error(known_nd_data, known_target_points, xydata)
        ax.plot(xdata, ydata, zdata, c=pt.ORANGE)
        ymax = max(ymax, ydata.max())
        ymin = min(ymin, ydata.min())
        zmin = min(zmin, zdata.min())
        zmax = max(zmax, zdata.max())
        ax.scatter(xdata, ydata, zdata, s=100, c=pt.ORANGE)
        #[t.set_color('white') for t in ax.xaxis.get_ticklines()]
        #[t.set_color('white') for t in ax.xaxis.get_ticklabels()]
    ax.set_aspect('auto')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    return ax


def learn_k():
    r"""
    CommandLine:
        python -m ibeis.dev.optimize_k --test-learn_k
        python -m ibeis.dev.optimize_k --test-learn_k --show
        python -m ibeis.dev.optimize_k --test-learn_k --show --dummy

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.dev.optimize_k import *  # NOQA
        >>> import plottool as pt
        >>> # build test data
        >>> # execute function
        >>> known_nd_data, known_target_points, given_data_dims, opt_model_params = learn_k()
        >>> # verify results
        >>> ut.quit_if_noshow()
        >>> plot_search_surface(known_nd_data, known_target_points, given_data_dims, opt_model_params)
        >>> pt.all_figures_bring_to_front()
        >>> pt.show_if_requested()
    """
    # Compute Training Data
    varydict = {
        #'K': [4, 7, 10, 13, 16, 19, 22, 25][:4],
        'K': [1, 2, 3, 4, 8, 10, 13, 15],
        #'nDaids': [20, 100, 250, 500, 750, 1000],
    }
    nDaids_basis = [20, 30, 50, 75, 100, 200, 250, 300, 350, 400, 500, 600, 750, 800, 900, 1000, 1500]
    DUMMY = ut.get_argflag('--dummy')
    if DUMMY:
        nDaids_list, K_list, nError_list = test_training_data(varydict, nDaids_basis)
    else:
        dbname = 'PZ_Master0'
        ibs = ibeis.opendb(dbname)
        nDaids_list, K_list, nError_list = evaluate_training_data(ibs, varydict, nDaids_basis)
    #unique_nDaids = np.unique(nDaids_list)

    # Alias to general optimization problem
    known_nd_data = np.vstack([nDaids_list, K_list]).T
    known_target_points = nError_list
    # Mark the data we are given vs what we want to learn
    given_data_dims = [0]
    #learn_data_dims = [1]

    # Minimize K params
    opt_model_params = minimize_compute_K_params(known_nd_data, known_target_points, given_data_dims)
    return known_nd_data, known_target_points, given_data_dims, opt_model_params

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.dev.optimize_k
        python -m ibeis.dev.optimize_k --allexamples
        python -m ibeis.dev.optimize_k --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
