"""
not really used
most things in here can be depricated
"""
from __future__ import absolute_import, division, print_function
import utool
import numpy as np
import utool as ut
import six
from ibeis.dev import results_organizer
from ibeis.dev import results_analyzer
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[results_all]', DEBUG=False)


class AllResults(utool.DynStruct):
    """
    Data container for all compiled results
    """
    def __init__(allres):
        super(AllResults, allres).__init__(child_exclude_list=['qaid2_qres'])
        allres.ibs = None
        allres.qaid2_qres = None
        allres.allorg = None
        allres.cfgstr = None
        allres.dbname = None

    def get_orgtype(allres, orgtype):
        orgres = allres.allorg.get(orgtype)
        return orgres

    def get_cfgstr(allres):
        return allres.cfgstr

    def make_title(allres):
        return allres.dbname + '\n' + ut.packstr(allres.get_cfgstr(), textwidth=80, break_words=False, breakchars='_', wordsep='_')

    def get_qres(allres, qaid):
        return allres.qaid2_qres[qaid]

    def get_orgres_desc_match_dists(allres, orgtype_list):
        return results_analyzer.get_orgres_desc_match_dists(allres, orgtype_list)

    def get_orgres_annotationmatch_scores(allres, orgtype_list):
        return results_analyzer.get_orgres_annotationmatch_scores(allres, orgtype_list)


def init_allres(ibs, qaid2_qres, qreq_=None):
    if qreq_ is not None:
        allres_cfgstr = qreq_.get_cfgstr()
    else:
        allres_cfgstr = '???'
    print('Building allres')
    allres = AllResults()
    allres.qaid2_qres = qaid2_qres
    allres.allorg = results_organizer.organize_results(ibs, qaid2_qres)
    allres.cfgstr = allres_cfgstr
    allres.dbname = ibs.get_dbname()
    allres.ibs = ibs
    return allres


def learn_score_normalization(ibs, qres_list, qaid_list):
    """
    Args:
        qaid2_qres (int): query annotation id

    Example:
        >>> from ibeis.dev.results_all import *   # NOQA
        >>> import plottool as pt  # NOQA
        >>> from ibeis.dev import results_all
        >>> import ibeis
        >>> #ibs = ibeis.opendb('PZ_MTEST')
        >>> dbname = 'GZ_ALL'
        >>> #dbname = 'PZ_Master0'
        >>> ibs = ibeis.opendb(dbname)
        >>> qaid_list = daid_list = ibs.get_valid_aids()
        >>> hard_aids = ibs.get_hard_annot_rowids()
        >>> easy_aids = ibs.get_easy_annot_rowids()
        >>> qaid_list = hard_aids + easy_aids[::1]
        >>> cfgdict = dict(codename='nsum')
        >>> qaid2_qres, qreq_ = results_all.get_qres_and_qreq_(ibs, qaid_list, daid_list, cfgdict)
        >>> qres_list = [qaid2_qres[aid] for aid in qaid_list]
        >>> results_all.learn_score_normalization(ibs, qres_list, qaid_list)
        >>> pt.present()

    References:
        http://en.wikipedia.org/wiki/Statistical_hypothesis_testing
        http://en.wikipedia.org/wiki/Type_I_and_type_II_errors
        http://en.wikipedia.org/wiki/P-value
        ftp://ftp.stat.duke.edu/pub/WorkingPapers/10-13.pdf

    Dev::
        valid_aids = ibs.get_valid_aids()
        hard_aids = ibs.get_hard_annot_rowids()
        qaid2_qres = ibs._query_chips4(hard_aids, daid_list, cfgdict=cfgdict)
        qres = qres_list[0]
        #sorted_nids, sorted_scores = qres.get_sorted_nids_and_scores(ibs)
        #data = sorted_scores
        #qaid_list = [aid for aid in six.iterkeys(qaid2_qres)]
        #qres_list = [qaid2_qres[qaid] for qaid in qaid_list]
        #for qres in qres_list:
        #    qres.rrr(verbose=False)
    """
    import plottool as pt  # NOQA
    good_tp_nscores = []
    good_tn_nscores = []
    good_tp_ndiff = []
    good_tn_ndiff = []
    good_tp_nmatches = []
    good_tn_nmatches = []
    good_tp_aidnid_pairs = []
    good_tn_aidnid_pairs = []
    for qx, qres in enumerate(qres_list):
        qaid = qres.get_qaid()
        if not qres.is_nsum():
            raise AssertionError('must be nsum')
        if not ibs.get_annot_has_groundtruth(qaid):
            continue
        qnid = ibs.get_annot_name_rowids(qres.get_qaid())
        #sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)

        def get_num_nmatches(qres, sorted_aids):
            # returns number of matches to a name
            sorted_nmatches = []
            for aids in sorted_aids:
                fs_list = [qres.aid2_fs[aid] for aid in aids]
                num_nmatches = np.sum([(fs > 1E-6).sum() for fs in fs_list])
                sorted_nmatches.append(num_nmatches)
            return sorted_nmatches

        nscoretup = qres.get_nscoretup(ibs)
        (sorted_nids, sorted_nscores, sorted_aids, sorted_scores)  = nscoretup
        sorted_nmatches = get_num_nmatches(qres, sorted_aids)

        sorted_ndiff = -np.diff(sorted_nscores.tolist())
        sorted_nids = np.array(sorted_nids)
        is_positive  = sorted_nids == qnid
        is_negative = np.logical_and(~is_positive, sorted_nids > 0)
        if not np.any(is_positive) or not np.any(is_negative):
            continue
        gt_rank = np.where(is_positive)[0][0]
        gf_rank = np.nonzero(is_negative)[0][0]
        if gt_rank == 0 and len(sorted_nscores) > gf_rank:
            if len(sorted_ndiff) > gf_rank:
                good_tp_nscores.append(sorted_nscores[gt_rank])
                good_tn_nscores.append(sorted_nscores[gf_rank])
                good_tp_ndiff.append(sorted_ndiff[gt_rank])
                good_tn_ndiff.append(sorted_ndiff[gf_rank])
                good_tp_nmatches.append(sorted_nmatches[gt_rank])
                good_tn_nmatches.append(sorted_nmatches[gf_rank])
                good_tp_aidnid_pairs.append((qaid, sorted_nids[gt_rank]))
                good_tn_aidnid_pairs.append((qaid, sorted_nids[gf_rank]))

    good_tp_nscores = np.array(good_tp_nscores)
    good_tn_nscores = np.array(good_tn_nscores)
    good_tp_ndiff = np.array(good_tp_ndiff)
    good_tn_ndiff = np.array(good_tn_ndiff)

    clip_score = 2000
    #overshoot_factor = good_tp_nscores.max() / good_tn_nscores.max()
    #if overshoot_factor > 5:
    #    clip_score = good_tp_nscores.mean() + good_tp_nscores.std() * 2

    #pt.close_all_figures()
    #import imp
    #imp.reload(pt.plots)
    #imp.reload(pt)

    good_tp = good_tp_nscores  # NOQA
    good_tn = good_tn_nscores  # NOQA
    lbl = 'score'  # NOQA
    inspect_pdfs(good_tp_nscores, good_tn_nscores, 'score', clip_score)
    #inspect_pdfs(good_tp_ndiff, good_tn_ndiff, 'diff', clip_score)

    #posfeat_tup = (good_tp_nscores, good_tp_ndiff)
    #negfeat_tup = (good_tn_nscores, good_tn_ndiff)
    #lbltup = ('nscores', 'ndiff')

    #posfeat_tup = (good_tp_nscores, good_tp_ndiff, good_tp_nmatches)
    #negfeat_tup = (good_tn_nscores, good_tn_ndiff, good_tn_nmatches)

    #posfeat_tup = (good_tp_nscores, good_tp_nmatches)
    #negfeat_tup = (good_tn_nscores, good_tn_nmatches)
    #lbltup = ('nscores', 'nmatches')

    #posfeat_tup = (good_tp_ndiff, good_tp_nmatches)
    #negfeat_tup = (good_tn_ndiff, good_tn_nmatches)
    #lbltup = ('ndiff', 'nmatches')

    #inspect_svm_classifier(ibs, posfeat_tup, negfeat_tup, good_tp_aidnid_pairs,
    #                       good_tn_aidnid_pairs, clip_score, lbltup)
    #pt.present()


def inspect_pdfs(good_tp, good_tn, lbl, clip_score):
    import plottool as pt  # NOQA

    #good_all = np.hstack((good_tp, good_tn))
    score_tp_pdf = ut.estimate_pdf(good_tp, gridsize=512, adjust=8)
    score_tn_pdf = ut.estimate_pdf(good_tn, gridsize=512, adjust=8)
    #score_pdf = ut.estimate_pdf(good_all, gridsize=512)
    score_domain = np.linspace(0, clip_score, 1024)

    p_score_given_tp = score_tp_pdf.evaluate(score_domain)
    p_score_given_tn = score_tn_pdf.evaluate(score_domain)
    #p_score = score_pdf.evaluate(score_domain)

    p_score = np.array(p_score_given_tp) + np.array(p_score_given_tn)

    # Apply bayes
    p_tp = .5
    p_tn = 1.0 - p_tp
    p_tp_given_score = ut.bayes_rule(p_score_given_tp, p_score, p_tp)
    p_tn_given_score = ut.bayes_rule(p_score_given_tn, p_score, p_tn)

    pt.plots.plot_sorted_scores(
        (good_tn, good_tp),
        (lbl + ' | tn', lbl + ' | tp'),
        figtitle='sorted nscores')

    pt.plots.plot_densities(
        (p_score_given_tn,  p_score_given_tp, p_score),
        (lbl + ' given tn', lbl + ' given tp', lbl),
        figtitle='pre_bayes pdf ' + lbl,
        xdata=score_domain)

    pt.plots.plot_densities(
        (p_tn_given_score, p_tp_given_score),
        ('tn given ' + lbl, 'tp given ' + lbl),
        figtitle='post_bayes pdf ' + lbl,
        xdata=score_domain)


def inspect_svm_classifier(ibs, posfeat_tup, negfeat_tup, good_tp_aidnid_pairs,
                           good_tn_aidnid_pairs, clip_score, lbltup=None):
    #import sklearn
    #assert sklearn.__version__ >= '0.15.2'
    from sklearn import svm
    import plottool as pt  # NOQA
    # Build SVM Features and targets

    if lbltup is None:
        lbltup = list(map(ut.get_varname_from_stack, posfeat_tup))

    positive_features  = np.vstack(posfeat_tup).T
    negative_features = np.vstack(negfeat_tup).T

    nTrue = len(negative_features)
    nFalse = len(negative_features)
    # Pack svm features and targets
    all_aidnid_pairs = np.vstack((good_tp_aidnid_pairs, good_tn_aidnid_pairs))
    X = svm_features = np.vstack((positive_features, negative_features))
    Y = svm_targets = np.hstack((np.ones(nTrue), -np.ones(nFalse)))
    # Create support vector classifier
    svc = svm.LinearSVC(C=1.0, dual=False)
    #svc = svm.NuSVC(kernel='linear', probability=True)
    with ut.Timer('training SVM'):
        svc.fit(svm_features, svm_targets)

    if hasattr(svc, 'probability'):
        svc.decision_function(svm_features)

    # Evaluate performance on training set
    is_falsepositive = svc.predict(negative_features) == 1.0  # type1 -- false positive
    is_falsenegative = svc.predict(positive_features) == -1.0  # type2 -- false negative
    nType1 = is_falsepositive.sum()
    nType2 = is_falsenegative.sum()
    nError = nType1 + nType2
    nTotal = nTrue + nFalse
    nCorrect = nTotal - nError
    percentCorrect =  100 * nCorrect / float(nTotal)
    print('SVM made %d/%d correct decisions. %.2f%% correct' % (nCorrect, nTotal, percentCorrect))
    print('SVM made %d/%d type1 errors. (false positive)' % (nType1, nFalse))
    print('SVM made %d/%d type1 errors. (false negative)' % (nType2, nTrue))

    def plot_2d_svc(svc, X, Y, clip_score):
        # make grid for feature 1 and 2
        h = 100
        f1_min = X[:, 0].min() - 1
        f2_min = X[:, 1].min() - 1
        f1_max = min(X[:, 0].max() + 1, clip_score + 1)
        f2_max = min(X[:, 1].max() + 1, clip_score + 1)
        f1xs, f2xs = np.meshgrid(np.arange(f1_min, f1_max, h),
                                 np.arange(f2_min, f2_max, h))
        zinput = np.c_[f1xs.ravel(), f2xs.ravel()]
        Z = svc.predict(zinput)
        Z = Z.reshape(f1xs.shape)
        # Plot decision boundary
        fnum = pt.next_fnum()
        pt.figure(fnum=fnum)
        cmap = pt.get_binary_svm_cmap()
        pt.plt.contourf(f1xs, f2xs, Z, cmap=cmap, alpha=0.8)
        # plot training points
        valid_X = X[:, 0] < clip_score
        pt.plt.scatter(X[valid_X, 0], X[valid_X, 1], c=Y[valid_X], cmap=cmap)
        pt.update()

    def plot_nd_svc(svc, X, Y, clip_score):

        h = 100
        feat_mins = [X[:, dimx].min() - 1 for dimx in range(len(X.T))]
        feat_maxs = [min(X[:, dimx].max() + 1, clip_score + 1) for dimx in range(len(X.T))]
        feat_basis = [np.arange(fmin, fmax, min(h, fmax - fmin / 100))
                      for fmin, fmax in zip(feat_mins, feat_maxs)]

        feat_grids = np.meshgrid(*feat_basis)
        cinput = np.vstack([fgrid.ravel() for fgrid in feat_grids]).T
        C = svc.predict(cinput)
        C = C.reshape(feat_grids[0].shape)

        # make grid for feature 1 and 2
        if len(X.T) != 3:
            fnum = pt.next_fnum()
            fig = pt.figure(fnum=fnum)

            # Plot decision boundary
            cmap = pt.get_binary_svm_cmap()
            pt.plt.contourf(*(feat_grids + [C]), cmap=cmap, alpha=0.8)
            # plot training points
            valid_X = X.T[0] < clip_score
            featiter = (arg[valid_X] for arg in X.T)
            pt.plt.scatter(*featiter, c=Y[valid_X], cmap=cmap)

            ax = pt.plt.gca()
            if len(lbltup) > 0:
                ax.set_xlabel(lbltup[0])
            if len(lbltup) > 1:
                ax.set_ylabel(lbltup[1])
            if len(lbltup) > 2:
                ax.set_zlabel(lbltup[2])

            def on_press(event):
                'on button press we will see if the mouse is over us and store some data'
                if not event.inaxes:
                    return
                ax, x, y = event.inaxes, event.xdata, event.ydata  # NOQA
                fx = utool.nearest_point(x, y, X)[0]
                qaid, nid = all_aidnid_pairs[fx]
                print('qaid = %r, nid = %r' % (qaid, nid))
                dbname = ibs.get_dbname()
                print('python dev.py --cfg codename:nsum -t query --qaid %d --db %s -w' % (qaid, dbname))
                print('python dev.py --cfg codename:nsum --query-aid %d --db %s --gui' % (qaid, dbname))
                #qaid = %r, nid = %r' % (qaid, nid))
                if ut.DEBUG2:
                    ut.embed()
            pt.interact_helpers.connect_callback(fig, 'button_press_event', on_press)
            pt.update()
        # Pytinstaller should not user vtk
        #else:
        #    import vtk
        #    data_matrix = (C + 2).astype(np.uint8)
        #    #data_matrix = np.zeros([75, 75, 75], dtype=np.uint8)
        #    #data_matrix[0:35, 0:35, 0:35] = 50
        #    #data_matrix[25:55, 25:55, 25:55] = 100
        #    #data_matrix[45:74, 45:74, 45:74] = 150
        #    dataImporter = vtk.vtkImageImport()
        #    data_string = data_matrix.tostring()
        #    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        #    dataImporter.SetDataScalarTypeToUnsignedChar()
        #    dataImporter.SetNumberOfScalarComponents(1)
        #    #data_extent = ut.flatten(zip(feat_mins, feat_maxs))
        #    #de = data_extent
        #    #dataImporter.SetDataExtent(de[0], de[1], de[2], de[3], de[4], de[5])
        #    #dataImporter.SetWholeExtent(de[0], de[1], de[2], de[3], de[4], de[5])
        #    dataImporter.SetDataExtent(0, 74, 0, 74, 0, 74)
        #    dataImporter.SetWholeExtent(0, 74, 0, 74, 0, 74)
        #    alphaChannelFunc = vtk.vtkPiecewiseFunction()
        #    alphaChannelFunc.AddPoint(1, 0.05)
        #    alphaChannelFunc.AddPoint(3, 0.1)
        #    colorFunc = vtk.vtkColorTransferFunction()
        #    colorFunc.AddRGBPoint(1, 1.0, 0.0, 0.0)
        #    colorFunc.AddRGBPoint(3, 0.0, 0.0, 1.0)
        #    volumeProperty = vtk.vtkVolumeProperty()
        #    volumeProperty.SetColor(colorFunc)
        #    volumeProperty.SetScalarOpacity(alphaChannelFunc)
        #    volumeProperty.ShadeOn()
        #    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        #    volumeMapper = vtk.vtkVolumeRayCastMapper()
        #    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
        #    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

        #    volume = vtk.vtkVolume()
        #    volume.SetMapper(volumeMapper)
        #    volume.SetProperty(volumeProperty)
        #    renderer = vtk.vtkRenderer()
        #    renderWin = vtk.vtkRenderWindow()
        #    renderWin.AddRenderer(renderer)
        #    renderInteractor = vtk.vtkRenderWindowInteractor()
        #    renderInteractor.SetRenderWindow(renderWin)
        #    renderer.AddVolume(volume)
        #    renderer.SetBackground(0, 0, 0)
        #    renderWin.SetSize(400, 400)
        #    def exitCheck(obj, event):
        #        if obj.GetEventPending() != 0:
        #            obj.SetAbortRender(1)
        #    print('about to start vtk')
        #    renderWin.AddObserver("AbortCheckEvent", exitCheck)
        #    renderInteractor.Initialize()
        #    renderWin.Render()
        #    renderInteractor.Start()

    plot_nd_svc(svc, X, Y, clip_score)


def get_stem_data(ibs, qaid2_qres):
    """
    returns data for pt.plot_stems

    data is sorted by result ranks. nsum is taken into acount if it exists

    get_stem_data

    Args:
        qaid2_qres (int): query annotation id

    Example:
        >>> from ibeis.dev.results_all import *   # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_list = qaid_list = ibs.get_valid_aids()
        >>> qaid2_qres, qreq_ = results_all.get_qres_and_qreq_(ibs, qaid_list, daid_list)
    """
    #ut.embed()
    import numpy as np
    #unflat_xdata = []
    unflat_ydata = []

    for qx, (qaid, qres) in enumerate(six.iteritems(qaid2_qres)):
        #qres.rrr(verbose=False)
        is_nsum = qres.is_nsum()
        worst_possible_rank = qres.get_worse_possible_rank()
        gt_ranks  = np.array(qres.get_gt_ranks(ibs=ibs, fillvalue=worst_possible_rank))
        if len(gt_ranks) == 0:
            continue
        if is_nsum:
            #gt_scores = np.array(qres.get_gt_scores(ibs=ibs))
            argx = gt_ranks.argmin()
            best_rank = gt_ranks[argx:argx + 1]
            qres_ydata = best_rank
        else:
            qres_ydata = gt_ranks
        #qres_xdata  [qx] * len(qres_ydata)
        #unflat_xdata.append(qres_xdata)
        unflat_ydata.append(qres_ydata)

    unflat_max_y = map(sorted, unflat_ydata)
    unflat_ydata2 = ut.sortedby2(unflat_ydata, unflat_max_y)
    unflat_xdata2 = [[qx] * len(ydata) for qx, ydata in enumerate(unflat_ydata2)]
    y_data = ut.flatten(unflat_ydata2)
    x_data = ut.flatten(unflat_xdata2)
    #unflat_ydata2 = ut.sortedby2(unflat_ydata, unflat_max_y)
    #x_data = ut.flatten(unflat_xdata)
    #x_data = ut.sortedby2(x_data, y_data)
    #y_data = ut.sortedby2(y_data, y_data)
    return x_data, y_data


# ALL RESULTS CACHE


__ALLRES_CACHE__ = {}
__QRESREQ_CACHE__ = {}


def build_cache_key(ibs, qaid_list, daid_list, cfgdict):
    # a little overconstrained
    cfgstr = ibs.cfg.query_cfg.get_cfgstr()
    query_hashid = ibs.get_annot_hashid_semantic_uuid(qaid_list, prefix='Q')
    data_hashid  = ibs.get_annot_hashid_semantic_uuid(daid_list, prefix='D')
    key = (query_hashid, data_hashid, cfgstr, str(cfgdict))
    return key


def get_qres_and_qreq_(ibs, qaid_list, daid_list=None, cfgdict=None):
    if daid_list is None:
        daid_list = ibs.get_valid_aids()

    qres_cache_key = build_cache_key(ibs, qaid_list, daid_list, cfgdict)

    try:
        (qaid2_qres, qreq_) = __QRESREQ_CACHE__[qres_cache_key]
    except KeyError:
        qaid2_qres, qreq_ = ibs._query_chips4(qaid_list, daid_list,
                                              return_request=True,
                                              cfgdict=cfgdict)
        # Cache save
        __QRESREQ_CACHE__[qres_cache_key] = (qaid2_qres, qreq_)
    return (qaid2_qres, qreq_)


def get_allres(ibs, qaid_list, daid_list=None, cfgdict=None):
    """
    get_allres

    Args:
        ibs (IBEISController):
        qaid_list (int): query annotation id
        daid_list (list):

    Returns:
        AllResults: allres

    Example:
        >>> from dev import *  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_list = ibs.get_valid_aids()
        >>> daid_list = None
        >>> allres = get_allres(ibs, qaid_list, daid_list)
        >>> print(allres)
        >>> allres.allorg['top_true'].printme3()
    """
    print('[dev] get_allres')
    if daid_list is None:
        daid_list = ibs.get_valid_aids()
    allres_key = build_cache_key(ibs, qaid_list, daid_list, cfgdict)
    try:
        allres = __ALLRES_CACHE__[allres_key]
    except KeyError:
        qaid2_qres, qreq_ = get_qres_and_qreq_(ibs, qaid_list, daid_list, cfgdict)
        allres = init_allres(ibs, qaid2_qres, qreq_)
        # Cache save
        __ALLRES_CACHE__[allres_key] = allres
    return allres


def test_confidence_measures(ibs, qres_list, qaid_list):
    import scipy.stats as spstats

    def find_tscore(qres, ibs):
        # H0: null hypothesis nid_list[0] is not he same animal
        # H1: alt hypothesis nid_list[0] is the same animal
        # assuming the null hypothesis is true, how likely is it
        # that we got the sample that we did.
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        alt_score = sorted_nscores[0]
        sample_measurements = sorted_nscores[1:]
        num_samples = len(sample_measurements)
        sample_mean = sample_measurements.mean()
        sample_std  = sample_measurements.std()

        population_mean = sample_mean
        population_std = sample_std / np.sqrt(num_samples)
        # t-score is like the z-score but for a sample
        tscore = (alt_score - population_mean) / population_std
        return tscore

    def find_prob_tscore(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        alt_score = sorted_nscores[0]
        sample_measurements = sorted_nscores[1:]
        tscore, pval = spstats.ttest_1samp(sample_measurements, alt_score)
        return pval

    def find_prob_tscore_next(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        tscore, pval1 = spstats.ttest_1samp(sorted_nscores[1:], sorted_nscores[0])
        tscore, pval2 = spstats.ttest_1samp(sorted_nscores[2:], sorted_nscores[1])
        return pval1 - pval2

    def find_prob_densitity(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        alt_score = sorted_nscores[0]
        sample_measurements = sorted_nscores[1:]
        data_pdf = ut.estimate_pdf(sample_measurements)
        density = data_pdf.evaluate(alt_score)[0]
        prob_correct = (1 - density)
        return '%.2f' % prob_correct

    def find_prob_normtest1(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        sample_measurements = sorted_nscores[1:]
        kurt, pval = spstats.normaltest(sample_measurements)
        return '%.2f' % pval

    def find_prob_normtest2(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        kurt, pval = spstats.normaltest(sorted_nscores)
        return '%.2f' % pval

    def find_scorediff(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        return -np.diff(sorted_nscores)[0]

    def find_prob_scorediff(qres, ibs):
        sorted_nids, sorted_nscores = qres.get_sorted_nids_and_scores(ibs)
        diff_scores = -np.diff(sorted_nscores)
        tscore, pval1 = spstats.ttest_1samp(diff_scores[1:], diff_scores[0])
        tscore, pval2 = spstats.ttest_1samp(diff_scores[2:], diff_scores[1])
        return '%.2E' % ((1 - (pval1)) - ((1 - pval2)))

    if False:
        true_nid_list = ibs.get_annot_name_rowids(qaid_list)
        decision_tup_list = [qres.get_name_decisiontup(ibs) for qres in qres_list]
        decision_nid_list = ut.get_list_column(decision_tup_list, 0)
        decision_score_list = ut.get_list_column(decision_tup_list, 1)
        iscorrect_list = [nid_target == nid_res
                          for nid_target, nid_res in zip(decision_nid_list, true_nid_list)]

        t_list          = [find_tscore(qres, ibs) for qres in qres_list]
        pval_t_list     = [find_prob_tscore(qres, ibs) for qres in qres_list]
        pval_tnext_list = [find_prob_tscore_next(qres, ibs) for qres in qres_list]
        norm1_list      = [find_prob_normtest1(qres, ibs) for qres in qres_list]
        norm2_list      = [find_prob_normtest2(qres, ibs) for qres in qres_list]
        pden_list       = [find_prob_densitity(qres, ibs) for qres in qres_list]
        scorediff       = [find_scorediff(qres, ibs) for qres in qres_list]
        prob_scorediffs = [find_prob_scorediff(qres, ibs) for qres in qres_list]

        column_list = [iscorrect_list, decision_score_list, scorediff, prob_scorediffs, t_list, pval_t_list, pval_tnext_list, pden_list, norm1_list, norm2_list]
        column_labels  = ['Correct', 'score_list', 'scorediff', 'pscorediff', 't', 'pval_t', 'pnext', 'pden_list', 'norm1', 'norm2']

        print(ut.make_csv_table(column_list, column_labels))


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.dev.results_all; utool.doctest_funcs(ibeis.dev.results_all, allexamples=True)"
        python -c "import utool, ibeis.dev.results_all; utool.doctest_funcs(ibeis.dev.results_all)"
        python -m ibeis.dev.results_all --allexamples
        python -m ibeis.dev.results_all --test-learn_score_normalization --enableall
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
    import plottool as pt  # NOQA
    exec(pt.present())
