@six.add_metaclass(ut.ReloadingMetaclass)
class ScoreNormalizerUnsupervised(object):
    def __init__(encoder, X=None, **kwargs):
        encoder.learn_kw = ut.update_existing(
            dict(
                gridsize=1024,
                adjust=8,
                monotonize=False,
                #monotonize=True,
                #clip_factor=(ut.PHI + 1),
                clip_factor=None,
                reverse=None,
            ), kwargs)
        check_unused_kwargs(kwargs, encoder.learn_kw.keys())
        # Target recall for learned threshold
        # Support data
        encoder.support = dict(
            X=None,
        )
        # Learned score normalization
        encoder.score_domain = None
        if X is not None:
            encoder.fit(X)

    def fit(encoder, X, y=None, verbose=True):
        """
        Fits estimator to data.
        """
        encoder.learn_probabilities(X, y, verbose=verbose)

    def learn_probabilities(encoder, X, y=None, verbose=True):
        import vtool_ibeis as vt
        gridsize = encoder.learn_kw['gridsize']
        adjust = encoder.learn_kw['adjust']
        # Record support
        encoder.support['X'] = X
        xdata = X[~np.isnan(X)]
        xdata_pdf = vt.estimate_pdf(xdata, gridsize=gridsize, adjust=adjust)
        # Find good score domain range
        min_score, max_score = xdata.min(), xdata.max()
        xdata_domain = np.linspace(min_score, max_score, gridsize)
        p_xdata = xdata_pdf.evaluate(xdata_domain)
        encoder.p_xdata = p_xdata
        encoder.xdata_domain = xdata_domain
        import scipy.interpolate
        encoder.interp_fn = scipy.interpolate.interp1d(
            encoder.xdata_domain, encoder.p_xdata, kind='linear',
            copy=False, assume_sorted=False)

    def visualize(encoder):
        import plottool_ibeis as pt
        #is_timedata = False
        is_timedelta = True
        p_xdata = encoder.p_xdata
        xdata_domain = encoder.xdata_domain
        #if is_timedata:
        #    xdata_domain_ = [ut.unixtime_to_datetimeobj(unixtime) for unixtime in xdata_domain]
        if is_timedelta:
            #xdata_domain_ = [ut.unixtime_to_timedelta(unixtime) for unixtime in xdata_domain]
            pass
        else:
            pass
            #xdata_domain_ = xdata_domain
        pt.plot_probabilities([p_xdata], [''], xdata=xdata_domain)
        ax = pt.gca()

        # HISTOGRAM
        if False:
            X = encoder.support['X']
            xdata = X[~np.isnan(X)]
            n, bins, patches = pt.plt.hist(xdata, 1000)

        ax.set_xlabel('xdata')
        if is_timedelta:
            ax.set_xlabel('Time Delta')
            ax.set_title('Timedelta distribution')
            def timeTicks(x, pos):
                import datetime
                d = datetime.timedelta(seconds=x)
                return str(d)
            import matplotlib as mpl
            formatter = mpl.ticker.FuncFormatter(timeTicks)
            ax.xaxis.set_major_formatter(formatter)
            pt.gcf().autofmt_xdate()
        #if is_timedata:
        #    ax.set_xlabel('Date')
        #    ax.set_title('Timestamp distribution')
        #    pt.gcf().autofmt_xdate()
        #ax.set_title('Timestamp distribution of %s' % (ibs.get_dbname()))
        #pt.gcf().autofmt_xdate()


def bow_test():
    x  = np.array([1, 0, 0, 0, 0, 0], dtype=np.float)
    c1 = np.array([1, 0, 1, 0, 0, 1], dtype=np.float)
    c2 = np.array([1, 1, 1, 1, 1, 1], dtype=np.float)
    x /= x.sum()
    c1 /= c1.sum()
    c2 /= c2.sum()

    # fred_query = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    # sue_query  = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    # tom_query  = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    # # columns that are distinctive per name
    # #                      f1  f2  s1  s2  s3  t1  z1  z2  z3  z4, z5, z6
    # fred1      = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    # fred2      = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    # sue1       = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    # sue2       = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    # sue3       = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)
    # tom1       = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=np.float)

    names         = ['fred', 'sue', 'tom']
    num_exemplars = [     3,     2,     1]

    ax2_nx = np.array(ut.flatten([[nx] * num for nx, num in enumerate(num_exemplars)]))

    total = sum(num_exemplars)

    num_words = total * 2
    # bow vector for database
    darr = np.zeros((total, num_words))

    for ax in range(len(darr)):
        # nx = ax2_nx[ax]
        # num = num_exemplars[nx]
        darr[ax, ax] = 1
        darr[ax, ax + total] = 1

    # nx2_axs = dict(zip(*))
    import vtool_ibeis as vt
    groupxs = vt.group_indices(ax2_nx)[1]
    class_bows = np.vstack([arr.sum(axis=0) for arr in vt.apply_grouping(darr, groupxs)])
    # generate a query for each class
    true_class_bows = class_bows[:]
    # noise words
    true_class_bows[:, -total:] = 1
    true_class_bows = true_class_bows / true_class_bows.sum(axis=1)[:, None]

    class_bows = class_bows / class_bows.sum(axis=1)[:, None]

    confusion = np.zeros((len(names), len(names)))

    for trial in range(1000):
        # bow vector for query
        qarr = np.zeros((len(names), num_words))

        for cx in range(len(class_bows)):
            sample = np.random.choice(np.arange(num_words), size=30, p=true_class_bows[cx])
            hist = np.histogram(sample, bins=np.arange(num_words + 1))[0]
            qarr[cx] = (hist / hist.max()) >= .5
        # normalize histograms
        qarr = qarr / qarr.sum(axis=1)[:, None]

        # Scoring for each class
        similarity = qarr.dot(class_bows.T)
        distance = 1 - similarity
        confusion += distance

    x /= x.sum()
    c1 /= c1.sum()
    c2 /= c2.sum()

    print(x.dot(c1))
    print(x.dot(c2))


def match_inspect_graph():
    """

    CommandLine:
        python -m vtool_ibeis.inspect_matches match_inspect_graph --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.inspect_matches import *  # NOQA
        >>> import vtool_ibeis as vt
        >>> gt.ensure_qapp()
        >>> ut.qtensure()
        >>> self = match_inspect_graph()
        >>> self.show()
        >>> # xdoctest: +REQUIRES(--show)
        >>> self.update()
        >>> gt.qtapp_loop(qwin=self, freq=10)
    """
    import vtool_ibeis as vt
    annots = [lazy_test_annot('easy1.png'),
              lazy_test_annot('easy2.png'),
              lazy_test_annot('easy3.png'),
              lazy_test_annot('zebra.png'),
              lazy_test_annot('hard3.png')]
    matches = [vt.PairwiseMatch(a1, a2) for a1, a2 in ut.combinations(annots, 2)]
    self = MultiMatchInspector(matches=matches)
    return self


class MultiMatchInspector(INSPECT_BASE):
    # DEPRICATE

    def initialize(self, matches):
        self.matches = matches

        self.splitter = self.addNewSplitter(orientation='horiz')
        # tab_widget = self.addNewTabWidget(verticalStretch=1)
        # self.edge_tab = tab_widget.addNewTab('Edges')
        # self.match_tab = tab_widget.addNewTab('Matches')

        self.edge_api_widget = gt.APIItemWidget(
            doubleClicked=self.edge_doubleclick)
        self.match_inspector = MatchInspector(match=None)

        self.splitter.addWidget(self.edge_api_widget)
        self.splitter.addWidget(self.match_inspector)

        self.populate_edge_model()

    def edge_doubleclick(self, qtindex):
        row = qtindex.row()
        match = self.matches[row]
        self.match_inspector.set_match(match)

    def populate_edge_model(self):
        edge_api = gt.CustomAPI(
            col_name_list=['index', 'aid1', 'aid2'],
            col_getter_dict={
                'index': list(range(len(self.matches))),
                'aid1': [m.annot1['aid'] for m in self.matches],
                'aid2': [m.annot2['aid'] for m in self.matches],
            }, sort_reverse=False)
        headers = edge_api.make_headers(tblnice='Edges')
        self.edge_api_widget.change_headers(headers)
        self.edge_api_widget.resize_headers(edge_api)
        self.edge_api_widget.view.verticalHeader().setVisible(True)
        # self.edge_api_widget.view.verticalHeader().setDefaultSectionSize(24)
        # self.edge_api_widget.view.verticalHeader().setDefaultSectionSize(221)
        # self.edge_tab.setTabText('Matches (%r)' % (self.edge_api_widget.model.num_rows_total))


