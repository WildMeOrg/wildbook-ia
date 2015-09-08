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
        import vtool as vt
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
        import plottool as pt
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



