    # Broken:
        python -m ibeis.gui.inspect_gui --test-test_singleres_api --show

def test_singleres_api(ibs, qaid_list, daid_list):
    """

    Args:
        ibs       (IBEISController):
        qaid_list (list): query annotation id list
        daid_list (list): database annotation id list

    Returns:
        dict: locals_

    CommandLine:
        python -m ibeis.gui.inspect_gui --test-test_singleres_api --cmd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.gui.inspect_gui import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:1]
        >>> daid_list = ibs.get_valid_aids()
        >>> main_locals = test_singleres_api(ibs, qaid_list, daid_list)
        >>> main_execstr = ibeis.main_loop(main_locals)
        >>> print(main_execstr)
        >>> exec(main_execstr)
    """
    from ibeis.viz.interact import interact_qres2  # NOQA
    from ibeis.gui import inspect_gui
    from ibeis.dev import results_all
    allres = results_all.get_allres(ibs, qaid_list[0:1])
    guitool.ensure_qapp()
    tblname = 'qres'
    qaid2_qres = allres.qaid2_qres
    ranks_lt = 5
    qres_wgt = inspect_gui.QueryResultsWidget(ibs, qaid2_qres,
                                              ranks_lt=ranks_lt,
                                              name_scoring=True,
                                              singlematch_api=True)
    qres_wgt.show()
    qres_wgt.raise_()
    locals_ =  locals()
    return locals_


def make_singlematch_api(ibs, qaid2_qres, ranks_lt=None, name_scoring=False):
    """
    Builds columns which are displayable in a ColumnListTableWidget
    """
    if ut.VERBOSE:
        print('[inspect] make_qres_api')
    ibs.cfg.other_cfg.ranks_lt = 2
    ranks_lt = ranks_lt if ranks_lt is not None else ibs.cfg.other_cfg.ranks_lt
    # Get extra info
    assert len(qaid2_qres) == 1
    qaid = list(qaid2_qres.keys())[0]
    qres = qaid2_qres[qaid]
    (aids, scores) = qres.get_aids_and_scores(name_scoring=name_scoring, ibs=ibs)
    #ranks = scores.argsort()
    qaid = qres.get_qaid()

    def get_rowid_button(rowid):
        def get_button(ibs, qtindex):
            model = qtindex.model()
            aid2 = model.get_header_data('aid', qtindex)
            truth = ibs.get_match_truth(qaid, aid2)
            truth_color = vh.get_truth_color(truth, base255=True,
                                                lighten_amount=0.35)
            truth_text = ibs.get_match_text(qaid, aid2)
            callback = partial(review_match, ibs, qaid, aid2)
            #print('get_button, aid1=%r, aid2=%r, row=%r, truth=%r' % (aid1, aid2, row, truth))
            buttontup = (truth_text, callback, truth_color)
            return buttontup

    col_name_list = [
        'aid',
        'score',
        #'status',
        'resthumb',
    ]

    col_types_dict = dict([
        ('aid',        int),
        ('score',      float),
        #('status',     str),
        ('resthumb',   'PIXMAP'),
    ])

    col_getters_dict = dict([
        ('aid',        np.array(aids)),
        ('score',      np.array(scores)),
        ('resthumb',   ibs.get_annot_chip_thumbtup),
        #('status',     partial(get_status, ibs)),
        #('querythumb', ibs.get_annot_chip_thumbtup),
        #('truth',     truths),
        #('opt',       opts),
    ])

    col_bgrole_dict = {
        #'status': partial(get_status_bgrole, ibs),
    }
    col_ider_dict = {
        #'status'     : ('qaid', 'aid'),
        'resthumb'   : ('aid'),
        #'name'       : ('aid'),
    }
    col_setter_dict = {
    }
    editable_colnames =  []
    sortby = 'score'
    # Insert info into dict
    get_thumb_size = lambda: ibs.cfg.other_cfg.thumb_size
    qres_api = CustomAPI(col_name_list, col_types_dict, col_getters_dict,
                         col_bgrole_dict, col_ider_dict, col_setter_dict,
                         editable_colnames, sortby, get_thumb_size)
    return qres_api


