
#def begin(self, ibs, qres, aid=None, fnum=None,
#          figtitle='Inspect Query Result', same_fig=True, dodraw=True, **kwargs):
#    r"""
#    Args:
#        ibs (IBEISController):  ibeis controller object
#        qres (QueryResult):  object of feature correspondences and scores
#        aid (None):
#        fnum (int):  figure number
#        figtitle (str):
#        same_fig (bool):

#    CommandLine:
#        python -m ibeis.viz.interact.interact_matches --test-begin
#        python -m ibeis.viz.interact.interact_matches --test-begin --show

#    Example:
#        >>> # DISABLE_DOCTEST
#        >>> from ibeis.viz.interact.interact_matches import *  # NOQA
#        >>> import ibeis
#        >>> # build test data
#        >>> ibs = ibeis.opendb('testdb1')
#        >>> qres = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict())[1]
#        >>> aid2 = 2
#        >>> sel_fm = []
#        >>> # execute function
#        >>> self  = MatchInteraction(ibs, qres, aid2, annot_mode=1, dodraw=False)
#        >>> pt.show_if_requested()
#    """
#    if fnum is None:
#        fnum = pt.next_fnum()
#    fig = ih.begin_interaction('matches', fnum)  # call doclf docla and make figure
#    qaid = qres.qaid
#    if aid is None:
#        aid = qres.get_top_aids(num=1)[0]
#    rchip1, rchip2 = ibs.get_annot_chips([qaid, aid])
#    fm = qres.aid2_fm[aid]
#    mx = kwargs.pop('mx', None)
#    xywh2_ptr = [None]
#    annote_ptr = [kwargs.pop('mode', 0)]
#    self.same_fig = same_fig
#    self.last_fx = 0

#    # New state vars
#    self.vert = kwargs.pop('vert', None)
#    self.mx = None

#    # SET CLOSURE VARS
#    self.ibs      = ibs
#    self.qres     = qres
#    self.qaid     = qres.qaid
#    self.daid      = aid
#    self.fnum     = fnum
#    self.fnum2    = pt.next_fnum()
#    self.figtitle = figtitle
#    self.same_fig = same_fig
#    self.kwargs   = kwargs
#    #self.last_state = last_state
#    self.fig        = fig
#    self.annote_ptr = annote_ptr
#    self.xywh2_ptr  = xywh2_ptr
#    self.fm         = fm
#    self.rchip1     = rchip1
#    self.rchip2     = rchip2

#    if mx is None:
#        self.chipmatch_view()
#    else:
#        self.select_ith_match(mx)

#    self.set_callbacks()
#    # FIXME: this should probably not be called here
#    if dodraw:
#        ph.draw()  # ph-> adjust stuff draw -> fig_presenter.draw -> all figures show

