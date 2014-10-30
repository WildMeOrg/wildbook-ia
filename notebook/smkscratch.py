    if False:
        #
        print('\n+----------')
        print('Get a list of image ids (a stands for annotation)')
        aid_list = ibs.get_valid_aids()[0:3]
        print('aid_list = ')
        print(aid_list)
        print('L----------')
        #
        print('\n+----------')
        print('Get the set of vectors from each image/anotation')
        vecs_list = ibs.get_annot_vecs(aid_list)
        print('vecs_list = ')
        print(vecs_list)
        print('L----------')
        #__________
        print('\n+----------')
        print('Try using just the list of ndarrays to create a hierarchy (doesnt work)')
        vecs_series = pd.Series(vecs_list, index=aid_list, name='vecs')
        print('vecs_series = ')
        print(vecs_series)
        print('L----------')
        #
        print('\n+----------')
        print('Try mapping each numpy array in the list to a dataframe')
        vecs_dflist = map(pd.DataFrame, vecs_list)
        print('vecs_dflist = ')
        print(vecs_dflist)
        print('L----------')
        #__________
        print('\n+----------')
        print('Try using just the list of dataframes to create a hierarchy (doesnt work)')
        vecs_dfseries = pd.Series(vecs_dflist, index=aid_list, name='vecs')
        print('vecs_dfseries = ')
        print(vecs_dfseries)
        print('L----------')

        vecs_df = pd.DataFrame(vecs_dflist, index=aid_list, columns=['vecs'])
        vecs_df = pd.DataFrame(vecs_list, index=aid_list, columns=['vecs'])
        #
        kpts_dflist = map(pd.DataFrame, kpts_list)
        kpts_series = pd.Series(kpts_dflist, index=aid_list, name='kpts')

vecs_df = pd.DataFrame(vecs_list, index=aid_list)

kpts_series = pd.Series(kpts_list, index=aid_series, name='kpts')
vecs_series = pd.Series(vecs_list, index=aid_series, name='vecs')
annots_df = pd.concat([kpts_series, vecs_series], axis=1)
    kpts_df = pd.DataFrame(kpts_list, index=aid_series, columns=['kpts'])
    vecs_df = pd.DataFrame(vecs_list, index=aid_series, columns=['vecs'])
    annots_df = pd.concat([kpts_df, vecs_df], axis=1)


aid_list = ibs.get_valid_aids()
kpts_list = ibs.get_annot_kpts(aid_list)
vecs_list = ibs.get_annot_vecs(aid_list)
aid_series = pd.Series(aid_list, name='aid')


def kpts_to_dataframe(kpts):
    fx_series = pd.Series(np.arange(kpts.shape[0]), name='fx')
    #kpcols = ['loc_x', 'loc_y', 'shape_a', 'shape_c', 'shape_d', 'theta']
    kptsdf = pd.DataFrame(kpts, index=fx_series)  # , columns=kpcols)
    return kptsdf


def vecs_to_dataframe(vecs):
    fx_series = pd.Series(np.arange(vecs.shape[0]), name='fx')
    #dimx_series = pd.Series(np.arange(vecs.shape[1]), name='dimx')
    vecdf = pd.DataFrame(vecs, index=fx_series)  # , columns=dimx_series)
    return vecdf

kpts_dflist = map(kpts_to_dataframe, kpts_list)
vecs_dflist = map(vecs_to_dataframe, vecs_list)

kpts_df = pd.DataFrame(kpts_dflist, index=aid_series, columns=['kpts'])
vecs_df = pd.DataFrame(vecs_dflist, index=aid_series, columns=['vecs'])
annots_df = pd.concat([kpts_df, vecs_df], axis=1)

    #score_mi = pd.MultiIndex.from_product((qfxs, _idxs), names=('qfxs', '_idxs'))
    #print()
    #score_df = pd.DataFrame(score_matrix, index=score_mi)
    # Scores for each database vector
    #scores = pd.DataFrame(score_matrix.sum(axis=0), columns=['score'])
    # Use cartesian product of these indexes to produce feature matches
    #qfxs = pd.DataFrame(wx2_qfxs[wx], columns=['qfx'])
    #dfxs = pd.DataFrame(invindex.idx2_fx[_idxs], columns=['dfx'])
    #daxs = pd.DataFrame(invindex.idx2_ax[_idxs], columns=['dax'])
    #daids = pd.DataFrame(invindex.ax2_aid[invindex.idx2_ax[_idxs]], columns=['daid'])
    #print(scores)
    #print(daids)
    #result_df = pd.concat((scores, daids), axis=1)  # concat columns
    #daid_group = result_df.groupby(['daid'])
    #daid2_wordscore = daid_group['score'].sum()




    #qfx2_axs = []
    #qfx2_fm = []
    #qfx2_fs = []
    #aid_fm = []
    #aid_fs = []

    #idx2_dfx  = invindex.idx2_fx
    #idx2_wx  = invindex.idx2_wx
    #wx2_idxs_series = {wx: pd.Series(idxs, name='idx') for wx, idxs in
    #                   six.iteritems(invindex.wx2_idxs)}
    #wx2_qfxs_series = {wx: pd.Series(qfx, name='qfx') for wx, qfx in
    #                   six.iteritems(wx2_qfxs)}
    #qfx2_idx = np.tile(_idxs, (len(qfxs), 1))
    #qfx2_aid = np.tile(idx2_daid.take(_idxs), (len(qfxs), 1))
    #qfx2_fx = np.tile(idx2_dfx.take(_idxs), (len(qfxs), 1))


    #if __debug__:
    #    for wx in wx2_drvecs.keys():
    #        assert wx2_drvecs[wx].shape[0] == wx2_idxs[wx].shape[0]


    #qfx2_axs = []
    #qfx2_fm = []
    #qfx2_fs = []
    #aid_fm = []
    #aid_fs = []

    #idx2_dfx  = invindex.idx2_dfx
    #idx2_wx  = invindex.idx2_wx
    #wx2_idxs_series = {wx: pd.Series(idxs, name='idx') for wx, idxs in
    #                   six.iteritems(invindex.wx2_idxs)}
    #wx2_qfxs_series = {wx: pd.Series(qfx, name='qfx') for wx, qfx in
    #                   six.iteritems(wx2_qfxs)}
    #qfx2_idx = np.tile(_idxs, (len(qfxs), 1))
    #qfx2_aid = np.tile(idx2_daid.take(_idxs), (len(qfxs), 1))
    #qfx2_fx = np.tile(idx2_dfx.take(_idxs), (len(qfxs), 1))

    #qfxs   = wx2_qfxs[wx]
    #_idxs  = wx2_idxs[wx]

