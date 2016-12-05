    # def get_rank_cumhist(testres, bins='dense'):
    #     hist_list, edges = testres.get_rank_histograms(bins, asdict=False)
    #     config_cdfs = np.cumsum(hist_list, axis=1)
    #     return config_cdfs, edges


    # def get_rank_histogram_bin_edges(testres):
    #     bins = testres.get_rank_histogram_bins()
    #     bin_keys = list(zip(bins[:-1], bins[1:]))
    #     return bin_keys

    # def get_rank_histogram_qx_binxs(testres):
    #     rank_mat = testres.get_rank_mat()
    #     config_hists = testres.get_rank_histograms()
    #     config_binxs = []
    #     bin_keys = testres.get_rank_histogram_bin_edges()
    #     for hist_dict, ranks in zip(config_hists, rank_mat.T):
    #         bin_qxs = [np.where(np.logical_and(low <= ranks, ranks < high))[0]
    #                    for low, high in bin_keys]
    #         qx2_binx = -np.ones(len(ranks))
    #         for binx, qxs in enumerate(bin_qxs):
    #             qx2_binx[qxs] = binx
    #         config_binxs.append(qx2_binx)
    #     return config_binxs

    # def get_rank_histogram_qx_sample(testres, size=10):
    #     size = 10
    #     rank_mat = testres.get_rank_mat()
    #     config_hists = testres.get_rank_histograms()
    #     config_rand_bin_qxs = []
    #     bins = testres.get_rank_histogram_bins()
    #     bin_keys = list(zip(bins[:-1], bins[1:]))
    #     randstate = np.random.RandomState(seed=0)
    #     for hist_dict, ranks in zip(config_hists, rank_mat.T):
    #         bin_qxs = [np.where(np.logical_and(low <= ranks, ranks < high))[0]
    #                    for low, high in bin_keys]
    #         rand_bin_qxs = [qxs if len(qxs) <= size else
    #                         randstate.choice(qxs, size=size, replace=False)
    #                         for qxs in bin_qxs]
    #         config_rand_bin_qxs.append(rand_bin_qxs)
    #     return config_rand_bin_qxs
