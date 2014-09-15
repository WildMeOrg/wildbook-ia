"""
smk
Todo:
    * Implement correct cfgstrs based on algorithm input
    for cached computations.

    * Go pandas all the way
Issues:
    * errors when there is a word without any database vectors.
    currently a weight of zero is hacked in

"""
from __future__ import absolute_import, division, print_function
import ibeis
import utool
import numpy as np
import numpy.linalg as npl  # NOQA
import pandas as pd
from vtool import clustering2 as clustertool
from plottool import draw_func2 as df2
from ibeis.model.hots import pipeline
from ibeis.model.hots import query_request as hsqreq
import smk_index
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk]')

np.set_printoptions(precision=2)
pd.set_option('display.max_rows', 7)
pd.set_option('display.max_columns', 7)
pd.set_option('isplay.notebook_repr_html', True)


def testdata():
    np.set_printoptions(precision=2)
    pd.set_option('display.max_rows', 7)
    pd.set_option('display.max_columns', 7)
    pd.set_option('isplay.notebook_repr_html', True)
    ibeis.ensure_pz_mtest()
    ibs = ibeis.opendb('PZ_MTEST')
    # Pandas Annotation Dataframe
    annots_df = smk_index.make_annot_df(ibs)
    valid_aids = annots_df.index
    # Training set
    taids = valid_aids[:]
    # Database set
    daids = valid_aids[1:]
    # Search set
    #qaids = valid_aids[0::2]
    qaids = valid_aids[0:1]
    default = 1E3
    #default = 2E4
    #default=5)  # default=95000)
    nWords = utool.get_arg(('--nWords', '--nCentroids'), int, default=default)
    return ibs, annots_df, taids, daids, qaids, nWords


def main():
    """
    >>> from smk import *  # NOQA
    """
    ibs, annots_df, taids, daids, qaids, nWords = testdata()
    # Learn vocabulary
    words = smk_index.learn_visual_words(annots_df, taids, nWords)
    # Index a database of annotations
    invindex = smk_index.index_data_annots(annots_df, daids, words)
    # Query using SMK
    qaid = qaids[0]
    qreq_ = hsqreq.new_ibeis_query_request(ibs, [qaid], daids)
    # Smk Mach
    daid2_totalscore, chipmatch = smk_index.query_inverted_index(annots_df, qaid, invindex)
    daid2_totalscore.sort(axis=1, ascending=False)
    print(daid2_totalscore)
    # Pack into QueryResult
    qaid2_chipmatch = {qaid: chipmatch}
    qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, {}, qreq_)
    qres = qaid2_qres_[qaid]
    # Show match
    qres.show_top(ibs)

    #daid2_totalscore2, chipmatch = query_inverted_index(annots_df, daids[0], invindex)
    #print(daid2_totalscore2)
    #display_info(ibs, invindex, annots_df)
    print('finished main')
    return locals()


def display_info(ibs, invindex, annots_df):
    ################
    from ibeis.dev import dbinfo
    print(ibs.get_infostr())
    dbinfo.get_dbinfo(ibs, verbose=True)
    ################
    print('Inverted Index Stats: vectors per word')
    print(utool.stats_str(map(len, invindex.wx2_idxs.values())))
    ################
    #qfx2_vec     = annots_df['vecs'][1]
    centroids    = invindex.words
    num_pca_dims = 2  # 3
    whiten       = False
    kwd = dict(num_pca_dims=num_pca_dims,
               whiten=whiten,)
    #clustertool.rrr()
    def makeplot_(fnum, prefix, data, labels='centroids', centroids=centroids):
        return clustertool.plot_centroids(data, centroids, labels=labels,
                                          fnum=fnum, prefix=prefix + '\n', **kwd)
    makeplot_(1, 'centroid vecs', centroids)
    #makeplot_(2, 'database vecs', invindex.idx2_dvec)
    #makeplot_(3, 'query vecs', qfx2_vec)
    #makeplot_(4, 'database vecs', invindex.idx2_dvec)
    #makeplot_(5, 'query vecs', qfx2_vec)
    #################


if __name__ == '__main__':
    main_locals = main()
    main_execstr = utool.execstr_dict(main_locals, 'main_locals')
    exec(main_execstr)
    exec(df2.present())
