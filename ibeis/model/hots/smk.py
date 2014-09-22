"""
smk
Todo:
    * SIFT: Root_SIFT -> L2 normalized -> Centering.
    # http://hal.archives-ouvertes.fr/docs/00/84/07/21/PDF/RR-8325.pdf

    The devil is in the deatails
    http://www.robots.ox.ac.uk/~vilem/bmvc2011.pdf

    This says dont clip, do rootsift instead
    # http://hal.archives-ouvertes.fr/docs/00/68/81/69/PDF/hal_v1.pdf

    * Quantization of residual vectors

    * Burstiness normalization for N-SMK

    * Implemented A-SMK

    * Incorporate Spatial Verification

    * Implement correct cfgstrs based on algorithm input
    for cached computations.

    * Go pandas all the way

    * Color by word

    * Profile on hyrule

    * Train vocab on paris

    * Remove self matches.

    * New SIFT parameters for pyhesaff (root, powerlaw, meanwhatever, output_dtype)

Issues:
    * 10GB are in use when performing query on Oxford 5K
    * errors when there is a word without any database vectors.
    currently a weight of zero is hacked in


Paper Style Guidelines:
   * use real code examples instead of pseudocode
    (show off power of python)
   * short and consice
   * never cryptic

Paper outline:

abstract:
    contributions:

algorithms:
    lnbnn
    a/smk
    modification (name scoring? next level categorization)

parameters:
    database size
    sift threshold
    vocabulary?

Databases:
    pzall
    gzall
    oxford
    paris

"""
from __future__ import absolute_import, division, print_function
import six
import ibeis
import utool
import numpy as np
import pandas as pd
from vtool import clustering2 as clustertool
from ibeis.model.hots import query_request
from ibeis.model.hots import smk_index
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[smk]')


def testdata():
    qaid = 37  # NOQA new test case for PZ_MTEST
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
    daids = valid_aids[1:10]
    # Search set
    #qaids = valid_aids[0::2]
    qaids = [valid_aids[0], valid_aids[4]]
    #default = 1E3
    default = 8E3
    #default = 5E2
    #default = 2E4
    #default=5)  # default=95000)
    nWords = utool.get_arg(('--nWords', '--nCentroids'), int, default=default)
    return ibs, annots_df, taids, daids, qaids, nWords


def main():
    """
    >>> from ibeis.model.hots.smk import *  # NOQA
    """
    ibs, annots_df, taids, daids, qaids, nWords = testdata()
    # Learn vocabulary
    words = smk_index.learn_visual_words(annots_df, taids, nWords)
    # Index a database of annotations
    invindex = smk_index.index_data_annots(annots_df, daids, words)
    # Query using SMK
    #qaid = qaids[0]
    qreq_ = query_request.new_ibeis_query_request(ibs, qaids, daids)
    # Smk Mach
    qaid2_qres_ = smk_index.query_smk(ibs, annots_df, invindex, qreq_)
    for count, (qaid, qres) in enumerate(six.iteritems(qaid2_qres_)):
        print('+================')
        #qres = qaid2_qres_[qaid]
        qres.show_top(ibs, fnum=count)
        print(qres.get_inspect_str(ibs))
        print('L================')
    #print(qres.aid2_fs)
    #daid2_totalscore, chipmatch = smk_index.query_inverted_index(annots_df, qaid, invindex)
    ## Pack into QueryResult
    #qaid2_chipmatch = {qaid: chipmatch}
    #qaid2_qres_ = pipeline.chipmatch_to_resdict(qaid2_chipmatch, {}, qreq_)
    ## Show match
    #daid2_totalscore.sort(axis=1, ascending=False)
    #print(daid2_totalscore)

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
    import multiprocessing
    from plottool import draw_func2 as df2
    np.set_printoptions(precision=2)
    pd.set_option('display.max_rows', 7)
    pd.set_option('display.max_columns', 7)
    pd.set_option('isplay.notebook_repr_html', True)
    multiprocessing.freeze_support()  # for win32
    main_locals = main()
    main_execstr = utool.execstr_dict(main_locals, 'main_locals')
    exec(main_execstr)
    exec(df2.present())
