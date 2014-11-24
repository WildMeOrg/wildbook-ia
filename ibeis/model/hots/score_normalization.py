"""
TODO:
    1) vsmany
       * works resaonable for very few and very many
       * stars with small k and then k becomes a percent or log percent
       * distinctiveness from different location

    2) 1-vs-1
       * uses distinctivness and foreground when available
       * start with ratio test and ransac

    3) First N decision are interactive until we learn a good threshold

    4) Always show numbers between 0 and 1 spatial verification is based on
    single best exemplar

    5) Add exemplars that are distinct from exiting (matches below threshold)

    (no rebuilding ing kd-tree for each image)


"""
from __future__ import absolute_import, division, print_function
from os.path import join, split, exists, basename
import utool
import numpy as np
import utool as ut
import six  # NOQA
import cPickle
from zipfile import error as BadZipFile  # Screwy naming convention.
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[scorenorm]', DEBUG=False)


class Savable(object):
    def __init__(self):
        pass

    def load(self, cachedir, verbose=True or ut.VERBOSE, force_miss=False):
        """ Loads the result from the given database """
        fpath = self.get_fpath(cachedir)
        if verbose:
            print('[qr] cache tryload: %r' % (basename(fpath),))
        try:
            with open(fpath, 'rb') as file_:
                loaded_dict = cPickle.load(file_)
                self.__dict__.update(loaded_dict)
            if verbose:
                print('... self cache hit: %r' % (basename(fpath),))
        except IOError as ex:
            if not exists(fpath):
                msg = '... self cache miss: %r' % (basename(fpath),)
                if verbose:
                    print(msg)
                raise
            msg = '[!qr] QueryResult(qaid=%d) is corrupt' % (self.qaid)
            utool.printex(ex, msg, iswarning=True)
            raise
        except BadZipFile as ex:
            msg = '[!qr] QueryResult(qaid=%d) has bad zipfile' % (self.qaid)
            utool.printex(ex, msg, iswarning=True)
            raise
            #if exists(fpath):
            #    #print('[qr] Removing corrupted file: %r' % fpath)
            #    #os.remove(fpath)
            #    raise hsexcept.HotsNeedsRecomputeError(msg)
            #else:
            #    raise Exception(msg)
        except Exception as ex:
            utool.printex(ex, 'unknown exception while loading query result')
            raise
    #def get_fname(self):
    #    return 'saveable'

    def get_fpath(self, cachedir):
        fpath = join(cachedir, self.get_fname())
        return fpath

    def save(self, cachedir, verbose=True or ut.VERBOSE):
        """
        saves query result to directory
        """
        fpath = self.get_fpath(cachedir)
        if verbose:
            print('[qr] cache save: %r' % (split(fpath)[1],))
        with open(fpath, 'wb') as file_:
            cPickle.dump(self.__dict__, file_)


class ScoreNormalizer(Savable):
    def __init__(self, cfgstr, score_domain=None, p_tp_given_score=None):
        super(ScoreNormalizer, self).__init__()
        self.cfgstr = cfgstr
        self.set_values(score_domain, p_tp_given_score)

    def set_values(self, score_domain, p_tp_given_score):
        self.score_domain = score_domain
        self.p_tp_given_score = p_tp_given_score

    def normalize_score(self, score):
        if score > self.score_domain[-1]:
            return 1.0
        else:
            index = np.nonzero(self.score_domain < score)[0]
            return self.p_tp_given_score[index]

    def get_fname(self):
        if self.cfgstr is None:
            raise AssertionError('Must specify cfgstr')
        return self.cfgstr + '_scorenorm.cPkl'

    def __call__(self, score_list):
        prob_list = [self.normalize_score(score) for score in score_list]
        return prob_list


def learn_score_normalizer(good_tn, good_tp, cfgstr=None):
    #clip_score = 2000
    max_true_negative_score = good_tn.max()
    max_true_positive_score = good_tp.max()
    overshoot_factor = max_true_positive_score / max_true_negative_score
    if overshoot_factor > 3:
        clip_score = max_true_negative_score * overshoot_factor
    else:
        clip_score = max_true_positive_score
    score_tp_pdf = ut.estimate_pdf(good_tp, gridsize=512, adjust=8)
    score_tn_pdf = ut.estimate_pdf(good_tn, gridsize=512, adjust=8)
    score_domain = np.linspace(0, clip_score, 1024)
    p_score_given_tp = score_tp_pdf.evaluate(score_domain)
    p_score_given_tn = score_tn_pdf.evaluate(score_domain)
    p_score = np.array(p_score_given_tp) + np.array(p_score_given_tn)
    # Apply bayes
    p_tp = .5
    p_tp_given_score = ut.bayes_rule(p_score_given_tp, p_score, p_tp)
    normalizer = ScoreNormalizer(cfgstr, score_domain, p_tp_given_score)
    return normalizer

    #p_tn = 1.0 - p_tp
    #p_tn_given_score = ut.bayes_rule(p_score_given_tn, p_score, p_tn)
    #lbl = 'score'  # NOQA
    #inspect_pdfs('score', score_domain, good_tn, good_tp, p_score_given_tn, p_score_given_tp, p_score, p_tn_given_score, p_tp_given_score)


def tryload_score_normalizer(cachedir, cfgstr):
    normalizer = ScoreNormalizer(cfgstr)
    try:
        normalizer.load(cachedir)
        return normalizer
    except Exception:
        #ut.printex(ex)
        return None


def learn_ibeis_score_normalizer(ibs, qaid_list, qres_list):
    """
    Args:
        qaid2_qres (int): query annotation id

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.score_normalization import *   # NOQA
        >>> import ibeis
        >>> dbname = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(dbname)
        >>> qaid_list = daid_list = ibs.get_valid_aids()
        >>> cfgdict = dict(codename='nsum')
        >>> qres_list = ibs.query_chips(qaid_list, daid_list, cfgdict)
        >>> score_normalizer = learn_ibeis_score_normalizer(ibs, qaid_list, qres_list)
        >>> result = score_normalizer.get_fname()
        >>> print(result)
        PZ_MTEST_UUIDS((119)htc%i42+w9plda&d)_scorenorm.cPkl
    """
    # Collect training data
    cfgstr = ibs.get_dbname() + ibs.get_annot_uuid_hashid(qaid_list)
    normalizer = tryload_score_normalizer(ibs.cachedir, cfgstr)
    if normalizer is not None:
        print('returning cached normalizer')
        return normalizer
    print('computing normalizer')
    good_tp_nscores = []
    good_tn_nscores = []
    good_tp_aidnid_pairs = []
    good_tn_aidnid_pairs = []
    for qx, qres in enumerate(qres_list):
        qaid = qres.get_qaid()
        if not qres.is_nsum():
            raise AssertionError('must be nsum')
        if not ibs.get_annot_has_groundtruth(qaid):
            continue
        qnid = ibs.get_annot_nids(qres.get_qaid())

        nscoretup = qres.get_nscoretup(ibs)
        (sorted_nids, sorted_nscores, sorted_aids, sorted_scores) = nscoretup

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
                good_tp_aidnid_pairs.append((qaid, sorted_nids[gt_rank]))
                good_tn_aidnid_pairs.append((qaid, sorted_nids[gf_rank]))
    good_tp = np.array(good_tp_nscores)
    good_tn = np.array(good_tn_nscores)
    normalizer = learn_score_normalizer(good_tp, good_tn, cfgstr)
    normalizer.save(ibs.cachedir)
    return normalizer


def inspect_pdfs(lbl, score_domain, good_tn, good_tp, p_score_given_tn, p_score_given_tp, p_score, p_tn_given_score, p_tp_given_score):
    import plottool as pt  # NOQA

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

if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, ibeis.model.hots.score_normalization; utool.doctest_funcs(ibeis.model.hots.score_normalization, allexamples=True)"
        python -c "import utool, ibeis.model.hots.score_normalization; utool.doctest_funcs(ibeis.model.hots.score_normalization)"
        python ibeis\model\hots\score_normalization.py
        python ibeis\model\hots\score_normalization.py --allexamples
        python ibeis\model\hots\score_normalization.py --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
