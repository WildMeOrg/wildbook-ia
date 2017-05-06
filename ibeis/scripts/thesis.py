from ibeis.scripts.script_vsone import OneVsOneProblem
import pandas as pd
import numpy as np
from os.path import basename, join, splitext, exists
import utool as ut
import plottool as pt
import vtool as vt
import pathlib
import matplotlib as mpl
import random
import sys
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP  # NOQA

DPI = 300

TMP_RC = {
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'font.family': 'DejaVu Sans',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    # 'legend.fontsize': 18,
    # 'legend.alpha': .8,
    'legend.fontsize': 12,
    'legend.facecolor': 'w',
}

TMP_RC = {
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'font.family': 'sans-serif',
    'font.serif': 'CMU Serif',
    'font.sans-serif': 'CMU Sans Serif',
    'font.monospace': 'CMU Typewriter Text',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    # 'legend.alpha': .8,
    'legend.fontsize': 12,
    'legend.facecolor': 'w',
}

# W, H = 7.4375, 3.125
W, H = 7.4375, 3.0
# W, H = W * 1.25, H * 1.25


def measure_worker(dbname):
    self = Chap3(dbname)
    self.measure_all()


@ut.reloadable_class
class Chap3Commands(object):

    @classmethod
    def run_all(Chap3):
        """
        CommandLine:
            python -m ibeis.scripts.thesis Chap3.run_all
        """
        agg_dbnames = ['PZ_Master1', 'GZ_Master1', 'GIRM_Master1', 'humpbacks_fb']
        agg_dbnames = agg_dbnames[::-1]

        if True:
            for dbname in agg_dbnames:
                self = Chap3(dbname)
                self.measure_all()
                self.draw_time_distri()
        else:
            from concurrent import futures
            executor = futures.ProcessPoolExecutor(len(agg_dbnames))
            print('About to submit')
            fs = [executor.submit(measure_worker, dbname) for dbname in agg_dbnames]
            while True:
                print('waiting')
                print('fs = %r' % (fs,))
            futures.wait(fs)

            for dbname in agg_dbnames:
                self = Chap3(dbname)
                self.draw_time_distri()

        Chap3.agg_db_stats()
        Chap3.draw_agg_baseline()

    def measure_all(self):
        """
        Example:
            from ibeis.scripts.thesis import *
            self = Chap3('PZ_Master1')
            self.measure_all()
            self = Chap3('GZ_Master1')
            self.measure_all()
            self = Chap3('GIRM_Master1')
            self.measure_all()
        """
        if self.ibs is None:
            self._precollect()
        self.measure_baseline()
        if self.dbname in {'PZ_Master1', 'GZ_Master1'}:
            self.measure_foregroundness()
            self.measure_smk()
            self.measure_nsum()
            self.measure_dbsize()
            self.measure_kexpt()
            self.measure_invariance()


@ut.reloadable_class
class Chap3Agg(object):

    @classmethod
    def agg_dbstats(Chap3):
        """
        CommandLine:
            python -m ibeis.scripts.thesis Chap3.agg_db_stats
        """
        agg_dbnames = ['PZ_Master1', 'GZ_Master1', 'GIRM_Master1', 'humpbacks_fb']
        infos = ut.ddict(list)
        for dbname in agg_dbnames:
            self = Chap3(dbname)
            info = self.ensure_results('dbstats')
            infos['enc'].append(info['enc'])
            infos['qual'].append(info['qual'])
            infos['view'].append(info['view'])
            # labels.append(self.species_nice.capitalize())

        df = pd.DataFrame(infos['enc'])
        # df = df.reindex_axis(ut.partial_order(df.columns, ['species_nice']), axis=1)
        df = df.rename(columns={'species_nice': 'Database'})
        text = df.to_latex(index=False).replace('±', '\pm')
        text = text.replace(r'n\_singleton\_names', r'\thead{\#names\\(singleton)}')
        text = text.replace(r'n\_resighted\_names', r'\thead{\#names\\(resighted)}')
        text = text.replace(r'n\_encounter\_per\_resighted\_name', r'\thead{\#encounter per\\name (resighted)}')
        text = text.replace(r'n\_annots\_per\_encounter', r'\thead{\#annots per\\encounter}')
        text = text.replace(r'n\_annots', r'\thead{\#annots}')
        enc_text = text.replace('lrrllr', 'lrrrrr')
        # ut.render_latex_text(text, preamb_extra=r'\usepackage{makecell}')

        df = pd.DataFrame(infos['qual'])
        df = df.rename(columns={'species_nice': 'Database'})
        df = df.reindex_axis(ut.partial_order(
            df.columns, ['Database', 'excellent', 'good', 'ok', 'poor', 'None']), axis=1)
        qual_text = df.to_latex(index=False)

        df = pd.DataFrame(infos['view'])
        df = df.rename(columns={
            'species_nice': 'Database',
            'back': 'B', 'left': 'L', 'right': 'R', 'front': 'F',
            'backleft': 'BL', 'backright': 'BR', 'frontright': 'FR',
            'frontleft': 'FL',
        })
        order = ut.partial_order(
            df.columns, ['Database', 'BL', 'L', 'FL', 'F', 'FR', 'R', 'BR',
                         'B', 'None'])
        df = df.reindex_axis(order, axis=1)
        df = df.set_index('Database')
        df[pd.isnull(df)] = 0
        df = df.astype(np.int).reset_index()
        view_text = df.to_latex(index=False)

        ut.write_to(join(Chap3.base_dpath, 'agg-enc.tex'), enc_text)
        ut.write_to(join(Chap3.base_dpath, 'agg-view.tex'), view_text)
        ut.write_to(join(Chap3.base_dpath, 'agg-qual.tex'), qual_text)

    @classmethod
    def draw_agg_baseline(Chap3):
        """
        CommandLine:
            python -m ibeis.scripts.thesis Chap3.draw_agg_baseline --diskshow
        """
        agg_dbnames = ['GZ_Master1', 'PZ_Master1', 'GIRM_Master1', 'humpbacks_fb']
        cdfs = []
        labels = []
        for dbname in agg_dbnames:
            self = Chap3(dbname)
            results = self.ensure_results('baseline')
            cdf, config = results[0]
            dsize = config['dsize']
            qsize = config['t_n_names']
            baseline_cdf = results[0][0]
            cdfs.append(baseline_cdf)
            labels.append('{},qsize={},dsize={}'.format(
                self.species_nice, qsize, dsize))
            # labels.append(self.species_nice.capitalize())

        fig = self.plot_cmcs2(cdfs, labels, fnum=1, ymin=.5)
        fig.set_size_inches([W, H * 1.5])
        fpath = join(Chap3.base_dpath, 'agg-baseline.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        if ut.get_argflag('--diskshow'):
            ut.startfile(fpath)

    def ensure_results(self, expt_name=None):
        if expt_name is None:
            # Load all
            fpaths = ut.glob(self.dpath, '*.pkl')
            expt_names = [splitext(basename(fpath))[0] for fpath in fpaths]
            for fpath, expt_name in zip(fpaths, expt_names):
                self.expt_results[expt_name] = ut.load_data(fpath)
        else:
            fpath = join(self.dpath, expt_name + '.pkl')
            expt_name = splitext(basename(fpath))[0]
            if not exists(fpath):
                if self.ibs is None:
                    self._precollect()
                getattr(self, 'measure_' + expt_name)()
            self.expt_results[expt_name] = ut.load_data(fpath)
            return self.expt_results[expt_name]


@ut.reloadable_class
class Chap3Inputs(object):
    """
        humpbacks_fb
        Seals
        snails_drop1
        LF_Bajo_bonito
    """
    base_dpath = ut.truepath('~/latex/crall-thesis-2017/figuresY')

    def __init__(self, dbname=None):
        self.dbname = dbname
        if 'GZ' in dbname:
            self.species_nice = "Grévy's zebras"
        if 'PZ' in dbname:
            self.species_nice = "plains zebras"
        if 'GIRM' in dbname:
            self.species_nice = "Masai Giraffes"
        if 'humpback' in dbname:
            self.species_nice = "Humpbacks"
        self.expt_results = {}
        self.ibs = None
        if dbname is not None:
            self.dpath = join(self.base_dpath, self.dbname)
            ut.ensuredir(self.dpath)

    def _precollect(self):
        """
        Example:
            >>> from ibeis.scripts.thesis import *
            >>> self = Chap3('humpbacks_fb')
            >>> self = Chap3('GZ_Master1')
            >>> self = Chap3('GIRM_Master1')
            >>> self = Chap3('PZ_MTEST')
            >>> self = Chap3('PZ_PB_RF_TRAIN')
            >>> self = Chap3('PZ_Master1')
            >>> self._precollect()
        """
        import ibeis
        from ibeis.init import main_helpers
        self.dbdir = ibeis.sysres.lookup_dbdir(self.dbname)
        ibs = ibeis.opendb(dbdir=self.dbdir)
        if ibs.dbname.startswith('PZ_Master'):
            aids = ibs.filter_annots_general(require_timestamp=True, is_known=True,
                                             # require_viewpoint=True,
                                             # view='left',
                                             species='primary',
                                             # view_ext=2,  # FIXME
                                             min_pername=2, minqual='poor')
            # flags = ['right' not in text for text in ibs.annots(aids).yaw_texts]
            flags = ['left' in text for text in ibs.annots(aids).yaw_texts]
            # sum(['left' == text for text in ibs.annots(aids).yaw_texts])
            aids = ut.compress(aids, flags)
        # elif ibs.dbname == 'GZ_Master1':
        else:
            aids = ibs.filter_annots_general(require_timestamp=True,
                                             is_known=True,
                                             species='primary',
                                             # require_viewpoint=True,
                                             # view='right',
                                             # view_ext2=2,
                                             # view_ext1=2,
                                             minqual='poor')
            # flags = ['left' not in text for text in ibs.annots(aids).yaw_texts]
            # aids = ut.compress(aids, flags)
        ibs.print_annot_stats(aids, prefix='P')
        main_helpers.monkeypatch_encounters(ibs, aids, minutes=30)
        print('post monkey patch')
        ibs.print_annot_stats(aids, prefix='P')
        self.ibs = ibs
        self.aids_pool = aids
        if False:
            # check encounter stats
            annots = ibs.annots(aids)
            encounters = annots.group(annots.encounter_text)[1]
            nids = ut.take_column(ibs._annot_groups(encounters).nids, 0)
            nid_to_enc = ut.group_items(encounters, nids)
            nenc_list = ut.lmap(len, nid_to_enc.values())
            hist = ut.range_hist(nenc_list, [1, 2, 3, (4, np.inf)])
            print('enc per name hist:')
            print(ut.repr2(hist))

            # singletons = [a for a in encounters if len(a) == 1]
            multitons = [a for a in encounters if len(a) > 1]
            deltas = []
            for a in multitons:
                times = a.image_unixtimes_asfloat
                deltas.append(max(times) - min(times))
            ut.lmap(ut.get_posix_timedelta_str, sorted(deltas))

    @staticmethod
    def _alt_splits(ibs, aids, qenc_per_name, denc_per_name_, annots_per_enc):
        """ This cannot be used for cross validation """
        # Group annotations by encounter
        # from ibeis.other import ibsfuncs
        # primary_view = ibsfuncs.get_primary_species_viewpoint(ibs.get_primary_database_species())

        annots = ibs.annots(aids)
        encounters = ibs._annot_groups(annots.group(annots.encounter_text)[1])
        enc_nids = ut.take_column(encounters.nids, 0)
        nid_to_encs = ut.group_items(encounters, enc_nids)
        rng = ut.ensure_rng(0)
        pyrng = random.Random(rng.randint(sys.maxsize))

        n_need = qenc_per_name + denc_per_name_

        confusor_encs = {}
        sample_splits = {}

        for nid, encs in nid_to_encs.items():
            if len(encs) < n_need:
                confusor_encs[nid] = encs
            else:
                # For each name choose a query / database encounter.
                chosen_encs = pyrng.sample(encs, n_need)
                ibs._annot_groups(chosen_encs).aids
                # Choose high quality annotations from each encounter
                best_subencs = []
                for enc in chosen_encs:
                    if len(enc) > annots_per_enc:
                        sortx = ut.argsort(enc.qualities)[::-1]
                        subenc = enc.take(sortx[0:annots_per_enc])
                        best_subencs.append(subenc)
                    else:
                        assert len(enc) == annots_per_enc
                        best_subencs.append(enc)
                ibs._annot_groups(best_subencs).aids
                qsubenc = [a.aids for a in best_subencs[0:qenc_per_name]]
                dsubenc = [a.aids for a in best_subencs[qenc_per_name:]]
                sample_splits[nid] = (qsubenc, dsubenc)

        # make confusor encounters subject to the same constraints
        confusor_pool = []
        for encs in confusor_encs.values():
            confusor_pool.extend(
                ut.flatten([enc[0:annots_per_enc].aids for enc in encs])
            )

        qaids = ut.total_flatten(ut.take_column(sample_splits.values(), 0))
        dname_encs = ut.take_column(sample_splits.values(), 1)
        return qaids, dname_encs, confusor_pool

    def _same_occur_split(self):
        """
            >>> from ibeis.scripts.thesis import *
            >>> self = Chap3('PZ_Master1')
            >>> self._precollect()
        """
        ibs = self.ibs
        aids = self.aids_pool
        annots = ibs.annots(aids)
        # occurrences = ibs._annot_groups(annots.group(annots.occurrence_text)[1])
        encounters = ibs._annot_groups(annots.group(annots.encounter_text)[1])

        nid_to_splits = ut.ddict(list)
        # Find the biggest occurrences and pick an annotation from that
        # occurrence to be sampled
        occur_to_encs = ut.group_items(encounters, ut.take_column(encounters.occurrence_text, 0))
        occur_encs = ut.sortedby(list(occur_to_encs.values()), list(map(len, occur_to_encs.values())))[::-1]
        for encs in occur_encs:
            for enc in encs:
                sortx = ut.argsort(enc.qualities)[::-1]
                annot = enc[sortx[0]]
                if len(nid_to_splits[annot.nid]) < 2:
                    nid_to_splits[annot.nid].append(annot.aid)

        rng = ut.ensure_rng(0)
        pyrng = random.Random(rng.randint(sys.maxsize))

        qaids = []
        dname_encs = []
        confusor_pool = []
        for nid, aids_ in nid_to_splits.items():
            if len(aids_) < 2:
                confusor_pool.extend(aids_)
            else:
                pyrng.shuffle(aids_)
                qaids.append(aids_[0])
                dname_encs.append([[aids_[1]]])
        confusor_pool = ut.shuffle(confusor_pool, rng=0)
        return qaids, dname_encs, confusor_pool

    def _rand_splits(ibs, aids, qenc_per_name, denc_per_name_, annots_per_enc):
        """ This can be used for cross validation """
        # Find a split of query/database encounters and confusors
        from ibeis.init.filter_annots import encounter_crossval
        enc_splits, nid_to_confusors = encounter_crossval(
            ibs, aids, qenc_per_name=1, annots_per_enc=1,
            denc_per_name=denc_per_name_, rebalance=True, rng=0, early=True)

        qname_encs, dname_encs = enc_splits[0]
        qaids = sorted(ut.flatten(ut.flatten(qname_encs)))
        confusor_pool = ut.flatten(ut.flatten(nid_to_confusors.values()))
        confusor_pool = ut.shuffle(confusor_pool, rng=0)
        return qaids, dname_encs, confusor_pool

    def _varied_inputs(self, denc_per_name=[1], extra_dbsize_fracs=None,
                       method='alt'):
        """
        Vary num per name and total number of annots

        Example:
            >>> from ibeis.scripts.thesis import *  # NOQA
            >>> self = Chap3('PZ_Master1')
            >>> self._precollect()
            >>> print('--------')
            >>> ibs, qaids, daids_list, info_list = self._varied_inputs([1], [1], method='same_occur')
            >>> print('qsize = %r' % (len(qaids),))
            >>> for info in info_list:
            >>>     print(ut.repr4(info))
            >>> print('--------')
            >>> ibs, qaids, daids_list, info_list = self._varied_inputs([1, 2], [0])
            >>> print('qsize = %r' % (len(qaids),))
            >>> for info in info_list:
            >>>     print(ut.repr4(info))
            >>> print('--------')
            >>> ibs, qaids, daids_list, info_list = self._varied_inputs([3], [0, 1])
            >>> print('qsize = %r' % (len(qaids),))
            >>> for info in info_list:
            >>>     print(ut.repr4(info))

        Ignore:
            ibs, aids = self.ibs, self.aids_pool
            denc_per_name = 1
            denc_per_name = [1]
            extra_dbsize_fracs = None

            extra_dbsize_fracs = [0, .5, 1]
        """
        ibs, aids = self.ibs, self.aids_pool
        qenc_per_name = 1
        annots_per_enc = 1
        denc_per_name_ = max(denc_per_name)

        if method == 'alt':
            qaids, dname_encs, confusor_pool = self._alt_splits(
                ibs, aids, qenc_per_name, denc_per_name_, annots_per_enc)
        elif method == 'same_occur':
            assert denc_per_name_ == 1
            assert annots_per_enc == 1
            assert qenc_per_name == 1
            qaids, dname_encs, confusor_pool = self._same_occur_split()
        else:
            raise KeyError(method)

        # Vary the number of database encounters in each sample
        target_daids_list = []
        target_info_list_ = []
        for num in denc_per_name:
            dname_encs_ = ut.take_column(dname_encs, slice(0, num))
            dnames_ = ut.lmap(ut.flatten, dname_encs_)
            daids_ = ut.flatten(dnames_)
            target_daids_list.append(daids_)
            name_lens = ut.lmap(len, dnames_)
            dpername = (name_lens[0] if ut.allsame(name_lens) else
                        np.mean(name_lens))
            target_info_list_.append(ut.odict([
                ('t_n_names', len(dname_encs_)),
                ('t_dpername', dpername),
                ('t_denc_pername', num),
                ('t_dsize', len(daids_)),
            ]))

        # Append confusors to maintain a constant dbsize in each base sample
        dbsize_list = ut.lmap(len, target_daids_list)
        max_dsize = max(dbsize_list)
        n_need = max_dsize - min(dbsize_list)
        n_extra_avail = len(confusor_pool) - n_need
        assert len(confusor_pool) > n_need, 'not enough confusors'
        padded_daids_list = []
        padded_info_list_ = []
        for daids_, info_ in zip(target_daids_list, target_info_list_):
            num_take = max_dsize - len(daids_)
            pad_aids = confusor_pool[:num_take]
            new_aids = daids_ + pad_aids
            info_ = info_.copy()
            info_['n_pad'] = len(pad_aids)
            info_['pad_dsize'] = len(new_aids)
            padded_info_list_.append(info_)
            padded_daids_list.append(new_aids)

        # Vary the dbsize by appending extra confusors
        if extra_dbsize_fracs is None:
            extra_dbsize_fracs = [1.]
        extra_fracs = np.array(extra_dbsize_fracs)
        n_extra_list = np.unique(extra_fracs * n_extra_avail).astype(np.int)
        daids_list = []
        info_list = []
        for n in n_extra_list:
            for daids_, info_ in zip(padded_daids_list, padded_info_list_):
                extra_aids = confusor_pool[len(confusor_pool) - n:]
                daids = sorted(daids_ + extra_aids)
                daids_list.append(daids)
                info = info_.copy()
                info['n_extra'] = len(extra_aids)
                info['dsize'] = len(daids)
                info_list.append(info)

        import pandas as pd
        verbose = 0
        if verbose:
            print(pd.DataFrame.from_records(info_list))
            print('#qaids = %r' % (len(qaids),))
            print('num_need = %r' % (n_need,))
            print('max_dsize = %r' % (max_dsize,))
        return self.ibs, qaids, daids_list, info_list


@ut.reloadable_class
class Chap3Measures(object):

    def measure_baseline(self):
        """
            >>> from ibeis.scripts.thesis import *
            >>> self = Chap3.collect('PZ_MTEST')
        """
        ibs, qaids, daids_list, info_list = self._varied_inputs(
            denc_per_name=[1], extra_dbsize_fracs=[1])
        cfgdict = {}
        daids = daids_list[0]
        info = info_list[0]
        cdf = self._exec_ranking(ibs, qaids, daids, cfgdict)
        results = [(cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict}))]

        expt_name = 'baseline'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_foregroundness(self):
        ibs, qaids, daids_list, info_list = self._varied_inputs(
            denc_per_name=[1], extra_dbsize_fracs=[1], method='same_occur')
        daids = daids_list[0]
        info = info_list[0]

        results = []
        cfgdict1 = {'fg_on': False}
        cdf = self._exec_ranking(ibs, qaids, daids, cfgdict1)
        results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict1})))

        cfgdict2 = {'fg_on': True}
        cdf = self._exec_ranking(ibs, qaids, daids, cfgdict2)
        results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict2})))

        expt_name = 'foregroundness'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_invariance(self):
        ibs, qaids, daids_list, info_list = self._varied_inputs(
            denc_per_name=[1], extra_dbsize_fracs=[1])
        daids = daids_list[0]
        info = info_list[0]

        cfgdict_list = [
            {'affine_invariance':  True, 'rotation_invariance': False, 'query_rotation_heuristic': False},
            # {'affine_invariance':  True, 'rotation_invariance':  True, 'query_rotation_heuristic': False},
            # {'affine_invariance': False, 'rotation_invariance':  True, 'query_rotation_heuristic': False},
            {'affine_invariance': False, 'rotation_invariance': False, 'query_rotation_heuristic': False},
            {'affine_invariance':  True, 'rotation_invariance': False, 'query_rotation_heuristic':  True},
            {'affine_invariance': False, 'rotation_invariance': False, 'query_rotation_heuristic':  True},
        ]
        results = []
        for cfgdict in cfgdict_list:
            cdf = self._exec_ranking(ibs, qaids, daids, cfgdict)
            results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict})))

        expt_name = 'invar'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_smk(self):
        from ibeis.algo.smk.smk_pipeline import SMKRequest
        # ibs = ibeis.opendb('PZ_MTEST')
        ibs, qaids, daids_list, info_list = self._varied_inputs(
            denc_per_name=[1], extra_dbsize_fracs=[1])
        daids = daids_list[0]
        info = info_list[0]

        config = {'nAssign': 1, 'num_words': 8000, 'sv_on': True}
        qreq_ = SMKRequest(ibs, qaids, daids, config)
        qreq_.ensure_data()
        cm_list = qreq_.execute()
        cm_list = [cm.extend_results(qreq_) for cm in cm_list]
        name_ranks = [cm.get_name_ranks([cm.qnid])[0] for cm in cm_list]
        # Measure rank probabilities
        bins = np.arange(len(qreq_.dnids))
        hist = np.histogram(name_ranks, bins=bins)[0]
        cdf = (np.cumsum(hist) / sum(hist))
        results = [(cdf, ut.update_dict(info.copy(), {'pcfg': config}))]

        expt_name = 'smk'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_nsum(self):
        ibs, qaids, daids_list, info_list = self._varied_inputs(
            denc_per_name=[1, 2, 3], extra_dbsize_fracs=[1])
        cfgdict1 = {'score_method': 'nsum', 'prescore_method': 'nsum', 'query_rotation_heuristic': True}
        cfgdict2 = {'score_method': 'csum', 'prescore_method': 'csum', 'query_rotation_heuristic': True}
        results = []
        for count, (daids, info) in enumerate(zip(daids_list, info_list), start=1):
            cdf1 = self._exec_ranking(ibs, qaids, daids, cfgdict1)
            results.append((cdf1, ut.update_dict(info.copy(), {'pcfg': cfgdict1})))
            cdf2 = self._exec_ranking(ibs, qaids, daids, cfgdict2)
            results.append((cdf2, ut.update_dict(info.copy(), {'pcfg': cfgdict2})))

        expt_name = 'nsum'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_dbsize(self):
        ibs, qaids, daids_list, info_list = self._varied_inputs(
            denc_per_name=[1, 2], extra_dbsize_fracs=[0, 1.0])
        cfgdict = {
            'query_rotation_heuristic': True,
        }
        results = []
        for daids, info in zip(daids_list, info_list):
            info = info.copy()
            cdf = self._exec_ranking(ibs, qaids, daids, cfgdict)
            results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict})))

        expt_name = 'dsize'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)
        cdfs, infos = zip(*results)

    def measure_kexpt(self):
        ibs, qaids, daids_list, info_list = self._varied_inputs(
            denc_per_name=[1, 2], extra_dbsize_fracs=[0, 1.0])
        cfg_grid = {
            'query_rotation_heuristic': True,
            'K': [1, 2, 4, 6, 10],
        }
        results = []
        for cfgdict in ut.all_dict_combinations(cfg_grid):
            for daids, info in zip(daids_list, info_list):
                cdf = self._exec_ranking(ibs, qaids, daids, cfgdict)
                results.append((cdf, ut.update_dict(info.copy(), {'pcfg': cfgdict})))

        expt_name = 'kexpt'
        self.expt_results[expt_name] = results
        ut.save_data(join(self.dpath, expt_name + '.pkl'), results)

    def measure_dbstats(self):
        if self.ibs is None:
            self._precollect()

        # self.ibs.print_annot_stats(self.aids_pool)
        annots = self.ibs.annots(self.aids_pool)

        encounters = annots.group(annots.encounter_text)[1]
        nids = ut.take_column(self.ibs._annot_groups(encounters).nids, 0)
        nid_to_enc = ut.group_items(encounters, nids)

        single_encs = {nid: e for nid, e in nid_to_enc.items() if len(e) == 1}
        multi_encs = {nid: e for nid, e in nid_to_enc.items() if len(e) > 1}
        multi_aids = ut.flatten(ut.flatten(multi_encs.values()))

        enc_deltas = []
        for encs_ in nid_to_enc.values():
            a = encs_[0]
            times = a.image_unixtimes_asfloat
            delta = times.max() - times.min()
            enc_deltas.append(delta)
        max_enc_timedelta = max(enc_deltas)
        print('max enc timedelta = %r' % (
            ut.get_unix_timedelta_str(max_enc_timedelta)))

        multi_stats = self.ibs.get_annot_stats_dict(multi_aids)
        multi_stats['enc_per_name']

        enc_info = ut.odict()
        enc_info['species_nice'] = self.species_nice
        enc_info['n_singleton_names'] = len(single_encs)
        enc_info['n_resighted_names'] = len(multi_encs)
        enc_info['n_encounter_per_resighted_name'] = '{:.1f}±{:.1f}'.format(
            *ut.take(multi_stats['enc_per_name'], ['mean', 'std']))
        n_annots_per_enc = ut.lmap(len, encounters)
        enc_info['n_annots_per_encounter'] = '{:.1f}±{:.1f}'.format(
            np.mean(n_annots_per_enc), np.std(n_annots_per_enc))
        enc_info['n_annots'] = sum(n_annots_per_enc)

        # qual_info = ut.odict()
        qual_info = ut.dict_hist(annots.quality_texts)
        qual_info['None'] = qual_info.pop('UNKNOWN', 0)
        qual_info['None'] += qual_info.pop(None, 0)
        qual_info['species_nice'] = self.species_nice

        view_info = ut.dict_hist(annots.yaw_texts)
        view_info['None'] = view_info.pop('UNKNOWN', 0)
        view_info['None'] += view_info.pop(None, 0)
        view_info['species_nice'] = self.species_nice

        info = {
            'enc': enc_info,
            'qual': qual_info,
            'view': view_info,
        }

        expt_name = ut.get_stack_frame().f_code.co_name.replace('measure_', '')
        expt_name = 'dbstats'
        self.expt_results[expt_name] = info
        ut.save_data(join(self.dpath, expt_name + '.pkl'), info)
        return info


@ut.reloadable_class
class Chap3Draw(object):

    @classmethod
    def measure_single(Chap3, expt_name, dbnames):
        """
        CommandLine:
            python -m ibeis.scripts.thesis Chap3.measure_single --expt=nsum --dbs=GZ_Master1,PZ_Master1
            python -m ibeis.scripts.thesis Chap3.measure_single --expt=foregroundness --dbs=GZ_Master1,PZ_Master1

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.scripts.thesis import *  # NOQA
            >>> expt_name = ut.get_argval('--expt', type_=str)
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[])
            >>> Chap3.measure_single(expt_name, dbnames)
        """
        for dbname in dbnames:
            self = Chap3(dbname)
            if self.ibs is None:
                self._precollect()
            getattr(self, 'measure_' + expt_name)()

    @classmethod
    def draw_single(Chap3, expt_name, dbnames):
        """
        CommandLine:
            python -m ibeis.scripts.thesis Chap3.draw_single --expt=nsum --dbs=GZ_Master1,PZ_Master1
            python -m ibeis.scripts.thesis Chap3.draw_single --expt=foregroundness --dbs=GZ_Master1,PZ_Master1 --diskshow
            python -m ibeis.scripts.thesis Chap3.draw_single --expt=kexpt --dbs=GZ_Master1 --diskshow

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.scripts.thesis import *  # NOQA
            >>> expt_name = ut.get_argval('--expt', type_=str)
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[])
            >>> Chap3.draw_single(expt_name, dbnames)
        """
        print('dbnames = %r' % (dbnames,))
        print('expt_name = %r' % (expt_name,))
        for dbname in dbnames:
            self = Chap3(dbname)
            fpath = getattr(self, 'draw_' + expt_name)()
            if ut.get_argflag('--diskshow'):
                ut.startfile(fpath)

    def draw_nsum(self):
        expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        expt_name = 'nsum'
        results = self.ensure_results(expt_name)
        cdfs, infos = list(zip(*results))
        # pcfgs = ut.take_column(infos, 'pcfg')
        alias = {
            'nsum': 'fmech',
            'csum': 'amech',
        }
        labels = [alias[x['pcfg']['score_method']] + ',dpername={}'.format(x['t_dpername']) for x in infos]
        fig = self.plot_cmcs2(cdfs, labels, fnum=1, ymin=.5)
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw_foregroundness(self):
        expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        results = self.ensure_results(expt_name)
        cdf = results[0][0]
        baseline_cdf = results[1][0]
        cdfs = [cdf, baseline_cdf]
        labels = ['fg=F', 'fg=T']
        fig = self.plot_cmcs2(cdfs, labels, fnum=1, ymin=.5)
        fig.set_size_inches([W, H * .6])
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw_kexpt(self):
        expt_name = ut.get_stack_frame().f_code.co_name.replace('draw_', '')
        results = self.ensure_results(expt_name)
        # results = self.expt_results[expt_name]
        cdfs, infos = list(zip(*results))
        pcfgs = ut.take_column(infos, 'pcfg')
        df = pd.DataFrame.from_records(infos)
        df['cdfs'] = cdfs
        df['K'] = ut.take_column(pcfgs, 'K')
        import plottool as pt
        # groups = list(df.groupby(('dsize', 't_denc_pername')))
        df = df[df['K'] != 10]
        groups = list(df.groupby(('dsize')))
        fig = pt.figure(fnum=1)
        pnum_ = pt.make_pnum_nextgen(nCols=2, nSubplots=len(groups))
        for val, df_group in groups:
            # print('---')
            # print(df_group)
            relevant_df = df_group[['K', 'dsize', 't_dpername']]
            relevant_df = relevant_df.rename(columns={'t_dpername': 'dpername'})
            relevant_cfgs = relevant_df.to_dict('records')
            nonvaried_kw, varied_kws = ut.partition_varied_cfg_list(relevant_cfgs)
            labels_ = [ut.get_cfg_lbl(kw)[1:] for kw in varied_kws]
            cdfs_ = df_group['cdfs'].values
            self.plot_cmcs(cdfs_, labels_, fnum=1, pnum=pnum_(), ymin=.5)
            ax = pt.gca()
            ax.set_title(ut.get_cfg_lbl(nonvaried_kw)[1:])
        pt.adjust_subplots(top=.9, bottom=.1, left=.12, right=.9, hspace=.5, wspace=.2)
        fig.set_size_inches([W * 1.5, H])
        fpath = join(self.dpath, expt_name + '.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        return fpath

    def draw(self):
        """
        CommandLine:
            python -m ibeis.scripts.thesis Chap3.draw --dbs=GZ_Master1,PZ_Master1,GIRM_Master1

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.scripts.thesis import *  # NOQA
            >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[dbname])
            >>> for dbname in dbnames:
            >>>     print('dbname = %r' % (dbname,))
            >>>     self = Chap3(dbname)
            >>>     self.draw()
        """
        import plottool as pt
        self.ensure_results()

        mpl.rcParams.update(TMP_RC)

        expt_name = 'baseline'
        if expt_name in self.expt_results:
            baseline_cdf = self.expt_results['baseline'][0][0]
            fig = self.plot_cmcs2([baseline_cdf], ['baseline'], fnum=1)
            fig.set_size_inches([W, H * .6])
            fpath = join(self.dpath, expt_name + '.png')
            vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))

        if not ('PZ' in self.dbname or 'GZ' in self.dbname):
            return

        expt_name = 'foregroundness'
        if expt_name in self.expt_results:
            self.draw_foregroundness()

        expt_name = 'invar'
        if expt_name in self.expt_results:
            ALIAS_KEYS = ut.invert_dict({
                'RI': 'rotation_invariance',
                'AI': 'affine_invariance',
                'QRH': 'query_rotation_heuristic', })
            results = self.expt_results[expt_name]
            results = [(c, i) for c, i in results
                       if not i['pcfg'].get('rotation_invariance', False)]
            cdfs, infos = list(zip(*results))
            pcfgs = ut.take_column(infos, 'pcfg')

            for p in pcfgs:
                del p['rotation_invariance']

            labels = [ut.get_cfg_lbl(ut.map_keys(ALIAS_KEYS, pcfg))[1:] for pcfg in pcfgs]
            labels = ut.lmap(label_alias, labels)
            fig = self.plot_cmcs2(cdfs, labels, fnum=1, ymin=.5)
            fig.set_size_inches([W, H * .6])
            fpath = join(self.dpath, expt_name + '.png')
            vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))

        expt_name = 'smk'
        if expt_name in self.expt_results:
            results = self.expt_results[expt_name]
            cdf = self.expt_results[expt_name][0][0]
            baseline_cdf = self.expt_results['baseline'][0][0]
            cdfs = [cdf, baseline_cdf]
            labels = ['smk', 'baseline']
            fig = self.plot_cmcs2(cdfs, labels, fnum=1)
            fig.set_size_inches([W, H * .6])
            fpath = join(self.dpath, expt_name + '.png')
            vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))

        expt_name = 'nsum'
        if expt_name in self.expt_results:
            self.draw_nsum()

        expt_name = 'kexpt'
        if expt_name in self.expt_results:
            self.draw_kexpt()

    def _exec_ranking(self, ibs, qaids, daids, cfgdict):
        # Execute the ranking algorithm
        qaids = sorted(qaids)
        daids = sorted(daids)
        qreq_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict)
        cm_list = qreq_.execute()
        cm_list = [cm.extend_results(qreq_) for cm in cm_list]
        name_ranks = [cm.get_name_ranks([cm.qnid])[0] for cm in cm_list]
        # Measure rank probabilities
        bins = np.arange(len(qreq_.dnids))
        hist = np.histogram(name_ranks, bins=bins)[0]
        cdf = (np.cumsum(hist) / sum(hist))
        return cdf

    def prepare_cdfs(self, cdfs, labels):
        total = max(map(len, cdfs))
        # Pad length
        cdfs = [np.hstack([c, np.ones(total - len(c))]) for c in cdfs]
        cdfs = np.array(cdfs)
        # Sort so the best is on top
        sortx = np.lexsort(cdfs.T[::-1])[::-1]
        cdfs = cdfs[sortx]
        labels = ut.take(labels, sortx)
        return cdfs, labels

    def plot_cmcs(self, cdfs, labels, fnum=1, pnum=(1, 1, 1), ymin=.4):
        cdfs, labels = self.prepare_cdfs(cdfs, labels)
        # Truncte to 20 ranks
        num_ranks = min(cdfs.shape[-1], 20)
        xdata = np.arange(1, num_ranks + 1)
        cdfs_trunc = cdfs[:, 0:num_ranks]
        label_list = ['%6.2f%% - %s' % (cdf[0] * 100, lbl)
                      for cdf, lbl in zip(cdfs_trunc, labels)]

        # ymin = .4
        num_yticks = (10 - int(ymin * 10)) + 1

        pt.multi_plot(
            xdata, cdfs_trunc, label_list=label_list,
            xlabel='rank', ylabel='match probability',
            use_legend=True, legend_loc='lower right', num_yticks=num_yticks,
            ymax=1, ymin=ymin, ypad=.005, xmin=.9, num_xticks=5,
            xmax=num_ranks + 1 - .5,
            pnum=pnum, fnum=fnum,
            rcParams=TMP_RC,
        )

    def plot_cmcs2(self, cdfs, labels, fnum=1, **kwargs):
        fig = pt.figure(fnum=fnum)
        self.plot_cmcs(cdfs, labels, fnum=fnum, **kwargs)
        pt.adjust_subplots(top=.8, bottom=.2, left=.12, right=.9)
        fig.set_size_inches([W, H])
        return fig

    def draw_time_distri(self):
        """
        CommandLine:
            python -m ibeis.scripts.thesis Chap3.draw_time_distri --dbs=GZ_Master1,PZ_Master1,GIRM_MasterV
            python -m ibeis.scripts.thesis Chap3.draw_time_distri --dbs=GIRM_Master1
            python -m ibeis.scripts.thesis Chap3.draw_time_distri --dbs=GZ_Master1
            python -m ibeis.scripts.thesis Chap3.draw_time_distri --dbs=PZ_Master1
            python -m ibeis.scripts.thesis Chap3.draw_time_distri --dbs=humpbacks_fb

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.scripts.thesis import *  # NOQA
            >>> dbname = ut.get_argval('--db', default='PZ_MTEST')
            >>> dbnames = ut.get_argval('--dbs', type_=list, default=[dbname])
            >>> for dbname in dbnames:
            >>>     print('dbname = %r' % (dbname,))
            >>>     self = Chap3(dbname)
            >>>     self.draw_time_distri()
        """
        import matplotlib as mpl
        mpl.rcParams.update(TMP_RC)
        if self.ibs is None:
            self._precollect()
        ibs = self.ibs
        annots = ibs.annots(self.aids_pool)
        images = ibs.images(set(annots.gids))
        unixtimes_ = images.unixtime_asfloat
        num_nan = np.isnan(unixtimes_).sum()
        num_total = len(unixtimes_)
        unixtimes = unixtimes_[~np.isnan(unixtimes_)]

        mintime = vt.safe_min(unixtimes)
        maxtime = vt.safe_max(unixtimes)
        unixtime_domain = np.linspace(mintime, maxtime, 1000)

        import matplotlib as mpl
        mpl.rcParams.update(TMP_RC)

        from sklearn.neighbors.kde import KernelDensity
        bw = ut.get_argval('--bw', default=None)
        day = 60 * 60 * 24
        if bw is not None:
            pass
        elif 'GIRM' in self.dbname:
            bw = day / 4
        elif 'GZ' in self.dbname:
            bw = day * 30
        elif 'PZ' in self.dbname:
            bw = day * 30
        elif 'humpbacks_fb' in self.dbname:
            bw = day * 30
        else:
            from sklearn.model_selection import RandomizedSearchCV
            space = np.linspace(day, day * 14, 14).tolist()
            grid_params = {'bandwidth': space}
            searcher = ut.partial(RandomizedSearchCV, n_iter=5, n_jobs=8)
            print('Searching for best bandwidth')
            grid = searcher(KernelDensity(kernel='gaussian'), grid_params,
                            cv=2, verbose=0)
            grid.fit(unixtimes[:, None])
            bw = grid.best_params_['bandwidth']
            print('bw = %r' % (bw,))
            print('bw(days) = %r' % (bw / day,))

        kde = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde.fit(unixtimes[:, None])
        log_density = kde.score_samples(unixtime_domain[:, None])
        density = np.exp(log_density)
        ydata = density.copy()
        # emphasize smaller values
        ydata /= ydata.max()
        ydata = np.sqrt(ydata)
        xdata = unixtime_domain
        xdata_ts = ut.lmap(ut.unixtime_to_datetimeobj, xdata)

        pt.multi_plot(xdata_ts, [ydata], label_list=['time'],
                      alpha=.7, fnum=1, pnum=(1, 1, 1), ymin=0, fill=True,
                      marker='', xlabel='Date', ylabel='# images',
                      num_xticks=5,
                      use_legend=False)
        infos = []
        if num_nan > 0:
            infos.append('#nan={}'.format(num_nan))
            infos.append('#total={}'.format(num_total))
        else:
            infos.append('#total={}'.format(num_total))
        text = '\n'.join(infos)
        pt.relative_text(.02, .02, text, halign='left', valign='top')

        ax = pt.gca()
        fig = pt.gcf()
        ax.set_yticks([])
        if False:
            icon = ibs.get_database_icon()
            pt.overlay_icon(icon, coords=(0, 1), bbox_alignment=(0, 1),
                            as_artist=1, max_asize=(100, 200))
        pt.adjust_subplots(top=.9, bottom=.1, left=.12, right=.9)
        fig.set_size_inches([W, H * .4])
        fpath = join(self.dpath, 'timedist.png')
        vt.imwrite(fpath, pt.render_figure_to_image(fig, dpi=DPI))
        if ut.get_argflag('--diskshow'):
            ut.startfile(fpath)


@ut.reloadable_class
class Chap3(Chap3Inputs, Chap3Agg, Chap3Draw, Chap3Measures, Chap3Commands):
    pass


# @ut.reloadable_class
class Chap4(object):
    """
    Collect data from experiments to visualize

    Ignore:
        >>> from ibeis.scripts.thesis import *
        >>> fpath = ut.glob(ut.truepath('~/Desktop/mtest_plots'), '*.pkl')[0]
        >>> self = ut.load_data(fpath)
    """

    @classmethod
    def collect(Chap4, defaultdb):
        r"""
        CommandLine:
            python -m ibeis.scripts.thesis Chap4.collect --db PZ_PB_RF_TRAIN
            python -m ibeis.scripts.thesis Chap4.collect --db PZ_MTEST
            python -m ibeis.scripts.thesis Chap4.collect

            python -m ibeis.scripts.thesis Chap4.collect --db GZ_Master1

        Example:
            >>> from ibeis.scripts.thesis import *
            >>> defaultdb = 'PZ_PB_RF_TRAIN'
            >>> #defaultdb = 'GZ_Master1'
            >>> defaultdb = 'PZ_MTEST'
            >>> self = Chap4.collect(defaultdb)
            >>> self.draw()
        """
        pblm = OneVsOneProblem.from_empty(defaultdb)
        data_key = pblm.default_data_key
        clf_key = pblm.default_clf_key
        pblm.eval_task_keys = ['match_state', 'photobomb_state']
        pblm.eval_data_keys = [data_key]
        pblm.eval_clf_keys = [clf_key]
        pblm.setup_evaluation()

        # pblm.evaluate_classifiers()
        ibs = pblm.infr.ibs
        pblm.samples.print_info()

        species_code = ibs.get_database_species(pblm.infr.aids)[0]
        if species_code == 'zebra_plains':
            species = 'Plains Zebras'
        if species_code == 'zebra_grevys':
            species = 'Grévy\'s Zebras'
        dbcode = '{}_{}'.format(ibs.dbname, len(pblm.samples))

        self = Chap4()
        self.eval_task_keys = pblm.eval_task_keys
        self.species = species
        self.dbcode = dbcode
        self.data_key = data_key
        self.clf_key = clf_key

        self.task_nice_lookup = {
            'match_state': ibs.const.REVIEW.CODE_TO_NICE,
            'photobomb_state': {
                'pb': 'Phototomb',
                'notpb': 'Not Phototomb',
            }
        }

        if ibs.dbname == 'PZ_MTEST':
            self.dpath = ut.truepath('~/Desktop/' + self.dbcode)
        else:
            base = '~/latex/crall-thesis-2017/figures_pairclf'
            self.dpath = ut.truepath(base + '/' + self.dbcode)
        self.dpath = pathlib.Path(self.dpath)
        ut.ensuredir(self.dpath)

        #-----------
        # COLLECTION
        #-----------
        task_key = 'match_state'
        if task_key in pblm.eval_task_keys:
            self.build_importance_data(pblm, task_key)
            self.build_roc_data_positive(pblm)
            self.build_score_freq_positive(pblm)
            self.build_hard_cases(pblm, task_key, num_top=4)
            self.build_metrics(pblm, task_key)

        task_key = 'photobomb_state'
        if task_key in pblm.eval_task_keys:
            # self.build_roc_data_photobomb(pblm)
            self.build_importance_data(pblm, task_key)
            self.build_metrics(pblm, task_key)

        fname = 'collected_data.pkl'
        ut.save_data(str(self.dpath.joinpath(fname)), self)
        return self

    def __init__(self):
        self.dpath = ut.truepath('~/latex/crall-thesis-2017/figures_pairclf')
        self.dpath = pathlib.Path(self.dpath)
        self.species = None
        self.dbcode = None
        self.data_key = None
        self.clf_key = None
        # info
        self.eval_task_keys = None
        self.task_importance = {}
        self.task_rocs = {}
        self.hard_cases = {}
        self.task_confusion = {}
        self.task_metrics = {}
        self.task_nice_lookup = None

        self.score_hist_lnbnn = None
        self.score_hist_pos = None

    def draw(self):
        task_key = 'photobomb_state'
        if task_key in self.eval_task_keys:
            self.write_importance(task_key)
            self.write_metrics(task_key)

        task_key = 'match_state'
        if task_key in self.eval_task_keys:

            # self.build_score_freq_positive(pblm)
            self.draw_class_score_hist()
            self.draw_roc(task_key)

            self.draw_wordcloud(task_key)
            self.write_importance(task_key)
            self.write_metrics(task_key)

            if not ut.get_argflag('--nodraw'):
                self.draw_hard_cases(task_key)

    def build_metrics(self, pblm, task_key):
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        res.augment_if_needed()
        pred_enc = res.clf_probs.argmax(axis=1)
        y_pred = pred_enc
        y_true = res.y_test_enc
        sample_weight = res.sample_weight
        target_names = res.class_names

        from ibeis.scripts import sklearn_utils
        metric_df, confusion_df = sklearn_utils.classification_report2(
            y_true, y_pred, target_names, sample_weight, verbose=False)
        self.task_confusion[task_key] = confusion_df
        self.task_metrics[task_key] = metric_df
        return ut.partial(self.write_metrics, task_key)

    def write_metrics(self, task_key='match_state'):
        """
        CommandLine:
            python -m ibeis.scripts.thesis Chap4.write_metrics --db PZ_PB_RF_TRAIN --task-key=match_state
            python -m ibeis.scripts.thesis Chap4.write_metrics --db GZ_Master1 --task-key=match_state

        Example:
            >>> from ibeis.scripts.thesis import *
            >>> kwargs = ut.argparse_funckw(Chap4.write_metrics)
            >>> defaultdb = 'GZ_Master1'
            >>> defaultdb = 'PZ_PB_RF_TRAIN'
            >>> self, pblm = precollect(defaultdb)
            >>> task_key = kwargs['task_key']
            >>> self.build_metrics(pblm, task_key)
            >>> self.write_metrics(task_key)
        """
        df = self.task_confusion[task_key]
        df = df.rename_axis(self.task_nice_lookup[task_key], 0)
        df = df.rename_axis(self.task_nice_lookup[task_key], 1)
        df.index.name = None
        df.columns.name = None

        latex_str = df.to_latex(
            float_format=lambda x: '' if np.isnan(x) else str(int(x)),
        )
        sum_pred = df.index[-1]
        sum_real = df.columns[-1]
        latex_str = latex_str.replace(sum_pred, '$\sum$ predicted')
        latex_str = latex_str.replace(sum_real, '$\sum$ real')
        colfmt = '|l|' + 'r' * (len(df) - 1) + '|l|'
        newheader = '\\begin{tabular}{%s}' % (colfmt,)
        latex_str = '\n'.join([newheader] + latex_str.split('\n')[1:])
        lines = latex_str.split('\n')
        lines = lines[0:-4] + ['\\midrule'] + lines[-4:]
        latex_str = '\n'.join(lines)
        latex_str = latex_str.replace('midrule', 'hline')

        fname = 'confusion_{}.tex'.format(task_key)
        print(latex_str)
        ut.write_to(str(self.dpath.joinpath(fname)), latex_str)
        # sum_real = '\\sum real'

        df = self.task_metrics[task_key]
        df = df.rename_axis(self.task_nice_lookup[task_key], 0)
        df = df.drop(['markedness', 'bookmaker'], axis=1)
        df.index.name = None
        df.columns.name = None
        df['support'] = df['support'].astype(np.int)
        latex_str = df.to_latex(
            float_format=lambda x: '%.2f' % (x)
        )
        lines = latex_str.split('\n')
        lines = lines[0:-4] + ['\\midrule'] + lines[-4:]
        latex_str = '\n'.join(lines)
        print(latex_str)
        fname = 'eval_metrics_{}.tex'.format(task_key)
        ut.write_to(str(self.dpath.joinpath(fname)), latex_str)

    def write_importance(self, task_key):
        # Print info for latex table
        importances = self.task_importance[task_key]
        vals = importances.values()
        items = importances.items()
        top_dims = ut.sortedby(items, vals)[::-1]
        lines = []
        for k, v in top_dims[:5]:
            k = feat_alias(k)
            k = k.replace('_', '\\_')
            lines.append('{} & {:.4f} \\\\'.format(k, v))
        latex_str = '\n'.join(ut.align_lines(lines, '&'))

        fname = 'feat_importance_{}.tex'.format(task_key)
        ut.write_to(str(self.dpath.joinpath(fname)), latex_str)

        print('TOP 5 importances for ' + task_key)
        print('# of dimensions: %d' % (len(importances)))
        print()

    def build_importance_data(self, pblm, task_key):
        self.task_importance[task_key] = pblm.feature_importance(task_key=task_key)

    def build_score_freq_positive(self, pblm):
        task_key = 'match_state'
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        y = res.target_bin_df[POSTV]
        scores = res.probs_df[POSTV]
        bins = np.linspace(0, 1, 100)
        pos_freq = np.histogram(scores[y], bins)[0]
        neg_freq = np.histogram(scores[~y], bins)[0]
        pos_freq = pos_freq / pos_freq.sum()
        neg_freq = neg_freq / neg_freq.sum()
        freqs = {'bins': bins, 'pos_freq': pos_freq, 'neg_freq': neg_freq}
        self.score_hist_pos = freqs

        scores = pblm.samples.simple_scores['score_lnbnn_1vM']
        y = pblm.samples[task_key].indicator_df[POSTV].loc[scores.index]
        # Get 95% of the data at least
        maxbin = scores[scores.argsort()][-max(1, int(len(scores) * .05))]
        bins = np.linspace(0, max(maxbin, 10), 100)
        pos_freq = np.histogram(scores[y], bins)[0]
        neg_freq = np.histogram(scores[~y], bins)[0]
        pos_freq = pos_freq / pos_freq.sum()
        neg_freq = neg_freq / neg_freq.sum()
        freqs = {'bins': bins, 'pos_freq': pos_freq, 'neg_freq': neg_freq}
        self.score_hist_lnbnn = freqs

    def build_roc_data_positive(self, pblm):
        task_key = 'match_state'
        target_class = POSTV
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        c2 = pblm.simple_confusion('score_lnbnn_1vM', task_key=task_key)
        c3 = res.confusions(target_class)
        self.task_rocs[task_key] = {
            'target_class': target_class,
            'curves': [
                {'label': 'LNBNN', 'fpr': c2.fpr, 'tpr': c2.tpr, 'auc': c2.auc},
                {'label': 'learned', 'fpr': c3.fpr, 'tpr': c3.tpr, 'auc': c3.auc},
            ]
        }

    def build_roc_data_photobomb(self, pblm):
        task_key = 'photobomb_state'
        target_class = 'pb'
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        c1 = res.confusions(target_class)
        self.task_rocs[task_key] = {
            'target_class': target_class,
            'curves': [
                {'label': 'learned', 'fpr': c1.fpr, 'tpr': c1.tpr, 'auc': c1.auc},
            ]
        }

    def build_hard_cases(self, pblm, task_key, num_top=2):
        """ Find a failure case for each class """
        res = pblm.task_combo_res[task_key][self.clf_key][self.data_key]
        case_df = res.hardness_analysis(pblm.samples, pblm.infr)
        # group = case_df.sort_values(['real_conf', 'easiness'])
        case_df = case_df.sort_values(['easiness'])

        # failure_cases = case_df[(case_df['real_conf'] > 0) & case_df['failed']]
        failure_cases = case_df[case_df['failed']]
        if len(failure_cases) == 0:
            print('No reviewed failures exist. Do pblm.qt_review_hardcases')

        cases = []
        for (pred, real), group in failure_cases.groupby(('pred', 'real')):
            # Prefer examples we have manually reviewed before
            group = group.sort_values(['real_conf', 'easiness'])
            for idx in range(min(num_top, len(group))):
                case = group.iloc[idx]
                edge = tuple(ut.take(case, ['aid1', 'aid2']))
                cases.append({
                    'edge': edge,
                    'real': res.class_names[real],
                    'pred': res.class_names[pred],
                    'probs': res.probs_df.loc[edge]
                })

        # Augment cases with their one-vs-one matches
        infr = pblm.infr
        config = pblm.hyper_params['vsone_match'].asdict()
        config.update(pblm.hyper_params['vsone_kpts'])
        edges = [case['edge'] for case in cases]
        matches = infr._exec_pairwise_match(edges, config)
        for case, match in zip(cases, matches):
            # TODO: decouple the match from the database
            # store its chip fpath and other required info
            case['match'] = match

        self.hard_cases[task_key] = cases

    def draw_hard_cases(self, task_key):
        """ draw hard cases with and without overlay """
        subdir = 'cases_{}'.format(task_key)
        dpath = self.dpath.joinpath(subdir)
        ut.ensuredir(dpath)
        code_to_nice = self.task_nice_lookup[task_key]

        for case in ut.ProgIter(self.hard_cases[task_key], 'draw hard case'):
            aid1, aid2 = case['edge']
            real_name = case['real']
            pred_name = case['pred']
            match = case['match']
            real_nice, pred_nice = ut.take(code_to_nice,
                                           [real_name, pred_name])
            fname = 'fail_{}_{}_{}_{}'.format(real_nice, pred_nice, aid1, aid2)
            # Draw case
            probs = case['probs'].to_dict()
            order = list(code_to_nice.values())
            order = ut.setintersect(order, probs.keys())
            probs = ut.map_dict_keys(code_to_nice, probs)
            probstr = ut.repr2(probs, precision=2, strkeys=True, nobr=True,
                               key_order=order)
            xlabel = 'real={}, pred={},\n{}'.format(real_nice, pred_nice,
                                                    probstr)
            fig = pt.figure(fnum=1, clf=True)
            ax = pt.gca()
            # Draw with feature overlay
            match.show(ax, vert=False,
                       heatmask=True,
                       show_lines=False,
                       show_ell=False,
                       show_ori=False,
                       show_eig=False,
                       # ell_alpha=.3,
                       modifysize=True)
            ax.set_xlabel(xlabel)
            # fpath = str(dpath.joinpath(fname + '_overlay.jpg'))
            fpath = str(dpath.joinpath(fname + '.jpg'))
            self.savefig(fig, fpath)
            # Draw without feature overlay
            # ax.cla()
            # match.show(ax, vert=False, overlay=False, modifysize=True)
            # ax.set_xlabel(xlabel)
            # fpath = str(dpath.joinpath(fname + '.jpg'))
            # self.savefig(fig, fpath)

    def _draw_score_hist(self, freqs, xlabel, fnum):
        """ helper """
        bins, freq0, freq1 = ut.take(freqs, ['bins', 'neg_freq', 'pos_freq'])
        width = np.diff(bins)[0]
        xlim = (bins[0] - (width / 2), bins[-1] + (width / 2))
        fig = pt.multi_plot(
            bins, (freq0, freq1), label_list=('negative', 'positive'),
            color_list=(pt.FALSE_RED, pt.TRUE_BLUE),
            kind='bar', width=width, alpha=.7, edgecolor='none',
            xlabel=xlabel, ylabel='frequency', fnum=fnum, pnum=(1, 1, 1),
            rcParams=TMP_RC, stacked=True,
            ytickformat='%.2f', xlim=xlim,
            # title='LNBNN positive separation'
        )
        pt.adjust_subplots(top=.8, bottom=.2, left=.12, right=.9)
        fig.set_size_inches([W, H])
        return fig

    def draw_class_score_hist(self):
        """ Plots distribution of positive and negative scores """
        freqs = self.score_hist_pos
        fig1 = self._draw_score_hist(freqs, 'positive probability', 1)

        freqs = self.score_hist_lnbnn
        fig2 = self._draw_score_hist(freqs, 'LNBNN score', 2)

        fname = 'score_hist_pos_{}.png'.format(self.data_key)
        self.savefig(fig1, str(self.dpath.joinpath(fname)))

        fname = 'score_hist_lnbnn.png'
        self.savefig(fig2, str(self.dpath.joinpath(fname)))

    def draw_roc(self, task_key):
        mpl.rcParams.update(TMP_RC)

        roc_data = self.task_rocs[task_key]

        fig = pt.figure(fnum=1)  # NOQA
        ax = pt.gca()
        for data in roc_data['curves']:
            ax.plot(data['fpr'], data['tpr'],
                    label='%s AUC=%.2f' % (data['label'], data['auc']))
        ax.set_xlabel('false positive rate')
        ax.set_ylabel('true positive rate')
        # ax.set_title('%s ROC for %s' % (target_class.title(), self.species))
        ax.legend()
        pt.adjust_subplots(top=.8, bottom=.2, left=.12, right=.9)
        fig.set_size_inches([W, H])

        fname = 'roc_{}.png'.format(task_key)
        self.savefig(fig, str(self.dpath.joinpath(fname)))

    def draw_wordcloud(self, task_key):
        import plottool as pt
        importances = ut.map_keys(feat_alias, self.task_importance[task_key])

        fig = pt.figure(fnum=1)
        pt.wordcloud(importances, ax=fig.axes[0])

        fname = 'wc_{}.png'.format(task_key)
        fig_fpath = str(self.dpath.joinpath(fname))
        self.savefig(fig, fig_fpath)

    def savefig(self, fig, fpath):
        image = pt.render_figure_to_image(fig, dpi=DPI)
        # image = vt.clipwhite(image)
        vt.imwrite(fpath, image)


def label_alias(k):
    k = k.replace('True', 'T')
    k = k.replace('False', 'F')
    return k


def feat_alias(k):
    # presentation values for feature dimension
    k = k.replace('weighted_', 'wgt_')
    k = k.replace('norm_x', 'x')
    k = k.replace('norm_y', 'y')
    k = k.replace('yaw', 'view')
    return k


def test_mcc():
    num = 100
    xdata = np.linspace(0, 1, num * 2)
    ydata = np.linspace(1, -1, num * 2)
    pt.plt.plot(xdata, ydata, '--k',
                label='linear')

    y_true = [1] * num + [0] * num
    y_pred = y_true[:]
    import sklearn.metrics
    xs = []
    for i in range(0, len(y_true)):
        y_pred[-i] = 1 - y_pred[-i]
        xs.append(sklearn.metrics.matthews_corrcoef(y_true, y_pred))

    import plottool as pt
    pt.plot(xdata, xs, label='change one class at a time')

    y_true = ut.flatten(zip([1] * num, [0] * num))
    y_pred = y_true[:]
    import sklearn.metrics
    xs = []
    for i in range(0, len(y_true)):
        y_pred[-i] = 1 - y_pred[-i]
        xs.append(sklearn.metrics.matthews_corrcoef(y_true, y_pred))

    pt.plot(xdata, xs, label='change classes evenly')
    pt.gca().legend()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.scripts.thesis
        python -m ibeis.scripts.thesis --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
