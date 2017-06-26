# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
from os.path import basename, join, splitext, exists
import six
import numpy as np
import utool as ut
import matplotlib as mpl
from ibeis.algo.graph.state import POSTV, NEGTV, INCMP  # NOQA
(print, rrr, profile) = ut.inject2(__name__)

DPI = 300

# TMP_RC = {
#     'axes.titlesize': 12,
#     'axes.labelsize': 12,
#     'font.family': 'DejaVu Sans',
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     # 'legend.fontsize': 18,
#     # 'legend.alpha': .8,
#     'legend.fontsize': 12,
#     'legend.facecolor': 'w',
# }

TMP_RC = {
    'axes.titlesize': 12,
    'axes.labelsize': ut.get_argval('--labelsize', default=12),
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

W, H = 7.4375, 3.0


@ut.reloadable_class
class DBInputs(object):
    def __init__(self, dbname=None):
        self.ibs = None
        self.expt_results = {}
        self.dbname = dbname
        self.dname = None
        self.dpath = None
        if 'GZ' in dbname:
            self.species_nice = "Grévy's zebras"
        if 'PZ' in dbname:
            self.species_nice = "plains zebras"
        if 'GIRM' in dbname:
            self.species_nice = "Masai giraffes"
        if 'humpback' in dbname:
            self.species_nice = "humpbacks"
            # self.species_nice = "humpback whales"
        if dbname is not None:
            # self.dpath = join(self.base_dpath, self.dbname)
            computer_id = ut.get_argval('--comp', default=ut.get_computer_name())
            self.dname = self.dbname + '_' + computer_id
            self.orig_dpath = join(self.base_dpath, 'configured', self.dname)
            print('self.orig_dpath = %r' % (self.orig_dpath,))
            self.dpath = self.orig_dpath
            # ut.ensuredir(self.dpath)

    def _setup_links(self, cfg_prefix, config=None):
        # Setup directory
        from os.path import expanduser
        assert self.dname is not None

        cfgstr = ut.repr3(config)
        hashid = ut.hash_data(cfgstr)[0:6]
        dbcode = '{}_{}_{}'.format(self.dname, cfg_prefix, hashid)

        dpath = expanduser(join(self.base_dpath, 'configured', dbcode))
        link1 = expanduser(join(self.base_dpath, 'configured', self.dname))

        link2 = expanduser(join(self.base_dpath, self.dbname))

        ut.ensuredir(dpath)
        self.real_dpath = dpath
        print('self.real_dpath = %r' % (self.real_dpath,))
        self.dpath = self.real_dpath

        for link in [link1, link2]:
            try:
                self.link = ut.symlink(dpath, link, overwrite=True)
            except Exception:
                if exists(dpath):
                    newpath = ut.non_existing_path(dpath, suffix='_old')
                    ut.move(link, newpath)
                    self.link = ut.symlink(dpath, link)

        ut.writeto(join(self.dpath, 'info.txt'), cfgstr)

    def ensure_setup(self):
        if self.ibs is None:
            self._setup()

    def ensure_results(self, expt_name=None):
        """
        Subclasses must obey the measure_<expt_name>, draw_<expt_name> contract
        """
        if expt_name is None and exists(self.dpath):
            # Load all
            fpaths = ut.glob(str(self.dpath), '*.pkl')
            expt_names = [splitext(basename(fpath))[0] for fpath in fpaths]
            for fpath, expt_name in zip(fpaths, expt_names):
                self.expt_results[expt_name] = ut.load_data(fpath)
        else:
            # expt_name = splitext(basename(fpath))[0]
            fpath = join(str(self.dpath), expt_name + '.pkl')
            if not exists(fpath):
                ut.cprint('Experiment results {} do not exist'.format(expt_name), 'red')
                ut.cprint('First re-setup to check if it is a path issue', 'red')
                if self.ibs is None:
                    self._precollect()
                ut.cprint('Checking new fpath', 'yellow')
                fpath = join(str(self.dpath), expt_name + '.pkl')
                print('fpath = %r' % (fpath,))
                if not exists(fpath):
                    ut.cprint('Results still missing need to re-measure', 'red')
                    # assert False
                    self._setup()
                    getattr(self, 'measure_' + expt_name)()
                else:
                    ut.cprint('Re-setup fixed it', 'gren')
            else:
                print('Experiment results {} exist'.format(expt_name))
            self.expt_results[expt_name] = ut.load_data(fpath)
            return self.expt_results[expt_name]

    @classmethod
    def measure(ChapX, expt_name, dbnames, *args):
        """
        CommandLine:
            python -m ibeis Chap3.measure all --dbs=GZ_Master1
            python -m ibeis Chap3.measure all --dbs=PZ_Master1

            python -m ibeis Chap3.measure nsum --dbs=GZ_Master1,PZ_Master1
            python -m ibeis Chap3.measure foregroundness --dbs=GZ_Master1,PZ_Master1

        # Example:
        #     >>> # Script
        #     >>> from ibeis.scripts.thesis import *  # NOQA
        #     >>> expt_name = ut.get_argval('--expt', type_=str, pos=1)
        #     >>> dbnames = ut.get_argval(('--dbs', '--db'), type_=list, default=[])
        #     >>> ChapX.measure(expt_name, dbnames)
        """
        print('expt_name = %r' % (expt_name,))
        print('dbnames = %r' % (dbnames,))
        print('args = %r' % (args,))
        dbnames = ut.smart_cast(dbnames, list)
        for dbname in dbnames:
            self = ChapX(dbname)
            if self.ibs is None:
                self._setup()
                # self._precollect()
            if expt_name == 'all':
                self.measure_all(*args)
            else:
                getattr(self, 'measure_' + expt_name)(*args)

    @classmethod
    def draw(ChapX, expt_name, dbnames, *args):
        """
        CommandLine:
            python -m ibeis Chap3.draw nsum --dbs=GZ_Master1,PZ_Master1
            python -m ibeis Chap3.draw foregroundness --dbs=GZ_Master1,PZ_Master1 --diskshow
            python -m ibeis Chap3.draw kexpt --dbs=GZ_Master1 --diskshow

            python -m ibeis Chap4.draw importance GZ_Master1

            python -m ibeis Chap4.draw hard_cases GZ_Master1,PZ_Master1 match_state,photobomb_state
            --diskshow

        # Example:
        #     >>> # Script
        #     >>> from ibeis.scripts.thesis import *  # NOQA
        #     >>> expt_name = ut.get_argval('--expt', type_=str, pos=1)
        #     >>> dbnames = ut.get_argval(('--dbs', '--db'), type_=list, default=[])
        #     >>> Chap3.draw(expt_name, dbnames)
        """
        print('expt_name = %r' % (expt_name,))
        print('dbnames = %r' % (dbnames,))
        print('args = %r' % (args,))
        dbnames = ut.smart_cast(dbnames, list)

        if len(dbnames) > 1:
            # parallelize drawing tasks
            from concurrent import futures
            multi_args = [ut.smart_cast(a, list) for a in args]
            with futures.ProcessPoolExecutor(max_workers=6) as executor:
                list(futures.as_completed([
                    executor.submit(ChapX.draw_serial, expt_name, *fsargs)
                    for fsargs in ut.product(dbnames, *multi_args)
                ]))
            print('\n\n Completed multiple tasks')
        else:
            ChapX.draw_serial(expt_name, dbnames, *args)

    @classmethod
    def draw_serial(ChapX, expt_name, dbnames, *args):
        dbnames = ut.smart_cast(dbnames, list)
        mpl.rcParams.update(TMP_RC)

        for dbname in dbnames:
            self = ChapX(dbname)
            if expt_name == 'all':
                self.draw_all()
            else:
                draw_func = getattr(self, 'draw_' + expt_name, None)
                if draw_func is None:
                    draw_func = getattr(self, 'write_' + expt_name, None)
                if draw_func is None:
                    raise ValueError('Cannot find a way to draw ' + expt_name)
                fpath = draw_func(*args)
                if ut.get_argflag('--diskshow'):
                    if isinstance(fpath, six.text_type):
                        ut.startfile(fpath)
                    elif fpath is not None:
                        fpath_list = fpath
                        for fpath in fpath_list:
                            ut.startfile(fpath)

    @classmethod
    def vd(ChapX):
        """
        CommandLine:
            python -m ibeis Chap3.vd
        """
        ut.vd(ChapX.base_dpath)

    @profile
    def _precollect(self):
        """
        Sets up an ibs object with an aids_pool

        Example:
            >>> from ibeis.scripts.thesis import *
            >>> self = Chap3('humpbacks_fb')
            >>> self = Chap3('GZ_Master1')
            >>> self = Chap3('GIRM_Master1')
            >>> self = Chap3('PZ_MTEST')
            >>> self = Chap3('PZ_PB_RF_TRAIN')
            >>> self = Chap3('PZ_Master1')
            >>> self._precollect()

            >>> from ibeis.scripts.thesis import *
            >>> self = Chap4('PZ_Master1')
            >>> self._precollect()
        """
        import ibeis
        from ibeis.init import main_helpers
        self.dbdir = ibeis.sysres.lookup_dbdir(self.dbname)
        ibs = ibeis.opendb(dbdir=self.dbdir)
        if ibs.dbname.startswith('PZ_PB_RF_TRAIN'):
            aids = ibs.get_valid_aids()
        elif ibs.dbname.startswith('PZ_Master'):
            # PZ_Master is too big to run in full.  Select a smaller sample.
            # Be sure to include photobomb and incomparable cases.
            aids = ibs.filter_annots_general(
                require_timestamp=True, species='primary', is_known=True,
                minqual='poor',
            )
            infr = ibeis.AnnotInference(ibs=ibs, aids=aids)
            infr.reset_feedback('staging', apply=True)
            minority_ccs = find_minority_class_ccs(infr)
            minority_aids = set(ut.flatten(minority_ccs))

            # We need to do our best to select a small sample here
            flags = ['left' in text for text in ibs.annots(aids).yaw_texts]
            left_aids = ut.compress(aids, flags)

            majority_aids = set(ibs.filter_annots_general(
                left_aids, require_timestamp=True, species='primary',
                minqual='poor', require_quality=True, min_pername=2,
                max_pername=15
            ))
            # This produces 5720 annotations
            aids = sorted(majority_aids.union(minority_aids))
        else:
            aids = ibs.filter_annots_general(require_timestamp=True,
                                             is_known=True,
                                             species='primary',
                                             minqual='poor')
        # ibs.print_annot_stats(aids, prefix='P')
        main_helpers.monkeypatch_encounters(ibs, aids, minutes=30)
        print('post monkey patch')
        if False:
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


def find_minority_class_ccs(infr):
    # Finds ccs involved in photobombs and incomparble cases
    pb_edges = [
        edge for edge, tags in infr.gen_edge_attrs('tags')
        if 'photobomb'in tags
    ]
    incomp_edges = list(infr.incomp_graph.edges())
    minority_edges = pb_edges + incomp_edges
    minority_nids = set(infr.node_labels(*set(ut.flatten(minority_edges))))
    minority_ccs = [infr.pos_graph._ccs[nid] for nid in minority_nids]
    return minority_ccs


def test_mcc():
    import plottool as pt
    import sklearn.metrics
    num = 100
    xdata = np.linspace(0, 1, num * 2)
    ydata = np.linspace(1, -1, num * 2)
    pt.plt.plot(xdata, ydata, '--k',
                label='linear')

    y_true = [1] * num + [0] * num
    y_pred = y_true[:]
    xs = []
    for i in range(0, len(y_true)):
        y_pred[-i] = 1 - y_pred[-i]
        xs.append(sklearn.metrics.matthews_corrcoef(y_true, y_pred))

    pt.plot(xdata, xs, label='change one class at a time')

    y_true = ut.flatten(zip([1] * num, [0] * num))
    y_pred = y_true[:]
    xs = []
    for i in range(0, len(y_true)):
        y_pred[-i] = 1 - y_pred[-i]
        xs.append(sklearn.metrics.matthews_corrcoef(y_true, y_pred))

    pt.plot(xdata, xs, label='change classes evenly')
    pt.gca().legend()


class ExpandingSample(ut.NiceRepr):

        #         nid = enc.nids[0]
        #         if len(nid_to_splits[nid]) == 0:
        #             chosen = pyrng.sample(enc.aids, min(len(enc), 2))
        #             nid_to_splits[nid].extend(chosen)

        #     qaids = []
        #     dname_encs = []
        #     confusor_pool = []
        #     for nid, aids_ in nid_to_splits.items():
        #         if len(aids_) < 2:
        #             confusor_pool.extend(aids_)
        #         else:
        #             pyrng.shuffle(aids_)
        #             qaids.append(aids_[0])
        #             dname_encs.append([[aids_[1]]])
        #     confusor_pool = ut.shuffle(confusor_pool, rng=0)
        #     self = ExpandingSample(qaids, dname_encs, confusor_pool)
        #     query_samples.append(self)
        # return query_samples
    def __init__(sample, qaids, dname_encs, confusor_pool):
        sample.qaids = qaids
        sample.dname_encs = dname_encs
        sample.confusor_pool = confusor_pool

    def __nice__(sample):
        denc_pername = ut.lmap(len, sample.dname_encs)
        n_denc_pername = np.mean(denc_pername)
        return 'nQaids={}, nDEncPerName={}, nConfu={}'.format(
            len(sample.qaids), n_denc_pername, len(sample.confusor_pool)
        )

    def expand(sample, denc_per_name=[1], extra_dbsize_fracs=[0]):
        # Vary the number of database encounters in each sample
        target_daids_list = []
        target_info_list_ = []
        for num in denc_per_name:
            dname_encs_ = ut.take_column(sample.dname_encs, slice(0, num))
            dnames_ = ut.lmap(ut.flatten, dname_encs_)
            daids_ = ut.total_flatten(dname_encs_)
            target_daids_list.append(daids_)
            name_lens = ut.lmap(len, dnames_)
            dpername = (name_lens[0] if ut.allsame(name_lens) else
                        np.mean(name_lens))
            target_info_list_.append(ut.odict([
                ('qsize', len(sample.qaids)),
                ('t_n_names', len(dname_encs_)),
                ('t_dpername', dpername),
                ('t_denc_pername', num),
                ('t_dsize', len(daids_)),
            ]))

        # Append confusors to maintain a constant dbsize in each base sample
        dbsize_list = ut.lmap(len, target_daids_list)
        max_dsize = max(dbsize_list)
        n_need = max_dsize - min(dbsize_list)
        n_extra_avail = len(sample.confusor_pool) - n_need
        assert len(sample.confusor_pool) > n_need, 'not enough confusors'
        padded_daids_list = []
        padded_info_list_ = []
        for daids_, info_ in zip(target_daids_list, target_info_list_):
            num_take = max_dsize - len(daids_)
            pad_aids = sample.confusor_pool[:num_take]
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
                extra_aids = sample.confusor_pool[len(sample.confusor_pool) - n:]
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
            print('#qaids = %r' % (len(sample.qaids),))
            print('num_need = %r' % (n_need,))
            print('max_dsize = %r' % (max_dsize,))
        return sample.qaids, daids_list, info_list


def split_tabular(text):
    top, rest = text.split('\\toprule')

    x = rest.split('\\midrule')
    header, body1, rest = x[0], x[1:-1], x[-1]
    # header, *body1, rest = rest.split('\\midrule')

    y = rest.split('\\bottomrule')
    body2, bot = y[0:-1], y[-1]
    # *body2, bot = rest.split('\\bottomrule')

    top = top.strip('\n')
    header = header.strip('\n')
    mid = [b.strip('\n') for b in body1 + body2]
    bot = bot.strip('\n')
    # print(top)
    # print(header)
    # print(body)
    # print(bot)
    parts = (top, header, mid, bot)
    return parts


@ut.reloadable_class
class Tabular(object):
    def __init__(self, data=None, colfmt=None, hline=None, caption='',
                 index=True, escape=True):
        self._data = data
        self.n_cols = None
        self.n_rows = None
        self.df = None
        self.text = None
        self.parts = None
        self.colfmt = colfmt
        self.hline = hline
        self.theadify = False
        self.caption = caption
        self.groupxs = None

        self.n_index_levels = 1
        # pandas options
        self.index = index
        self.escape = escape

    def _rectify_colfmt(self, colfmt=None):
        if colfmt is None:
            colfmt = self.colfmt
        if colfmt == 'numeric':
            assert self.n_cols is not None, 'need ncols for numeric'
            colfmt = 'l' * self.n_index_levels + 'r' * (self.n_cols)
        return colfmt

    def _rectify_text(self, text):
        import re
        text = text.replace('±', '\pm')
        # Put all numbers in math mode
        pat = ut.named_field('num', '[0-9.]+(\\\\pm)?[0-9.]*')
        text2 = re.sub(pat, '$' + ut.bref_field('num') + '$', text)

        # latex_str = re.sub(' -0.00 ', '  0.00 ', latex_str)
        return text2

    def as_text(self):
        if isinstance(self._data, str):
            text = self._data
        elif isinstance(self._data, pd.DataFrame):
            df = self._data
            text = df.to_latex(
                index=self.index, escape=self.escape,
                float_format=lambda x: 'nan' if np.isnan(x) else '%.2f' % (x))
            if self.index:
                self.n_index_levels = len(df.index.names)
            self.n_rows = df.shape[0]
            self.n_cols = df.shape[1]
        text = self._rectify_text(text)
        return text

    def as_parts(self):
        if self.parts is not None:
            return self.parts
        text = self.as_text()
        top, header, mid, bot = split_tabular(text)
        colfmt = self._rectify_colfmt()
        if colfmt is not None:
            top = '\\begin{tabular}{%s}' % (colfmt,)

        if self.theadify:
            import textwrap
            width = self.theadify
            wrapper = textwrap.TextWrapper(width=width, break_long_words=False)

            header_lines = header.split('\n')
            new_lines = []
            for line in header_lines:
                line = line.rstrip('\\')
                headers = [h.strip() for h in line.split('&')]
                headers = ['\\\\'.join(wrapper.wrap(h)) for h in headers]
                headers = [h if h == '{}' else '\\thead{' + h + '}'
                           for h in headers]
                line = ' & '.join(headers) + '\\\\'
                new_lines.append(line)
            new_header = '\n'.join(new_lines)
            header = new_header
        if True:
            groupxs = self.groupxs
            # Put midlines between multi index levels
            if groupxs is None and isinstance(self._data, pd.DataFrame):
                index = self._data.index
                if len(index.names) == 2 and len(mid) == 1:
                    groupxs = ut.group_indices(index.labels[0])[1]
                    # part = '\n\multirow{%d}{*}{%s}\n' % (len(chunk), key,)
                    # part += '\n'.join(['& ' + c for c in chunk])
            if groupxs is not None:
                bodylines = mid[0].split('\n')
                mid = ut.apply_grouping(bodylines, groupxs)
        parts = (top, header, mid, bot)
        return parts

    def as_tabular(self):
        parts = self.as_parts()
        tabular = join_tabular(parts, hline=self.hline)
        return tabular

    def as_table(self, caption=None):
        if caption is None:
            caption = self.caption
        tabular = self.as_tabular()
        table = ut.codeblock(
            r'''
            \begin{{table}}[h]
                \centering
                \caption{{{caption}}}
            ''').format(caption=caption)
        if tabular:
            table += '\n' + ut.indent(tabular)
        table += '\n' + ut.codeblock(
            '''
            \end{{table}}
            ''').format()
        return table


def upper_one(s):
    return s[0].upper() + s[1:]


def join_tabular(parts, hline=False, align=True):
    top, header, mid, bot = parts

    if hline:
        toprule = midrule = botrule = '\\hline'
    else:
        toprule = '\\toprule'
        midrule = '\\midrule'
        botrule = '\\bottomrule'

    ut.flatten(ut.bzip(['a', 'b', 'c'], ['-']))

    top_parts = [top, toprule, header]
    if mid:
        # join midblocks given as lists of lines instead of strings
        midblocks = []
        for m in mid:
            if isinstance(m, str):
                midblocks.append(m)
            else:
                midblocks.append('\n'.join(m))
        mid_parts = ut.flatten(ut.bzip([midrule], midblocks))
    else:
        mid_parts = []
    # middle_parts = ut.flatten(list(ut.bzip(body_parts, ['\\midrule'])))
    bot_parts = [botrule, bot]
    text = '\n'.join(top_parts + mid_parts + bot_parts)
    if align:
        text = ut.align(text, '&', pos=None)
        # text = ut.align(text, r'\\', pos=None)
    return text


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.scripts._thesis_helpers
        python -m ibeis.scripts._thesis_helpers --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
