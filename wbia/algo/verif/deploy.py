# -*- coding: utf-8 -*-
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)  # NOQA
from os.path import join, exists, basename
from wbia.algo.verif import sklearn_utils
from wbia.algo.verif import verifier
import utool as ut

print, rrr, profile = ut.inject2(__name__)


@ut.reloadable_class
class Deployer(object):
    """
    Transforms a OneVsOne problem into a deployable model.
    Registers and loads published models.
    """

    fname_parts = [
        'vsone',
        '{species}',
        '{task_key}',
        '{clf_key}',
        '{n_dims}',
        '{hashid}',
    ]

    fname_fmtstr = '.'.join(fname_parts)

    meta_suffix = '.meta.json'

    publish_info = {
        'remote': 'cthulhu.dyn.wildme.io',
        'path': '/data/public/models/pairclf',
    }

    published = {
        'zebra_grevys': {
            # 'photobomb_state': 'vsone.zebra_grevys.photobomb_state.RF.131.thwzdtnkjcwjqeve.cPkl',
            # 'match_state': 'vsone.zebra_grevys.match_state.RF.131.tranflbhimyzeeqi.cPkl',  # OLD PRE-TRAINED 0
            # 'match_state': 'vsone.zebra_grevys.match_state.RF.131.dlncrbzlpwjyqrdx.cPkl',  # OLD PRE-TRAINED 1
            # 'match_state': 'vsone.zebra_grevys.match_state.RF.131.kukigovqipdrjihg.ggr2.cPkl',  # GGR2 0
            # 'match_state': 'vsone.zebra_grevys.match_state.RF.131.djvqkmyzrjgaudok.ggr2.cPkl',  # GGR2 1
            'match_state': 'vsone.zebra_grevys.match_state.RF.131.qysrjnzuiziikxzp.kaia.cPkl',  # Kaia GZ CAs
        },
        'zebra_plains': {
            'match_state': 'vsone.zebra_plains.match_state.RF.131.eurizlstehqjvlsu.cPkl',  # OLD PRE-TRAINED
        },
        'giraffe_reticulated': {
            # 'match_state': 'vsone.giraffe_reticulated.match_state.RF.107.clvhhvwgwxpflnhu.ggr2.cPkl',  # GGR2 0
            'match_state': 'vsone.giraffe_reticulated.match_state.RF.131.kqbaqnrdyxpjrzjd.ggr2.cPkl',  # GGR2 1
        },
    }

    def __init__(self, dpath='.', pblm=None):
        self.dpath = dpath
        self.pblm = pblm

    def _load_published(self, ibs, species, task_key):
        """
        >>> from wbia.algo.verif.vsone import *  # NOQA
        >>> self = Deployer()
        >>> species = 'zebra_plains'
        >>> task_key = 'match_state'
        """

        base_url = 'https://{remote}/public/models/pairclf'.format(**self.publish_info)

        task_fnames = self.published[species]
        fname = task_fnames[task_key]

        grabkw = dict(appname='wbia', check_hash=False, verbose=0)

        meta_url = base_url + '/' + fname + self.meta_suffix
        meta_fpath = ut.grab_file_url(meta_url, **grabkw)  # NOQA

        deploy_url = base_url + '/' + fname
        deploy_fpath = ut.grab_file_url(deploy_url, **grabkw)

        verif = self._make_verifier(ibs, deploy_fpath, task_key)
        return verif

    def _make_ensemble_verifier(self, task_key, clf_key, data_key):
        pblm = self.pblm
        ibs = pblm.infr.ibs
        data_info = pblm.feat_extract_info[data_key]
        # Hack together an ensemble verifier
        clf_list = pblm.eval_task_clfs[task_key][clf_key][data_key]
        labels = pblm.samples.subtasks[task_key]
        eclf = sklearn_utils.voting_ensemble(clf_list, voting='soft')
        deploy_info = {
            'clf': eclf,
            'metadata': {
                'task_key': task_key,
                'clf_key': 'ensemble({})'.format(data_key),
                'data_key': data_key,
                'class_names': labels.class_names,
                'data_info': data_info,
            },
        }
        verif = verifier.Verifier(ibs, deploy_info)
        return verif

    def _make_verifier(self, ibs, deploy_fpath, task_key):
        """
        Ignore:
            # py3 side
            clf = deploy_info['clf']
            a = clf.estimators_[0]
            b = a.tree_
            ut.save_data('_tree.pkl', b)
            c = b.__getstate__()
            d = c['nodes']
            ut.save_data('_nodes.pkl', d)

            a.estimators_[0].tree_.__getstate__()['nodes']


        Ignore:
            # py2 side
            ut.load_data('_tree.pkl')
            ut.load_data('_nodes.pkl')

            >>> from wbia.algo.verif.vsone import *  # NOQA
            >>> params = dict(sample_method='random')
            >>> pblm = OneVsOneProblem.from_empty('PZ_MTEST', **params)
            >>> pblm.setup(with_simple=False)
            >>> task_key = pblm.primary_task_key
            >>> self = Deployer(dpath='.', pblm=pblm)
            >>> deploy_info = self.deploy()

            a = deploy_info['clf']
            d = a.estimators_[0].tree_.__getstate__()['nodes']


        Ignore:
            I'm having a similar issue when trying to use python2 to load a
            sklearn RandomForestClassifier that I saved in python3. I created a
            MWE.

            In python 3

                import numpy as np
                import pickle
                data = np.array(
                    [( 1, 26, 69,   5.32214928e+00,  0.69562945, 563,  908.,  1),
                     ( 2,  7, 62,   1.74883020e+00,  0.33854101, 483,  780.,  1),
                     (-1, -1, -2,  -2.00000000e+00,  0.76420451,   7,    9., -2),
                     (-1, -1, -2,  -2.00000000e+00,  0.        ,  62,  106., -2)],
                  dtype=[('left_child', '<i8'), ('right_child', '<i8'),
                  ('feature', '<i8'), ('threshold', '<f8'), ('impurity',
                  '<f8'), ('n_node_samples', '<i8'),
                  ('weighted_n_node_samples', '<f8'), ('missing_direction',
                  '<i8')])

                # Save using pickle
                with open('data.pkl', 'wb') as file_:
                    # Use protocol 2 to support python2 and 3
                    pickle.dump(data, file_, protocol=2)

                # Save with numpy directly
                np.save('data.npy', data)

            Then in python 2
                # Load with pickle
                import pickle
                with open('data.pkl', 'rb') as file_:
                    data = pickle.load(file_)
                # This results in `ValueError: non-string names in Numpy dtype unpickling`

                # Load with numpy directly
                data = np.load('data.npy')
                # This works

            However this still doesn't make sklearn play nice between 2 and 3.
            So, how can we get pickle to load this numpy object correctly?
            Here is the fix suggested in the link:

                from lib2to3.fixes.fix_imports import MAPPING
                import sys
                import pickle

                # MAPPING maps Python 2 names to Python 3 names. We want this in reverse.
                REVERSE_MAPPING = {}
                for key, val in MAPPING.items():
                    REVERSE_MAPPING[val] = key

                # We can override the Unpickler and loads
                class Python_3_Unpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module in REVERSE_MAPPING:
                            module = REVERSE_MAPPING[module]
                        __import__(module)
                        mod = sys.modules[module]
                        klass = getattr(mod, name)
                        return klass

                with open('data.pkl', 'rb') as file_:
                    data = Python_3_Unpickler(file_).load()

            This still doesn't work



            https://stackoverflow.com/questions/41720952/unpickle-sklearn-tree-descisiontreeregressor-in-python-2-from-python3

        """
        deploy_info = ut.load_data(deploy_fpath)
        verif = verifier.Verifier(ibs, deploy_info=deploy_info)
        if task_key is not None:
            assert (
                verif.metadata['task_key'] == task_key
            ), 'bad saved clf at fpath={}'.format(deploy_fpath)
        return verif

    def load_published(self, ibs, species):
        task_fnames = self.published[species]
        print('loading published: %r' % (task_fnames,))
        classifiers = {
            task_key: self._load_published(ibs, species, task_key)
            for task_key in task_fnames.keys()
        }
        print('loaded classifiers: %r' % (classifiers,))
        return classifiers

    def find_pretrained(self):
        import glob
        import parse

        fname_fmt = self.fname_fmtstr + '.cPkl'
        task_clf_candidates = ut.ddict(list)
        globstr = self.fname_parts[0] + '.*.cPkl'
        for fpath in glob.iglob(join(self.dpath, globstr)):
            fname = basename(fpath)
            result = parse.parse(fname_fmt, fname)
            if result:
                task_key = result.named['task_key']
                task_clf_candidates[task_key].append(fpath)
        return task_clf_candidates

    def find_latest_remote(self):
        """
        Used to update the published dict

        CommandLine:
            python -m wbia.algo.verif.vsone find_latest_remote

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.verif.vsone import *  # NOQA
            >>> self = Deployer()
            >>> task_clf_names = self.find_latest_remote()
        """
        base_url = 'https://{remote}/public/models/pairclf'.format(**self.publish_info)
        import requests
        import bs4

        resp = requests.get(base_url)
        soup = bs4.BeautifulSoup(resp.text, 'html.parser')
        table = soup.findAll('table')[0]

        def parse_bs_table(table):
            n_columns = 0
            n_rows = 0
            column_names = []
            # Find number of rows and columns
            # we also find the column titles if we can
            for row in table.find_all('tr'):
                td_tags = row.find_all('td')
                if len(td_tags) > 0:
                    n_rows += 1
                    if n_columns == 0:
                        n_columns = len(td_tags)
                # Handle column names if we find them
                th_tags = row.find_all('th')
                if len(th_tags) > 0 and len(column_names) == 0:
                    for th in th_tags:
                        column_names.append(th.get_text())

            # Safeguard on Column Titles
            if len(column_names) > 0 and len(column_names) != n_columns:
                raise Exception('Column titles do not match the number of columns')
            columns = column_names if len(column_names) > 0 else range(0, n_columns)
            import pandas as pd

            df = pd.DataFrame(columns=columns, index=list(range(0, n_rows)))
            row_marker = 0
            for row in table.find_all('tr'):
                column_marker = 0
                columns = row.find_all('td')
                for column in columns:
                    df.iat[row_marker, column_marker] = column.get_text().strip()
                    column_marker += 1
                if len(columns) > 0:
                    row_marker += 1
            return df

        df = parse_bs_table(table)
        # Find all available models
        df = df[df['Name'].map(lambda x: x.endswith('.cPkl'))]
        # df = df[df['Last modified'].map(len) > 0]

        fname_fmt = self.fname_fmtstr + '.cPkl'
        task_clf_candidates = ut.ddict(list)
        import parse

        for idx, row in df.iterrows():
            fname = basename(row['Name'])
            result = parse.parse(fname_fmt, fname)
            if result:
                task_key = result.named['task_key']
                species = result.named['species']
                task_clf_candidates[(species, task_key)].append(idx)

        task_clf_fnames = ut.ddict(dict)
        for key, idxs in task_clf_candidates.items():
            species, task_key = key
            # Find the classifier most recently created
            max_idx = ut.argmax(df.loc[idxs]['Last modified'].tolist())
            fname = df.loc[idxs[max_idx]]['Name']
            task_clf_fnames[species][task_key] = fname

        print('published = ' + ut.repr2(task_clf_fnames, nl=2))
        return task_clf_fnames

    def find_latest_local(self):
        """
        >>> self = Deployer()
        >>> self.find_pretrained()
        >>> self.find_latest_local()
        """
        from os.path import getctime

        task_clf_candidates = self.find_pretrained()
        task_clf_fpaths = {}
        for task_key, fpaths in task_clf_candidates.items():
            # Find the classifier most recently created
            fpath = fpaths[ut.argmax(map(getctime, fpaths))]
            task_clf_fpaths[task_key] = fpath
        return task_clf_fpaths

    def _make_deploy_metadata(self, task_key=None):
        pblm = self.pblm
        if pblm.samples is None:
            pblm.setup()

        if task_key is None:
            task_key = pblm.primary_task_key

        # task_keys = list(pblm.samples.supported_tasks())
        clf_key = pblm.default_clf_key
        data_key = pblm.default_data_key

        # Save the classifie
        data_info = pblm.feat_extract_info[data_key]
        feat_extract_config, feat_dims = data_info

        samples = pblm.samples
        labels = samples.subtasks[task_key]

        edge_hashid = samples.edge_set_hashid()
        label_hashid = samples.task_label_hashid(task_key)
        tasksamp_hashid = samples.task_sample_hashid(task_key)

        annot_hashid = ut.hashid_arr(samples._unique_annots.visual_uuids, 'annots')

        # species = pblm.infr.ibs.get_primary_database_species(
        #     samples._unique_annots.aid)
        species = '+'.join(sorted(set(samples._unique_annots.species)))

        metadata = {
            'tasksamp_hashid': tasksamp_hashid,
            'edge_hashid': edge_hashid,
            'label_hashid': label_hashid,
            'annot_hashid': annot_hashid,
            'class_hist': labels.make_histogram(),
            'class_names': labels.class_names,
            'data_info': data_info,
            'task_key': task_key,
            'species': species,
            'data_key': data_key,
            'clf_key': clf_key,
            'n_dims': len(feat_dims),
            # 'aid_pairs': samples.aid_pairs,
        }

        meta_cfgstr = ut.repr2(metadata, kvsep=':', itemsep='', si=True)
        hashid = ut.hash_data(meta_cfgstr)[0:16]

        deploy_fname = self.fname_fmtstr.format(hashid=hashid, **metadata) + '.cPkl'

        deploy_metadata = metadata.copy()
        deploy_metadata['hashid'] = hashid
        deploy_metadata['fname'] = deploy_fname
        return deploy_metadata, deploy_fname

    def _make_deploy_info(self, task_key=None):
        pblm = self.pblm
        if pblm.samples is None:
            pblm.setup()

        if task_key is None:
            task_key = pblm.primary_task_key

        deploy_metadata, deploy_fname = self._make_deploy_metadata(task_key)
        clf_key = deploy_metadata['clf_key']
        data_key = deploy_metadata['data_key']

        clf = None
        if pblm.deploy_task_clfs:
            clf = pblm.deploy_task_clfs[task_key][clf_key][data_key]
        if not clf:
            pblm.learn_deploy_classifiers([task_key], clf_key, data_key)
            clf = pblm.deploy_task_clfs[task_key][clf_key][data_key]

        deploy_info = {
            'clf': clf,
            'metadata': deploy_metadata,
        }
        return deploy_info

    def ensure(self, task_key):
        _, fname = self._make_deploy_metadata(task_key=task_key)
        fpath = join(self.dpath, fname)
        if exists(fpath):
            deploy_info = ut.load_data(fpath)
            assert bool(deploy_info['clf']), 'must have clf'
        else:
            deploy_info = self.deploy(task_key=task_key)
            assert exists(fpath), 'must now exist'
        verif = verifier.Verifier(self.pblm.infr.ibs, deploy_info=deploy_info)
        assert verif.metadata['task_key'] == task_key, 'bad saved clf at fpath={}'.format(
            fpath
        )
        return verif

    def deploy(self, task_key=None, publish=False):
        """
        Trains and saves a classifier for deployment

        Notes:
            A deployment consists of the following information
                * The classifier itself
                * Information needed to construct the input to the classifier
                    - TODO: can this be encoded as an sklearn pipeline?
                * Metadata concerning what data the classifier was trained with
                * PUBLISH TO /media/hdd/PUBLIC/models/pairclf

        Example:
            >>> from wbia.algo.verif.vsone import *  # NOQA
            >>> params = dict(sample_method='random')
            >>> pblm = OneVsOneProblem.from_empty('PZ_MTEST', **params)
            >>> pblm.setup(with_simple=False)
            >>> task_key = pblm.primary_task_key
            >>> self = Deployer(dpath='.', pblm=pblm)
            >>> deploy_info = self.deploy()

        Ignore:
            pblm.evaluate_classifiers(with_simple=False)
            res = pblm.task_combo_res[pblm.primary_task_key]['RF']['learn(sum,glob)']
        """
        deploy_info = self._make_deploy_info(task_key=task_key)
        deploy_fname = deploy_info['metadata']['fname']

        meta_fname = deploy_fname + self.meta_suffix
        deploy_fpath = join(self.dpath, deploy_fname)
        meta_fpath = join(self.dpath, meta_fname)

        ut.save_json(meta_fpath, deploy_info['metadata'])
        ut.save_data(deploy_fpath, deploy_info)

        if publish:
            user = ut.get_user_name()
            remote_uri = '{user}@{remote}:{path}'.format(user=user, **self.publish_info)

            ut.rsync(meta_fpath, remote_uri + '/' + meta_fname)
            ut.rsync(deploy_fpath, remote_uri + '/' + deploy_fname)
        return deploy_info


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.verif.deploy
        python -m wbia.algo.verif.deploy --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
